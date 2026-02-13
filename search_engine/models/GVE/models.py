import os
import logging
from typing import Optional, Dict
from dataclasses import dataclass
import PIL.Image
from PIL import ImageFile

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist

from transformers.modeling_outputs import ModelOutput
from peft import get_peft_model

from .model_utils import UnifierMnrlLoss, dist_gather_tensor, full_contrastive_scores_and_labels, get_num_images, get_num_videos, get_num_pixels, mean_pool, last_pool
from .qwen25vl import Qwen25VLForEmbedding


PIL.Image.MAX_IMAGE_PIXELS = 933120000
ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger(__name__)


@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    d_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class AutoModelForSentenceEmbedding(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        pooling: str = 'last',
        normalize: bool = True,
        add_pooler: bool = False,
        embedding_dim: Optional[int] = None,
        use_lora=False,
        lora_config=None,
        attn_type=None,
        mrl_loss=False,
        dims=None,
        dim_step=None,
        torch_dtype=torch.bfloat16,
        v2t_loss_weight=0.5,
        **kwargs,
    ):
        super().__init__()

        assert any(_ in model_name_or_path.lower() for _ in ['qwen25vl', 'qwen2.5-vl', 'qwen2.5_vl', 'qwen25-vl', 'qwen25_vl', 'qwen2_5_vl'])
        if torch.cuda.is_available():
            self.lm = Qwen25VLForEmbedding.from_pretrained(model_name_or_path, attn_implementation="flash_attention_2", torch_dtype=torch_dtype, device_map='auto', **kwargs)
        else:
            self.lm = Qwen25VLForEmbedding.from_pretrained(model_name_or_path, torch_dtype=torch_dtype, device_map='cpu', **kwargs)
        self.lm.config.use_cache = False

        self.config = self.lm.config
        self.pooling = pooling
        self.normalize = normalize
        self.add_pooler = add_pooler
        self.pooler = nn.Linear(self.lm.config.hidden_size, embedding_dim or self.lm.config.hidden_size) if add_pooler else nn.Identity()
        self.use_lora = False
        self.attn_type = attn_type
        self.mrl_loss = mrl_loss
        self.dims = dims
        self.dim_step = dim_step
        if self.mrl_loss:
            self.mrl_loss = UnifierMnrlLoss(dims=self.dims, dim_step=self.dim_step)
        self.v2t_loss_weight = v2t_loss_weight if 0 <= v2t_loss_weight <= 1 else 0

        if use_lora:
            self.use_lora = True
            self.lm.enable_input_require_grads()
            self.lm = get_peft_model(self.lm, lora_config)
            self.lm.print_trainable_parameters()
            self.lm.enable_input_require_grads()
            print(self.lm)

        self.extra_linear = None
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

    def _encode_by_sub_batch(self, texts, sub_batch_size=None):
        if sub_batch_size is None:
            return self.encode(texts)
        all_dense_vecs = []
        image_pixel_index = 0
        video_pixel_index = 0
        image_index = 0
        video_index = 0
        for i in range(0, len(texts['input_ids']), sub_batch_size):
            end_inx = min(i + sub_batch_size, len(texts['input_ids']))
            sub_batch_texts = {}
            for k, v in texts.items():
                # NOTE distinguish: pixel_values (image) and pixel_values_videos (video)
                # for image
                if k == 'pixel_values':
                    sub_batch_input_ids = texts['input_ids'][i:end_inx]
                    num_images = get_num_images(sub_batch_input_ids, self.lm.config.image_token_id)  # for indexing image_grid_thw
                    pixel_size = get_num_pixels(sub_batch_input_ids, self.lm.config.image_token_id)  # for indexing pixel_values
                    if pixel_size == 0:
                        continue
                    sub_batch_texts['pixel_values'] = texts['pixel_values'][image_pixel_index:image_pixel_index+pixel_size]
                    image_pixel_index += pixel_size
                    sub_batch_texts['image_grid_thw'] = texts['image_grid_thw'][image_index:image_index+num_images]
                    image_index += num_images
                elif k == 'image_grid_thw':
                    continue
                # for video
                elif k == 'pixel_values_videos':
                    sub_batch_input_ids = texts['input_ids'][i:end_inx]
                    num_videos = get_num_videos(sub_batch_input_ids, self.lm.config.video_token_id)  # for indexing video_grid_thw
                    pixel_size = get_num_pixels(sub_batch_input_ids, self.lm.config.video_token_id)  # for indexing pixel_values_videos
                    if pixel_size == 0:
                        continue
                    sub_batch_texts['pixel_values_videos'] = texts['pixel_values_videos'][video_pixel_index:video_pixel_index+pixel_size]
                    video_pixel_index += pixel_size
                    sub_batch_texts['video_grid_thw'] = texts['video_grid_thw'][video_index:video_index+num_videos]
                    video_index += num_videos
                elif k == 'video_grid_thw':
                    continue
                # others for image or video
                else:
                    sub_batch_texts[k] = v[i:end_inx]
            dense_vecs = self.encode(sub_batch_texts)
            all_dense_vecs.append(dense_vecs)
        dense_vecs = torch.cat(all_dense_vecs, 0).contiguous()
        return dense_vecs

    def encode(self, texts):
        if texts is None:
            return None
        pooling_mask = texts.pop('pooling_mask') if "pooling_mask" in texts else texts['attention_mask']
        if self.attn_type == 'bi':
            texts["is_causal"] = False
        outputs = self.lm(**texts, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        embeddings = self.pool_sentence_embedding(last_hidden_state, pooling_mask)
        if self.add_pooler and self.pooler.weight.dtype != embeddings.dtype:
            self.pooler = self.pooler.to(dtype=embeddings.dtype)
        embeddings = self.pooler(embeddings)
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.contiguous()

    def pool_sentence_embedding(self, last_hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
        if self.pooling == 'mean':
            return mean_pool(last_hidden_state, attention_mask)
        elif self.pooling == 'last':
            return last_pool(last_hidden_state, attention_mask)
        else:
            raise NotImplementedError(f"Currently do not support pooling method: {self.pooling}")

    def forward(
        self,
        query: Dict[str, Tensor] = None,
        doc: Dict[str, Tensor] = None,
        temperature: float = 1.0,
        negatives_x_device: bool = False,
        loss_scale: float = 1.0,
        full_contrastive_loss: bool = True,
        only_self_neg: bool = False,
        sub_batch_size: int = None,
        *args,
        **kwargs
    ):
        q_embeddings = self._encode_by_sub_batch(query, sub_batch_size)  # (batch_size, embedding_dim)
        d_embeddings = self._encode_by_sub_batch(doc, sub_batch_size)

        if q_embeddings is None or d_embeddings is None:  # for grad cache
            return EncoderOutput(q_reps=q_embeddings, d_reps=d_embeddings)

        if negatives_x_device and dist.is_initialized():
            q_embeddings = dist_gather_tensor(q_embeddings)
            d_embeddings = dist_gather_tensor(d_embeddings)

        scores, labels = full_contrastive_scores_and_labels(q_embeddings, d_embeddings, use_all_pairs=full_contrastive_loss, only_self_neg=only_self_neg, check_pos_neg_pair_conflict=True, check_false_neg=False)
        scores /= temperature

        loss = self.cross_entropy(scores, labels) * loss_scale
        return EncoderOutput(q_reps=q_embeddings, d_reps=d_embeddings, scores=scores, loss=loss)

    def gradient_checkpointing_enable(self, **kwargs):
        kwargs['gradient_checkpointing_kwargs'] = {'use_reentrant': False}
        self.lm.gradient_checkpointing_enable(**kwargs)
        self.lm.gradient_checkpointing = True

    def save_pretrained(self, output_path):
        if self.use_lora:
            self.lm.save_pretrained(output_path)
        else:
            self.lm.save_pretrained(output_path, safe_serialization=False)
        if self.add_pooler:
            torch.save(self.pooler.state_dict(), os.path.join(output_path, 'pooler.pt'))

    def load_pretrained(self, output_path):
        self.lm = self.lm.from_pretrained(output_path)
        if self.add_pooler:
            try:
                pooler_states = torch.load(os.path.join(output_path, 'pooler.pt'))
                self.pooer.load_state_dict(pooler_states)
            except FileNotFoundError:
                logger.info("Cannot find pooler.pt at %s", output_path)


class AutoModelForSentenceEmbeddingTriplet(AutoModelForSentenceEmbedding):

    def forward(
        self,
        query: Dict[str, Tensor] = None,
        pos: Dict[str, Tensor] = None,  # renamed from `doc`
        negs: Dict[str, Tensor] = None,  # additonal parameter
        temperature: float = 1.0,
        negatives_x_device: bool = False,
        loss_scale: float = 1.0,
        full_contrastive_loss: bool = True,
        only_self_neg=False,
        sub_batch_size: int = None,
        *args,
        **kwargs
    ):
        q_embeddings = self._encode_by_sub_batch(query, sub_batch_size)  # (batch_size, embedding_dim)
        p_embeddings = self._encode_by_sub_batch(pos, sub_batch_size)  # (batch_size, embedding_dim)
        if len(negs) > 0:
            n_embeddings = self._encode_by_sub_batch(negs, sub_batch_size)  # (batch_size * num_neg, embedding_dim)

        if negatives_x_device and dist.is_initialized():
            q_embeddings = dist_gather_tensor(q_embeddings)
            p_embeddings = dist_gather_tensor(p_embeddings)
            if len(negs) > 0:
                n_embeddings = dist_gather_tensor(n_embeddings)

        if len(negs) > 0:
            d_embeddings = torch.cat([p_embeddings, n_embeddings])
        else:
            d_embeddings = p_embeddings

        scores, labels = full_contrastive_scores_and_labels(
            q_embeddings,
            d_embeddings,
            use_all_pairs=False,
            only_self_neg=only_self_neg,
            check_pos_neg_pair_conflict=True,
            check_false_neg=True,
        )

        t2v_loss = self.cross_entropy(scores / temperature, labels)
        v2t_loss = self.cross_entropy(scores.T / temperature, labels)
        loss = (1 - self.v2t_loss_weight) * t2v_loss + self.v2t_loss_weight * v2t_loss
        loss *= loss_scale

        return EncoderOutput(q_reps=q_embeddings, d_reps=d_embeddings, scores=scores, loss=loss)
