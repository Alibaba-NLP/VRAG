from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F


class UnifierMnrlLoss(nn.Module):
    def __init__(self, dims=None, dim_step=32, **kwargs):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.dims = dims
        self.dim_step = dim_step
        print('dims', self.dims)

    def forward(self, reps_q, reps_d, temperature=1.0, use_all_pairs=True, only_self_neg=False, check_false_neg=True):
        loss = 0
        dims = self.dims
        dim_step = self.dim_step
        if dims is None and dim_step is None:
            dims = [reps_q.size(-1)]
        elif dims is None:
            dims = list(range(dim_step, reps_q.size(-1), dim_step))
        for dim in dims:
            sub_q = F.normalize(reps_q[:, :dim], p=2, dim=1)
            sub_d = F.normalize(reps_d[:, :dim], p=2, dim=1)
            scores, labels = full_contrastive_scores_and_labels(sub_q, sub_d, use_all_pairs=use_all_pairs, only_self_neg=only_self_neg, check_false_neg=check_false_neg)
            dense_loss = self.cross_entropy(scores / temperature, labels)
            loss += dense_loss
        loss = loss / len(dims)
        return loss


def dist_gather_tensor(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if t is None:
        return None

    t = t.contiguous()
    all_tensors = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(all_tensors, t)

    all_tensors[dist.get_rank()] = t
    all_tensors = torch.cat(all_tensors, dim=0)
    return all_tensors


def full_contrastive_scores_and_labels(
    query: torch.Tensor,
    key: torch.Tensor,
    use_all_pairs: bool = True,
    only_self_neg: bool = False,  # only use hard negatives, not in-batch negatives
    check_pos_neg_pair_conflict: bool = True,
    check_false_neg: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert key.shape[0] % query.shape[0] == 0, f'{key.shape[0]} % {query.shape[0]} > 0'
    if only_self_neg:
        pos_key = key[:query.size(0), :]
        neg_key = key[query.size(0):, :]
        n_passage = neg_key.size(0) // pos_key.size(0)
        neg_key = neg_key.view(query.size(0), n_passage, -1)
        key = torch.cat([pos_key.unsqueeze(1), neg_key], dim=1)
        query = query.unsqueeze(1)
        scores = torch.bmm(query, key.transpose(1, 2)).squeeze(1)
        labels = torch.zeros(query.size(0), device=query.device).long()
        return scores, labels

    labels = torch.arange(0, query.shape[0], dtype=torch.long, device=query.device)

    # batch_size x (batch_size x n_psg)
    qk = torch.mm(query, key.t())

    if check_pos_neg_pair_conflict:
        pos_key = key[:query.size(0), :]
        sim_scores = torch.mm(pos_key, key.t()).detach()
        mask = torch.isclose(sim_scores, torch.ones_like(sim_scores), atol=1e-5)
        mask.fill_diagonal_(0.0)
        qk[mask] = float('-inf')

    if not use_all_pairs:
        if check_false_neg:
            thresholds = torch.diagonal(qk).view(-1, 1) + 0.1
            thresholds = thresholds.detach()
            mask = qk > thresholds
            qk[mask] = float('-inf')
        return qk, labels
    # batch_size x dim
    sliced_key = key.index_select(dim=0, index=labels)
    assert query.shape[0] == sliced_key.shape[0]

    # batch_size x batch_size
    kq = torch.mm(sliced_key, query.t())
    kq.fill_diagonal_(float('-inf'))

    qq = torch.mm(query, query.t())
    qq.fill_diagonal_(float('-inf'))

    kk = torch.mm(sliced_key, sliced_key.t())
    kk.fill_diagonal_(float('-inf'))

    scores = torch.cat([qk, kq, qq, kk], dim=-1)
    if check_false_neg:
        thresholds = torch.diagonal(scores).view(-1, 1) + 0.1
        thresholds = thresholds.detach()
        mask = scores > thresholds
        scores[mask] = float('-inf')

    return scores, labels


def get_num_images(inputs_ids, image_token_id) -> int:
    mask = inputs_ids == image_token_id  # 形状: (B, L)
    shifted_mask = torch.cat([torch.zeros((inputs_ids.size(0), 1), dtype=torch.bool, device=inputs_ids.device), mask[:, :-1]], dim=1)  # 将掩码向右平移一位，并在开头填充 False，以便检测新的图片开始，形状: (B, L)
    new_image_starts = mask & (~shifted_mask)  # 新图片的起始位置是当前 token 为 image_token_id 且前一个 token 不是，形状: (B, L)
    num_images_per_batch = new_image_starts.sum(dim=1)  # 形状: (B,)
    total_images = num_images_per_batch.sum().item()
    return total_images


def get_num_videos(inputs_ids, video_token_id) -> int:
    mask = inputs_ids == video_token_id  # 形状: (B, L)
    shifted_mask = torch.cat([torch.zeros((inputs_ids.size(0), 1), dtype=torch.bool, device=inputs_ids.device), mask[:, :-1]], dim=1)  # 将掩码向右平移一位，并在开头填充 False，以便检测新的图片开始，形状: (B, L)
    new_video_starts = mask & (~shifted_mask)  # 新图片的起始位置是当前 token 为 video_token_id 且前一个 token 不是，形状: (B, L)
    num_videos_per_batch = new_video_starts.sum(dim=1)  # 形状: (B,)
    total_videos = num_videos_per_batch.sum().item()
    return total_videos


def get_num_pixels(input_ids, token_id) -> int:
    return torch.sum(input_ids == token_id).item() * 4


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
    embeddings = last_hidden.sum(dim=1) / torch.clamp(attention_mask.sum(dim=1), min=1e-9)[..., None]
    return embeddings.type_as(last_hidden_state)


def last_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_state[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_state.shape[0]
    return last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
