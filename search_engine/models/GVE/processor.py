from qwen_vl_utils import process_vision_info
from .qwen25vl import Qwen2_5_VLProcessor


class VLProcessor(object):

    MIN_PIXELS = 4 * 28 * 28
    MAX_PIXELS = 256 * 28 * 28
    DEFAULT_SIZE = {
        'shortest_edge': MIN_PIXELS,
        'longest_edge': MAX_PIXELS,
    }

    def __init__(self, model_name_or_path, eod_token=None):
        self.processor = Qwen2_5_VLProcessor.from_pretrained(model_name_or_path, size=self.DEFAULT_SIZE)
        self.processor.tokenizer.padding_side = 'right'

    def __call__(self, messages, padding=True, truncation=True, max_length=8192, return_tensors: str = 'pt'):
        texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        # image_inputs, video_inputs = process_vision_info(messages, return_video_kwargs=False)
        # print(video_kwargs)
        return self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            fps=video_kwargs['fps'],
            # fps=1,
            padding=padding,
            truncation=truncation,
            max_length=max_length-1,
            return_tensors=return_tensors
        )
