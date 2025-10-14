import json
from PIL import Image
import math

# This is the resize function of Qwen2.5-VL
def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
):
    """Rescales the image so that the following conditions are met:
    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def convert_to_qwen25vl_format(bbox, orig_height, orig_width, factor=28, min_pixels=56*56, max_pixels=14*14*4*1280):
    new_height, new_width = smart_resize(orig_height, orig_width, factor, min_pixels, max_pixels)
    scale_w = new_width / orig_width
    scale_h = new_height / orig_height
    
    x1, y1, x2, y2 = bbox
    x1_new = round(x1 * scale_w)
    y1_new = round(y1 * scale_h)
    x2_new = round(x2 * scale_w)
    y2_new = round(y2 * scale_h)
    
    x1_new = max(0, min(x1_new, new_width - 1))
    y1_new = max(0, min(y1_new, new_height - 1))
    x2_new = max(0, min(x2_new, new_width - 1))
    y2_new = max(0, min(y2_new, new_height - 1))
    
    return [x1_new, y1_new, x2_new, y2_new]


def convert(prompt,file_name,output_name='search_sft_w_crop'):
    with open(file_name, 'r') as f:
        data = [json.loads(line) for line in f]

    sft_results = []
    for item in data:
        query=item['query']
        reference_answer=item['reference_answer']
        score=item['eval_result']['score']
        passing=item['eval_result']['passing']
        reference_page=item['meta_info']['reference_page']
        if score != 5:
            continue
        overall_images = []
        conversations = [{
            "role": "user",
            "content": prompt.replace('{question}', query)
        }]
        good_data = True
        for message in item['history']:
            if isinstance(message,dict) and 'think' in message and 'search' in message:
                conversations.append({
                    "role": "assistant",
                    "content": f'<think>{message["think"]}</think>\n<search>{message["search"]}</search>'
                })
            elif isinstance(message,dict) and 'think' in message and 'answer' in message:
                conversations.append({
                    "role": "assistant",
                    "content": f'<think>{message["think"]}</think>\n<answer>{message["answer"]}</answer>'
                })
            elif isinstance(message,dict) and 'think' in message and 'bbox' in message:
                if 'search' not in conversations[-2]['content']:
                    good_data=False
                width, height = Image.open(overall_images[-1]).size
                if width*height > 14*14*4*1280 or width*height < 56*56:
                    good_data=False
                if message['bbox'][0] > width or message['bbox'][1] > height or message['bbox'][2] > width or message['bbox'][3] > height:
                    good_data=False
                bbox_noramal = convert_to_qwen25vl_format(message['bbox'], height, width, max_pixels=512*28*28)
                conversations.append({
                    "role": "assistant",
                    "content": f'<think>{message["think"]}</think>\n<bbox>{str(bbox_noramal)}</bbox>'
                })
            elif isinstance(message,list) and len(message) == 1:
                conversations.append({
                    "role": "user",
                    "content": "<image>"
                })
                overall_images.append(message[0]['image'])
            elif isinstance(message,list) and len(message) == 2:
                conversations.append({
                    "role": "user",
                    "content": "<image>"
                })
                overall_images.append(message[0]['image'])
            else:
                continue
        if not good_data:
            continue

        # if len(overall_images) in [1,2,3,4,5,6]:
        #     sft_results.append(dict(
        #         messages=conversations,
        #         images=overall_images,
        #     ))
    print(len(sft_results))
    with open(f"data/{output_name}.json", "w") as f:
        json.dump(sft_results, f, indent=4, ensure_ascii=False)


prompt_crop = '''Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and user will return the searched results. Every time you retrieve an image, you have the option to crop it to obtain a clearer view, the format for coordinates is <bbox>[x1, y1, x2, y2]</bbox>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}'''
file_name_raw = 'Path to the raw data'
convert(prompt=prompt_crop, file_name=file_name_raw,output_name='search_sft_w_crop')
