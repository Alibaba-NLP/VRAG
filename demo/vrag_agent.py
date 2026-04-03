import base64
import json
import re
import requests
import math
from io import BytesIO

from openai import OpenAI
from PIL import Image, ImageDraw

prompt_ins = '''Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and user will return the searched results. Every time you retrieve an image, you have the option to crop it to obtain a clearer view, the format for coordinates is <bbox>[x1, y1, x2, y2]</bbox>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}
'''

class VRAG:
    def __init__(self, 
                base_url='http://0.0.0.0:8002/v1', 
                search_url='http://0.0.0.0:8001/search',
                generator=True,
                api_key='EMPTY'):
        
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.search_url = search_url

        self.max_pixels = 512 * 28 * 28
        self.min_pixels = 256 * 28 * 28
        self.repeated_nums = 1
        self.max_steps = 10

        self.generator = generator

    def process_image(self, image):
        if isinstance(image, dict):
            image = Image.open(BytesIO(image['bytes']))
        elif isinstance(image, str):
            image = Image.open(image)

        if (image.width * image.height) > self.max_pixels:
            resize_factor = math.sqrt(self.max_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if (image.width * image.height) < self.min_pixels:
            resize_factor = math.sqrt(self.min_pixels / (image.width * image.height))
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height))

        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        byte_stream = BytesIO()
        image.save(byte_stream, format="JPEG")
        byte_array = byte_stream.getvalue()
        base64_encoded_image = base64.b64encode(byte_array)
        base64_string = base64_encoded_image.decode("utf-8")
        base64_qwen = f"data:image;base64,{base64_string}"

        return image, base64_qwen
    
    def search(self, query):
        if isinstance(query, str):
            query = [query]
        search_response = requests.post(self.search_url, json={"queries": query, "top_k": 3})
        search_results = search_response.json()
        image_path_list = [result['image_file'] for result in search_results[0]]
        return image_path_list
    
    def run(self, question):
        self.image_raw = []
        self.image_input = []
        self.image_path = []
        prompt = prompt_ins.format(question=question)
        messages = [dict(
            role="user",
            content=[
                {
                    "type": "text",
                    "text": prompt,
                }
            ]
        )]

        max_steps = self.max_steps
        while True:
            ## assistant
            response = self.client.chat.completions.create(
                model="Qwen/Qwen2.5-VL-7B-Instruct",
                messages=messages,
                stream=False
            )
            response_content = response.choices[0].message.content
            messages.append(dict(
                role="assistant",
                content=[{
                    "type": "text",
                    "text": response_content
                }]
            ))
            ## think
            pattern = r'<think>(.*?)</think>'
            match = re.search(pattern, response_content, re.DOTALL)
            thought = match.group(1)
            if self.generator:
                yield 'think', thought, match.group(0)
            ## opration
            pattern = r'<(search|answer|bbox)>(.*?)</\1>'
            match = re.search(pattern, response_content, re.DOTALL)
            if match:
                raw_content = match.group(0)
                content = match.group(2).strip()  # Return only the content inside the tags
                action = match.group(1)
            else:
                content = ''
                action = None

            ## whether end
            if action == 'answer':
                return 'answer', content, raw_content
            elif max_steps==0:
                return 'answer', 'Sorry, I can not retrieval something about the question.', ''
            elif self.generator:
                yield action, content, raw_content

            ## action
            if action == 'search':
                search_results = self.search(content)
                while len(search_results) > 0:
                    image_path = search_results.pop(0)
                    if self.image_path.count(image_path) >= self.repeated_nums:
                        continue
                    else:
                        self.image_path.append(image_path)
                        break
                
                image_raw = Image.open(image_path)
                image_input, img_base64 = self.process_image(image_raw)
                user_content=[{
                    'type': 'image_url',
                    'image_url': {
                        'url': img_base64
                    }
                }]
                self.image_raw.append(image_raw)
                self.image_input.append(image_input)
                if self.generator:
                    yield 'search_image', self.image_input[-1], raw_content
            elif action == 'bbox':
                bbox = json.loads(content)
                input_w, input_h = self.image_input[-1].size
                raw_w, raw_h = self.image_raw[-1].size
                crop_region_bbox = bbox[0] * raw_w / input_w, bbox[1] * raw_h / input_h, bbox[2] * raw_w / input_w, bbox[3] * raw_h / input_h
                pad_size = 56
                crop_region_bbox = [max(crop_region_bbox[0]-pad_size,0), max(crop_region_bbox[1]-pad_size,0), min(crop_region_bbox[2]+pad_size,raw_w), min(crop_region_bbox[3]+pad_size,raw_h)]
                crop_region = self.image_raw[-1].crop(crop_region_bbox)
                image_input, img_base64 = self.process_image(crop_region)
                user_content=[{
                    'type': 'image_url',
                    'image_url': {
                        'url': img_base64
                    }
                }]
                self.image_raw.append(crop_region)
                self.image_input.append(image_input)

                if self.generator:
                    image_to_draw = self.image_input[-2].copy()
                    draw = ImageDraw.Draw(image_to_draw)
                    draw.rectangle(bbox, outline=(160, 32, 240), width=7)
                    yield 'crop_image', self.image_input[-1], image_to_draw

            max_steps -= 1
            if max_steps == 0:
                user_content.append({
                    'type': 'text',
                    'text': 'please answer the question now with answer in <answer> ... </answer>' 
                })
            messages.append(dict(
                role='user',
                content=user_content
            ))

if __name__ == '__main__':
    agent = VRAG()
    generator = agent.run('How are u?')
    while True:
        print(next(generator))
