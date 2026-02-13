import os
import json
from tqdm import tqdm
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from llama_index.core.schema import NodeWithScore, ImageNode
import sys
import dashscope
from tqdm import tqdm
from http import HTTPStatus
import requests
import time
from PIL import Image

image_output_dir = './data/image_crop'
raw_image_dir = './search_engine/corpus/img'
model_scope_key = os.environ.get('DASH_SCOPE_KEY')
search_engine_url = 'http://localhost:8002/search'


prompt_inst = """## Character Introduction
You are an intelligent assistant capable of performing searches and providing precise answers to user queries. You need to think step by step and provide actions. Your thought should be as detailed as you can. When you cannot answer a question, please use search tool to search for more relevant information. If you think that a specific region of the last image could help answer the question, please use crop tool to provide a detailed view of the relevant region. Once you have gathered enough information to answer the question, provide your response immediately.

### Available Tools
1. search:  
   - Collect relevant information based on the query.  
   - Parameters: The keywords or question to search for.  
   - Returns: Search results for query.

2. crop:
   - Crop the last image based on the user's query.
   - Parameters: The crop region of the image.
   - Returns: Cropped image.

3. answer:  
   - Respond directly to the user based on search results.  
   - Parameters: The response to the user's query.  

### Requirements
- Ensure tool usage is precise and queries are well-formulated.
- Provide accurate and well-structured answers to user queries.
- Iterate search attempts if initial results are insufficient.
- Follow the response format.


### Reply Format
You **must** response with the following json format:
When you need to search, you need to provide the search query in the following format:
```
{
    "think": ...,
    "search": ...
}
```
When you need to crop the image, you need to provide the following format:
For tables, charts, or any visual elements, please use bounding boxes to completely encapsulate them.
```
{
    "think": ...,
    "bbox": [x1, y1, x2, y2],
    "description": the cropped content ...
}
```
When you have gathered enough information to answer the question, provide your response immediately:
```
{
    "think": ...,
    "answer": ...
}
```
"""

prompt_user_start = """Question: {question}"""

def extract_json(response):
    response = response.replace("```json","").replace("```","")
    response = response.replace("```\n","").replace("\n```","")
    response_json = json.loads(response)
    return response_json

def crop_and_dump(image_path, bbox, output_folder=image_output_dir):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # if any([item >=1000 for item in bbox]):
    #     return None

    try:
        image = Image.open(image_path)
    except Exception as e:
        print(f"cannot open {image_path}: {e}")
        return None
    width, height = image.size

    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

    cropped_image = image.crop((x1, y1, x2, y2))
    # cropped_image = image.crop((x1 * width /1000.0, y1 * height /1000.0, x2 * width /1000.0, y2 * height /1000.0))

    timestamp = int(time.time() * 1000)
    file_extension = os.path.splitext(image_path)[1]
    output_filename = f"{timestamp}{file_extension}"
    output_path = os.path.join(output_folder, output_filename)
    print(image_path)
    print(output_path)

    try:
        cropped_image.save(output_path)
        return output_path
    except Exception as e:
        print(f"fail: {e}")
        return None

class Model_Role:
    def __init__(self,model_name):
        self.model = model_name
    def generate(self,messages):
        if self.model == 'qwen-max-latest':
            return self.generate_llm(messages)
        elif 'vl' in self.model:
            return self.generate_vlm(messages)
    def generate_llm(self,messages):
        dashscope.api_key=model_scope_key
        time = 5
        while True:
            if time == 0:
                return None
            time -= 1
            try:
                response = dashscope.Generation.call(
                    model=self.model,
                    messages=messages,
                    result_format='message',
                )
                if response.status_code == HTTPStatus.OK:
                    return response['output']['choices'][0]['message']['content']
                else:
                    raise Exception(f"{response}")
            except Exception as e:
                print(e)
        
    def generate_vlm(self,messages):
        dashscope.api_key=model_scope_key
        # headers = {"X-DashScope-DataInspection": "disable"}
        times = 5
        while True:
            if times == 0:
                return None
            times -= 1
            try:
                response = dashscope.MultiModalConversation.call(model=self.model,
                                                                messages=messages,
                                                                headers=headers)
                if response.status_code == HTTPStatus.OK:
                    return response.output.choices[0].message.content[0]['text']
                else:
                    raise Exception(f"{response}")
            except Exception as e:
                print(e)

class MMRAG:
    def __init__(self,
                dataset='search_engine/coupus',
                query_file='rag_dataset.json',
                experiment_type = 'cot',
                workers_num = 1,
                topk=10):
        self.experiment_type = experiment_type
        self.workers_num = workers_num
        self.top_k = topk
        self.dataset = dataset
        self.query_file = query_file
        self.dataset_dir = os.path.join('./', dataset)
        self.img_dir = os.path.join(self.dataset_dir, "img")
        self.results_dir = os.path.join(self.dataset_dir, "results")
        os.makedirs(self.results_dir, exist_ok=True)

        # retrieval only
        if experiment_type == 'cot':
            self.eval_func = self.cot_collect
            self.output_file_name = f'cot_crop.jsonl'

        
        self.output_file_path = os.path.join(self.results_dir, self.output_file_name.replace("/","-"))

        self.llm = Model_Role(model_name='qwen-max-latest')
        self.vlm = Model_Role(model_name='qwen-vl-max-latest')
        self.vlm_grounding = Model_Role(model_name='qwen2.5-vl-72b-instruct')

    
    def cot_collect(self,sample):
        query = sample['query']
        reference_images = [f'{raw_image_dir}/'+sample['meta_info']['file_name'].replace('.pdf',f"_{i}.jpg") for i in sample['meta_info']['reference_page']]
        reference_answer = sample['reference_answer']
        all_images=[]
        history=[{
            "query": query
        }]
        messages = []
        messages.append({
            "role": "system",
            "content": [
                {"text": prompt_inst}
            ]
        })
        messages.append({
            "role": "user",
            "content": [
                {"text": prompt_user_start.replace('{question}', query)}
            ]
        })
        try_times = 10
        grounding = False
        while True:
            try_times -=1
            if try_times < 0:
                return None
            while True:
                try:
                    if grounding:
                        response = self.vlm_grounding.generate(messages)
                    else:
                        response = self.vlm.generate(messages)
                    if response is None:
                        return None
                    print(response)
                    response_json = extract_json(response)
                    break
                except Exception as e:
                    time.sleep(1)
                    continue
            # print(response_json)
            if 'think' in response_json and 'answer' in response_json:
                history.append(response_json)
                answer = response_json['answer']
                break
            elif 'think' in response_json and 'search' in response_json:
                search_query = response_json['search']
                queries = [search_query]
                search_result = requests.get(search_engine_url, params={"queries": queries})
                search_result_json = search_result.json()
                image_path_list = [item['image_file'] for item in search_result_json[0]]
                def select_element(A, B, C):
                    # 检查A中元素是否在B中且不在C中的第一个元素
                    for a in A:
                        if a in B and a not in C:
                            return a
                    # 如果没有找到这样的元素，则检查A中第一个不在C中的元素
                    for a in A:
                        if a not in C:
                            return a
                    # 如果所有元素都在C中，则返回None
                    return None
                
                image_input = select_element(image_path_list,reference_images,all_images)
                if image_input is None:
                    return None
                image_path_list = [image_input]

                # assistant
                history.append(response_json)
                messages.append({
                    "role": "assistant",
                    "content": [
                        {"text": response}
                    ]
                })
                # user
                images_content = [{'image': image_path} for image_path in image_path_list[:1]]
                images_content +=[{'text': "You should call crop tool to crop this image. The selected area must be complete and can be larger than the area that needs attention."}]
                messages.append({
                    "role": "user",
                    "content": images_content
                })
                history.append(images_content)
                all_images += image_path_list
                grounding = True
            elif 'think' in response_json and 'bbox' in response_json:
                bbox = response_json['bbox']
                if len(all_images) == 0:
                    continue
                croped_image_path = crop_and_dump(all_images[-1], bbox)
                if croped_image_path is None:
                    continue
                # assistant
                messages.append({
                    "role": "assistant",
                    "content": [
                        {"text": response}
                    ]
                })
                # user
                images_content = [{'image': croped_image_path}]
                messages.append({
                    "role": "user",
                    "content": images_content
                })

                history.append(response_json)
                history.append(images_content)
                grounding = False
        sample['history'] = history
        # sample['eval_result'] = self.evaluator.evaluate(query, sample['reference_answer'], str(answer))
        sample['response'] = answer
        sample['recall_results'] = dict(
            source_nodes=[NodeWithScore(node=ImageNode(image_path=image,metadata=dict(file_name=image)), score=None).to_dict() for image in all_images],
            response=None,
            metadata=None)
        return sample


    def eval_dataset(self):
        eval_func = self.eval_func
        
        rag_dataset_path = os.path.join(self.dataset_dir,self.query_file)
        with open(rag_dataset_path, "r") as f:
            data = json.load(f)
        data = data['examples']
        
        if os.path.exists(self.output_file_path):
            results = []
            with open(self.output_file_path, "r") as f:
                for line in f:
                    results.append(json.loads(line.strip()))
            uid_already = [item['uid'] for item in results]
            data = [item for item in data if item['uid'] not in uid_already]
            
        if self.workers_num == 1:
            for item in tqdm(data):
                result = eval_func(item)
                if result is None:
                    continue
                with open(self.output_file_path, "a") as f:
                    json.dump(result, f,ensure_ascii=False)
                    f.write("\n")
        else:
            with ThreadPoolExecutor(max_workers=self.workers_num) as executor:
                futures = [executor.submit(eval_func, item) for item in data]
                results = []
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                    result = future.result()
                    results.append(result)
                    if len(results) >= 3:
                        with open(self.output_file_path, "a") as f:
                            for res in results:
                                if res is None:
                                    continue
                                f.write(json.dumps(res,ensure_ascii=False) + "\n")
                        results = []
                if results:
                    with open(self.output_file_path, "a") as f:
                        for res in results:
                            if res is None:
                                continue
                            f.write(json.dumps(res,ensure_ascii=False) + "\n")

        return self.output_file_path
        
def arg_parse():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='example', help="The name of dataset")
    parser.add_argument("--query_file", type=str, default='rag_dataset.json', help="The name of anno_file")
    parser.add_argument("--experiment_type", type=str, default='cot', help="The type of experiment")
    parser.add_argument("--workers_num", type=int, default=10, help="The number of workers")
    parser.add_argument("--topk", type=int, default=10, help="The number of topk")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = arg_parse()
    mmrag = MMRAG(
        dataset=args.dataset,
        query_file=args.query_file,
        experiment_type=args.experiment_type,
        workers_num=args.workers_num,
        topk=args.topk
    )
    mmrag.eval_dataset()