# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
import json
import requests
import math
import numpy as np
import os
import re
from concurrent.futures import ThreadPoolExecutor


DEFAULT_SYSTEM_TEMPLATE = """You are an expert evaluation system for a question answering chatbot.

You are given the following information:
- the query
- a generated answer
- a reference answer

Your task is to evaluate the correctness of the generated answer.

## Query
{query}

## Reference Answer
{reference_answer}

## Generated Answer
{generated_answer}

Your response should be formatted as following:
<judge>True or False</judge>

If the generated answer is correct, please set "judge" to True. Otherwise, please set "judge" to False.

Please note that the generated answer may contain additional information beyond the reference answer.
"""

def parse_judge_response(response):
    pattern = r'<judge>(.*?)</judge>'
    match = re.search(pattern, response, re.DOTALL)
    if match:
        try:
            judge_str = match.group(0)
            if 'true' in judge_str or 'True' in judge_str:
                return True
            elif 'false' in judge_str or 'False' in judge_str:
                return False
            else:
                return None
        except Exception as e:
            return None
    else:
        return None

    
def dcg(relevance_scores):
    """
    计算折扣累积增益（DCG）
    :param relevance_scores: 一个列表，表示每个文档的相关性分数
    :return: DCG 值
    """
    dcg_value = 0.0
    for i, relevance in enumerate(relevance_scores, start=1):
        dcg_value += (2 ** relevance - 1) / np.log2(i + 1)
    return dcg_value

def ndcg(sorted_docs, golden_answer_list):
    """
    计算归一化折扣累积增益（NDCG）
    :param sorted_docs: 一个列表，表示已经排好序的文档
    :param golden_answer_list: 一个列表，表示所有相关文档（golden answers）
    :return: NDCG 值
    """
    # 将文档映射为相关性分数（在 golden_answer_list 中的文档为 1，否则为 0）
    relevance_scores = [1 if doc in golden_answer_list else 0 for doc in sorted_docs]
    
    # 计算 DCG
    dcg_value = dcg(relevance_scores)
    
    # 计算 IDCG（理想情况下的 DCG，所有相关文档都排在前面）
    ideal_relevance_scores = [1] * len(golden_answer_list) + [0] * (len(sorted_docs) - len(golden_answer_list))
    idcg_value = dcg(ideal_relevance_scores)
    
    # 防止分母为零
    if idcg_value == 0:
        return 0.0
    
    # 计算 NDCG
    ndcg_value = dcg_value / idcg_value
    return ndcg_value

def get_answer_from_predict_str(text):
    end_tag = '</answer>'
    start_tag = '<answer>'
    
    end_pos = text.rfind(end_tag)
    if end_pos == -1:
        return None  # 如果没有找到</answer>，返回None
    
    start_pos = text.rfind(start_tag, 0, end_pos)
    if start_pos == -1:
        return None  # 如果没有找到<answer>，返回None
    
    start_pos += len(start_tag)  # 跳过<answer>标签
    return text[start_pos:end_pos]


class RMManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, eval_mode=False, rm_workers_num=10, rm_url="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions", rm_key="EMPTY", rm_model_name="qwen-max-latest") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.eval_mode = eval_mode
        self.rm_workers_num = rm_workers_num
        self.rm_url = rm_url
        self.rm_key = rm_key
        self.rm_model_name = rm_model_name
    
    def rm_score(self,data_eval_item):
        pay_load = {
            "model": self.rm_model_name,
            "messages": [
                {
                    "role": "user", 
                    "content": DEFAULT_SYSTEM_TEMPLATE \ 
                        .replace("{query}", data_eval_item["query"]) \
                        .replace("{reference_answer}", data_eval_item["reference_answer"]) \ 
                        .replace("{generated_answer}", data_eval_item["generated_answer"])
                }
            ]
        }
        max_retry = 20
        while True:
            if max_retry <= 0:
                return 0.0
            try:
                response = requests.post(
                    self.rm_url,
                    headers={
                        "Content-Type": "application/json",
                        "X-Auth-Key": self.rm_key,
                    },
                    json=pay_load
                )
                response.raise_for_status()
                result = response.json()
                judge_str = parse_judge_response(result['output']['content'])
                if judge_str is not None:
                    if judge_str:
                        return 1.0
                    else:
                        return 0.0
            except Exception as e:
                continue

    def verify(self, data):
        scores = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            response_str = self.tokenizer.decode(valid_response_ids)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']

            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            scores.append(score)
        data.batch['acc'] = torch.tensor(scores, dtype=torch.float32, device=prompt_ids.device)
        return scores

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        data_eval = []
        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            extra_info = data_item.non_tensor_batch.get('extra_info', None)
            generated_answer = get_answer_from_predict_str(self.tokenizer.decode(valid_response_ids))
            if generated_answer is None:
                generated_answer = 'Please Judge False'
            data_eval.append(dict(
                query = extra_info['question'],
                generated_answer = generated_answer,
                reference_answer = data_item.non_tensor_batch['reward_model']['ground_truth']
            ))

        data_to_be_eval = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            response_str = self.tokenizer.decode(valid_response_ids)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']

            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            
            if score >0.0:
                data_to_be_eval.append(data_eval[i])
        
        if len(data_to_be_eval) > 0:
            
            if self.rm_workers_num > 1:
                with ThreadPoolExecutor(max_workers=self.rm_workers_num) as pool:
                    eval_results = list(pool.map(self.rm_score, data_to_be_eval))
            else:
                eval_results = [self.rm_score(data_to_be_eval[i]) for i in range(len(data_to_be_eval))]

            # request_data_to_be_eval = dict(
            #     bs=300,
            #     prompts=data_to_be_eval
            # )
            # prompts_json = json.dumps(request_data_to_be_eval)
            # print("=====================eval model start=====================")
            # response = requests.post(self.rm_url, json=prompts_json)
            # eval_results = response.json()
            # print("=====================eval model end=====================")
            
        
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            response_str = self.tokenizer.decode(valid_response_ids)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']

            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            
            if score >0.0:
                if self.eval_mode:
                    score = eval_results.pop(0)
                else:
                    retrievaled_images_basename_list = [os.path.basename(item.rstrip('/')).split(".jpg")[0] for item in data_item.non_tensor_batch['retrievaled_images']]
                    reference_images_basename_list = [f'{extra_info["file_name"].split(".pdf")[0]}_{page}' for page in extra_info["reference_page"].tolist()]
                    ndcg_value = ndcg(retrievaled_images_basename_list, reference_images_basename_list)

                    model_eval_score = eval_results.pop(0)
                    # score = 0.8*model_eval_score + 0.2*ndcg_value
                    score = 0.7*model_eval_score + 0.1*score + 0.2*ndcg_value

            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[score]", score)

        return reward_tensor    