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

import re
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

def calculate_anls(gold_labels, prediction, threshold=0.7):
    import numpy as np
    from Levenshtein import distance as levenshtein_distance
    max_scores = []
    for gold_label in gold_labels:
        ld = levenshtein_distance(prediction, gold_label)
        max_len = max(len(prediction), len(gold_label))
        if max_len == 0:
            nld = 0.0
        else:
            nld = ld / max_len
        if nld < threshold:
            score = 1 - nld
        else:
            score = 0.0
        max_scores.append(score)
    return max(max_scores)
def compute_score(predict_str: str, ground_truth: str, extra_info) -> float:
    predict_str = remove_text_between_tags(predict_str)
    format_reward_value=compute_format_reward_only(predict_str)

    if format_reward_value==1.0:
        answer = get_answer_from_predict_str(predict_str)
        if answer is None:
            return 0.0
        anls_score = calculate_anls([ground_truth], answer, 0.5)
        return anls_score
    else:
        return 0.0

def remove_text_between_tags(text):
    # 使用正则表达式匹配<|im_start|>和<|im_end|>之间的内容以及这两个标签
    pattern = r'<\|im_start\|>user.*?<\|im_end\|>'
    # 替换匹配到的内容为空字符串
    result = re.sub(pattern, '', text)
    return result
def compute_format_reward_only(predict_str: str, ground_truth: str, extra_info) -> float:
    predict_str = remove_text_between_tags(predict_str)
    answer_pattern = re.compile(r'<answer>.*</answer>', re.DOTALL)
    search_pattern = re.compile(r'<search>.*</search>', re.DOTALL)
    answer_match = re.search(answer_pattern, predict_str)
    search_match = re.search(search_pattern, predict_str)
    return 1.0 if answer_match and search_match else 0.0

