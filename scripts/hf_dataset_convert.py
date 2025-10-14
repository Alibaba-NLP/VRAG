import json
from datasets import Dataset
import os
import datasets
import argparse
from tqdm import tqdm


USER_PROMPT = '''Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and user will return the searched results. Every time you retrieve an image, you have the option to crop it to obtain a clearer view, the format for coordinates is <bbox>[x1, y1, x2, y2]</bbox>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}'''



# all_examples = [example for example in all_examples if example['query'] not in sft_questions]
def convert_dataset(USER_PROMPT,file_list,file_source_list,output_name):
    all_examples = []
    for file_name, source_type in zip(file_list, file_source_list):
        with open(file_name, "r") as f:
            file_data = json.load(f)
            data_list = file_data["examples"]
            for item in data_list:
                item['source'] = source_type
            all_examples.extend(data_list)

    for example in all_examples:
        if example['source'] == 'vidoseek':
            example['reason_type'] = example['meta_info']['query_type']
            example['content_type'] = example['meta_info']['source_type']
        elif example['source'] == 'slidevqa_test':
            query_type = example['meta_info']['query_type']
            if 'Multi-Hop' in query_type:
                example['reason_type'] = 'MultiHop'
            elif 'Single-Hop' in query_type:
                example['reason_type'] = 'SingleHop'
            if 'Non-Span' in query_type:
                example['content_type'] = 'NonSpan'
            elif 'Single-Span' in query_type:
                example['content_type'] = 'SingleSpan'
            elif 'Multi-Span' in query_type:
                example['content_type'] = 'MultiSpan'
        elif example['source'] == 'mmlongdoc':
            example['content_type'] = '####'.join(example['meta_info']['source_type'])
            example['reason_type'] = example['meta_info']['doc_type']
        else:
            example['content_type'] = 'Nan'
            example['reason_type'] = 'Nan'

    dataset = Dataset.from_dict({
        "id": [str(example["uid"]) for example in all_examples],
        "problem": [example["query"] for example in all_examples],
        "prompt": [USER_PROMPT.replace('{question}',example["query"]) for example in all_examples],
        "answer": [example["reference_answer"] for example in all_examples],
        "file_name": [example["meta_info"]["file_name"] for example in all_examples],
        "reference_page": [example["meta_info"]["reference_page"] for example in all_examples],
        "data_source_type": [example["source"] for example in all_examples],
        "query_content_type": [example["content_type"] for example in all_examples],
        "query_reason_type": [example["reason_type"] for example in all_examples]
    })

    def make_map_fn_test(split):
        def process_fn(example, idx):
            prompt = example.pop('prompt')
            answer = example.pop('answer')
            problem = example.pop('problem')
            data_source = example.pop('data_source_type')
            reference_page = example.pop('reference_page')
            file_name = example.pop('file_name')
            # images = example.pop('images')
            query_content_type = example.pop('query_content_type')
            query_reason_type = example.pop('query_reason_type')

            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": prompt,
                }],
                # "images": images,
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'answer': answer,
                    "question": problem,
                    "content_type": query_content_type,
                    "reason_type": query_reason_type,
                    "file_name": file_name,
                    "reference_page": reference_page
                }
            }
            return data
        return process_fn

    test_dataset = dataset.map(function=make_map_fn_test('test'), with_indices=True, num_proc=8)

    test_dataset.to_parquet(f'./data/{output_name}.parquet')


if __name__ == '__main__':
    convert_dataset(
        USER_PROMPT,
        ['./data/SlideVQA_test.json', './data/MMLongDoc.json','./data/SlideBench.json'],
        ['slidevqa_test', 'mmlongdoc', 'vidoseek'],
        'overall_test'
    )
    convert_dataset(
        USER_PROMPT,
        ['./data/SlideVQA_train.json'],
        ['slidevqa_train'],
        'overall_train'
    )