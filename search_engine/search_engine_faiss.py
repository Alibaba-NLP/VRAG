import faiss
from tqdm import tqdm
import numpy as np
import time
import os
import json
import torch


class SearchEngine:
    def __init__(self, embed_model_name='GVE'): # Alibaba-NLP/GVE-3B Alibaba-NLP/GVE-7B

        self.embed_model_name = embed_model_name

        self.embed_model_name_file = embed_model_name.replace('/', '_').replace('-','_')

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if 'GVE' in self.embed_model_name:
            from search_engine.models.GVE import AutoModelForSentenceEmbeddingTriplet
            from search_engine.models.GVE import VLProcessor
            if '3B' in self.embed_model_name:
                model_name_or_path = "Alibaba-NLP/GVE-3B"
                self.dimension = 2048
            elif '7B' in self.embed_model_name:
                model_name_or_path = "Alibaba-NLP/GVE-7B"
                self.dimension = 3584

            self.model = AutoModelForSentenceEmbeddingTriplet(model_name_or_path)
            self.processor = VLProcessor(model_name_or_path)
            
        elif 'bge-m3' in self.embed_model_name:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.embed_model_name)
            self.dimension = 1024

    def load_index(self, input_dir):
        # meta data
        with open(os.path.join(input_dir, f"{self.embed_model_name_file}_meta_data.json"), "r") as f:
            self.meta_data = json.load(f)
        assert self.meta_data["model_name"] == self.embed_model_name
        print(f"loading model {self.embed_model_name}; loading index from {input_dir}; loading index length {self.meta_data['num_vectors']}...")
        
        self.index = faiss.read_index(os.path.join(input_dir, f"{self.embed_model_name_file}_faiss_index.bin"))
        with open(os.path.join(input_dir, f"{self.embed_model_name_file}_file_data_list.jsonl"), "r") as f:
            self.file_data_list = [json.loads(line) for line in f]
        assert self.meta_data["num_vectors"] == self.index.ntotal \
            and len(self.file_data_list) == self.index.ntotal, \
            f"meta_data_num_vectors: {self.meta_data['num_vectors']} and index_ntotal: {self.index.ntotal} mismatch \n or file_data_list_length: {len(self.file_data_list)} and index_ntotal: {self.index.ntotal} mismatch"

    def load_multi_index_corpus_together(self, input_dir_list):
        for input_dir in input_dir_list:
            if not hasattr(self, "meta_data"):
                with open(os.path.join(input_dir, f"{self.embed_model_name_file}_meta_data.json"), "r") as f:
                    self.meta_data = json.load(f)
                    num_vectors = self.meta_data['num_vectors']
                    print(f"Loading index from {input_dir}: {num_vectors}")

                assert self.meta_data["model_name"] == self.embed_model_name
                self.index = faiss.read_index(os.path.join(input_dir, f"{self.embed_model_name_file}_faiss_index.bin"))

                with open(os.path.join(input_dir, f"{self.embed_model_name_file}_file_data_list.jsonl"), "r") as f:
                    self.file_data_list = [json.loads(line) for line in f]
                

            else:
                with open(os.path.join(input_dir, f"{self.embed_model_name_file}_meta_data.json"), "r") as f:
                    num_vectors = json.load(f)['num_vectors']
                    self.meta_data['num_vectors'] += num_vectors
                    print(f"Loading index from {input_dir}: {num_vectors}")
                assert self.meta_data["model_name"] == self.embed_model_name
                self.index.merge_from(faiss.read_index(os.path.join(input_dir, f"{self.embed_model_name_file}_faiss_index.bin")))
                with open(os.path.join(input_dir, f"{self.embed_model_name_file}_file_data_list.jsonl"), "r") as f:
                    self.file_data_list.extend([json.loads(line) for line in f])

            
            assert len(self.file_data_list) == self.index.ntotal, f"{len(self.file_data_list)} != {self.index.ntotal}"
        
        print(f"load index done: {self.embed_model_name}, {self.index.ntotal} index files loaded")

    def build_index(self, input_dir, index_output_path, corpus_output_path, bs=2, save_interval=512):
        os.makedirs(index_output_path, exist_ok=True)
        os.makedirs(corpus_output_path, exist_ok=True)

        # Step 1: 加载现有索引和元数据（如果存在）
        index_path = os.path.join(index_output_path, f"{self.embed_model_name_file}_faiss_index.bin")
        meta_data_path = os.path.join(index_output_path, f"{self.embed_model_name_file}_meta_data.json")
        file_data_list_path = os.path.join(corpus_output_path, f"{self.embed_model_name_file}_file_data_list.jsonl")
        
        if os.path.exists(index_path) and os.path.exists(meta_data_path) and os.path.exists(file_data_list_path):
            self.load_index(index_output_path)
            print(f"成功加载已有索引，当前索引包含 {len(self.file_data_list)} 个文件！")
        else:
            print("未检测到已有索引，将创建新索引...")
            self.index = faiss.IndexFlatIP(self.dimension)
            self.file_data_list = []

        # Step 2: 准备输入文件列表
        new_file_data_list = []

        if os.path.isfile(input_dir) and input_dir.endswith('.jsonl'):
            already_file_uid = {entry['uid'] for entry in self.file_data_list}
            with open(input_dir, "r") as f:
                print("Reading file:", input_dir)
                for line in tqdm(f):
                    line_json = json.loads(line)
                    if line_json['uid'] not in already_file_uid:
                        new_file_data_list.append(line_json)
        else:
            file_list = os.listdir(input_dir)
            file_path = [os.path.join(input_dir, file) for file in file_list]
            already_file_path = {entry['file_path'] for entry in self.file_data_list}
            print('loading files ...')
            for file in tqdm(file_path):
                # 检查文件是否已在 index 中存在（防止重复处理）
                if file in already_file_path:
                    continue
                
                suffix = os.path.splitext(file)[-1]
                if suffix in [".txt", ".md", '.json']:
                    new_file_data_list.append(dict(
                        type='text',
                        file_path=file,
                        content=open(file, 'r', encoding='utf-8').read()
                    ))
                elif suffix in [".mp4"]:
                    new_file_data_list.append(dict(
                        type='video',
                        file_path=file
                    ))
                elif suffix in [".jpg", ".png", ".jpeg"]:
                    new_file_data_list.append(dict(
                        type='image',
                        file_path=file
                    ))

        # 如果没有新文件需要处理，直接退出
        if len(new_file_data_list) == 0:
            print("没有新的文件需要处理，索引已是最新！")
            return
        
        print(f"本次将为 {len(new_file_data_list)} 个新文件生成索引...")

        # Step 3: 生成向量并更新索引
        processed_count = 0  # 计数器，记录已经处理的文件数
        if 'GVE' in self.embed_model_name:
            input_data_list = []

            for file in new_file_data_list:
                file_data = {
                    "type": file['type'],
                    file['type']: file['file_path'] if file['type'] != 'text' else file['content']
                }
                if file['type'] == 'video':
                    file_data['fps'] = 1.0
                    file_data['max_frames'] = 8
                    file_data['min_frames'] = 32
                    file_data['max_pixels'] = 200*28*28
                input_data_list.append([file_data])

            for i in tqdm(range(0, len(input_data_list), bs)):
                try:

                    file_batch = input_data_list[i:i+bs]
                    messages = [[
                        {
                            "role": "user",
                            "content": file_data,
                        }
                    ] for file_data in file_batch]
                    inputs = self.processor(messages)
                    # inputs = inputs.to("cuda")
                    inputs = inputs.to(self.device)
                    outputs = self.model.encode(inputs).detach().cpu().float().numpy()
                    self.index.add(outputs.astype('float32'))


                except Exception as e:
                    print(f"处理文件 {file_batch} 时出错: {e}")                        
                    continue
                
                # 更新当前状态
                processed_count += len(file_batch)
                self.file_data_list.extend(new_file_data_list[i:i+bs])
                
                # 每 1000 个保存一次
                if processed_count % save_interval == 0:
                    print(f"已处理 {processed_count} 个样本，保存当前索引...")
                    self._save_index_and_metadata(self.index, self.file_data_list, index_path, file_data_list_path, meta_data_path)
                    print(f"index size {self.index.ntotal}, file_data_list size {len(self.file_data_list)}")

        elif 'bge-m3' in self.embed_model_name:
            input_data_list = [file['content'] for file in new_file_data_list]
            for i in tqdm(range(0, len(input_data_list), bs)):
                sentences = input_data_list[i:i+bs]
                outputs = self.model.encode(sentences)
                self.index.add(outputs.astype('float32'))
                
                # 更新当前状态
                processed_count += len(sentences)
                self.file_data_list.extend(new_file_data_list[i:i+bs])
                
                # 每 1000 个保存一次
                if processed_count % save_interval == 0:
                    print(f"已处理 {processed_count} 个样本，保存当前索引...")
                    self._save_index_and_metadata(self.index, self.file_data_list, index_path, file_data_list_path, meta_data_path)

        # 最后保存一次
        print("保存最终索引文件...")
        self._save_index_and_metadata(self.index, self.file_data_list, index_path, file_data_list_path, meta_data_path)
        print(f"索引更新完成！当前总文件数：{len(self.file_data_list)}")


    def _save_index_and_metadata(self, index, file_data_list, index_path, file_data_list_path, meta_data_path):
        """
        内部方法，用于保存索引和元数据。
        """
        # 保存索引
        faiss.write_index(index, index_path)
        
        # 保存文件列表
        with open(file_data_list_path, "w") as f:
            for file_data in file_data_list:
                f.write(json.dumps(file_data) + "\n")
        
        # 保存元数据
        metadata = {
            "vector_dimension": index.d,
            "num_vectors": len(file_data_list),
            "model_name": self.embed_model_name,
            "date_created": time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        }
        with open(meta_data_path, "w") as f:
            json.dump(metadata, f, indent=4)




    def search(self, queries, top_k=3, search_corpus=None):
        if isinstance(queries, str):
            queries = [queries]
        
        if search_corpus is not None:
            assert hasattr(self, 'index_dict'), 'index_dict not found'

            if search_corpus not in self.index_dict:
                return []
            
            _index = self.index_dict[search_corpus]['index']
            _file_data_list = self.index_dict[search_corpus]['file_data_list']

        else:
            _index = self.index
            _file_data_list = self.file_data_list

        top_k = min(top_k, len(_file_data_list))

        if 'GVE' in self.embed_model_name:
            messages = [[
                {
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": query
                    }]
                }
            ] for query in queries]
            inputs = self.processor(messages)
            # inputs = inputs.to("cuda")
            inputs = inputs.to(self.device)
            query_vectors = self.model.encode(inputs).detach().cpu().float().numpy()
            similarities, indices = _index.search(query_vectors.astype('float32'), top_k)
            search_results = [dict(
               score =  score,
               indice = indice,
               data = [_file_data_list[index] for index in indice]
            ) for score, indice in zip(similarities.tolist(), indices.tolist())]

        elif 'bge-m3' in self.embed_model_name:
            query_vectors = self.model.encode(queries)
            similarities, indices = _index.search(query_vectors.astype('float32'), top_k)
            search_results = [dict(
               score =  score,
               indice = indice,
               data = [_file_data_list[index] for index in indice]
            ) for score, indice in zip(similarities.tolist(), indices.tolist())]
        
        return search_results


if __name__ == '__main__':
    # engine = SearchEngine("GVE-Qwen25-VL-7B")
    engine = SearchEngine("Alibaba-NLP/GVE-7B")

    bs_video = 1
    bs_image = 16
    bs_text = 32

    engine.build_index('./data/corpus/LVBench/videos_chunk60s','./data/corpus/LVBench/videos_chunk60s_index', './data/corpus/LVBench/videos_chunk60s_index', bs=bs_video)
