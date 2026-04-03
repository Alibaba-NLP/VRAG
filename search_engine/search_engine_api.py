from fastapi import FastAPI, HTTPException, Request
from typing import List
import uvicorn
from search_engine import SearchEngine

app = FastAPI()

model_path = "Qwen3-VL-Embedding-2B"
corpus_path = ["search_engine/corpus/image_index"]

engine = SearchEngine(model_path)
engine.load_multi_index_corpus_together(corpus_path)

@app.post("/search")
async def search(request: Request):
    try:
        body = await request.json()
        queries = body.get("queries", [])
        top_k = body.get("top_k", 3)
        vrag_ret = body.get("vrag_ret", False)

        search_results = engine.search(queries, top_k)

        if vrag_ret:
            results_batch = []
            for result in search_results:
                data_list = result.get("data", []) if isinstance(result, dict) else []
                query_images = []
                for idx, item in enumerate(data_list):
                    if isinstance(item, dict) and item.get("type") == "image":
                        query_images.append(
                            {
                                "idx": idx,
                                "image_file": item.get("file_path"),
                            }
                        )
                results_batch.append(query_images)
            return results_batch

        return {"results": search_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)

"""
Search Engine API - 多模态检索服务

启动服务:
    python search_engine_api.py

测试示例 (curl):
curl -X POST http://localhost:8001/search \
    -H "Content-Type: application/json" \
    -d '{"queries": ["查询1", "查询2"], "top_k": 3}'

只有在 vrag_ret 为 True 时，返回结果才包含查询图片的文件路径。
curl -X POST http://localhost:8001/search \
  -H "Content-Type: application/json" \
  -d '{
    "queries": ["查询1", "查询2"],
    "top_k": 3,
    "vrag_ret": true
  }'
"""
