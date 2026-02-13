from fastapi import FastAPI, HTTPException, Request
from typing import List
import uvicorn
from search_engine.search_engine_faiss import SearchEngine

app = FastAPI()

engine = SearchEngine("GVE-Qwen25-VL-7B")
engine.load_multi_index_corpus_together([
        # './data/corpus/squad/corpus_train_index',
        # './data/corpus/WebQA/images_train_index',
        # './data/corpus/SlideVQA/images_train_index',
        # './data/corpus/howto100m/videos_chunk60s_index',
    ])


@app.post("/search")
async def search(request: Request):
    try:
        body = await request.json() 
        queries = body.get("queries", [])
        top_k = body.get("top_k", 3)
        search_corpus = body.get("search_corpus", None)
        search_results = engine.search(queries,top_k,search_corpus)
        return {"results": search_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

