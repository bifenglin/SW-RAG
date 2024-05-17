from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
import sys
sys.path.append("/Users/wangyantong/SW-RAG/packages/rag-chroma-private")
from rag_chroma_private import chain as rag_chroma_private_chain
import sys
sys.path.append("/Users/wangyantong/SW-RAG/packages/rag-ollama-multi-query")
from rag_ollama_multi_query import chain as rag_ollama_multi_query_chain

app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


# Edit this to add the chain you want to add
add_routes(app, rag_chroma_private_chain, path="/rag-chroma-private")
add_routes(app, rag_ollama_multi_query_chain, path="/rag-ollama-multi-query")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
