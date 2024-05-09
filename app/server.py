from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from rag_ollama_multi_query import chain as rag_ollama_multi_query_chain
from rag_multi_index_router import chain as rag_multi_index_router_chain

app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


# Edit this to add the chain you want to add
add_routes(app, rag_ollama_multi_query_chain, path="/rag-ollama-multi-query")
add_routes(app, rag_multi_index_router_chain, path="/rag-multi-index-router")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
