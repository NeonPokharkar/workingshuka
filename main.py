from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from time import time

from pydantic import BaseModel

class Chat(BaseModel):
    query: str

app = FastAPI()

Settings.llm = Ollama(model="qwen3:30b", request_timeout=1000, streaming=True)
Settings.embed_model = OllamaEmbedding(model_name="qwen3-embedding:8b", request_timeout=1000, streaming=True)

documents = SimpleDirectoryReader("./data", filename_as_id=True).load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(streaming=True)

def chunked_response(query):
    yield " "

    response = query_engine.query(query)
    for chunk in response.response_gen:
        if chunk:
            yield chunk

# starttime = time()

# response = query_engine.query("Who are you? What is your purpose?")
# print(response)

# endtime = time()

# print(f"Time taken: {endtime - starttime} seconds")

@app.get("/query")
async def query(chat: str):
    return StreamingResponse(chunked_response(chat), media_type="text/plain")