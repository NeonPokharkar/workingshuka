from fastapi import FastAPI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

app = FastAPI()

Settings.llm = Ollama(model="qwen3:30b", request_timeout=60)
Settings.embed_model = OllamaEmbedding(model_name="qwen3-embedding:8b")

documents = SimpleDirectoryReader("./data", filename_as_id=True).load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

response = query_engine.query("What is the name of the agent?")
print(response)

@app.get("/query")
async def query(q: str):
    response = query_engine.query(q)
    return {"response": str(response)}