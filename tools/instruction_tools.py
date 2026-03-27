from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from langchain.tools import ToolException, tool

import static.constants as constants

Settings.llm = Ollama(model=constants.BASE_MODEL, request_timeout=100, thinking='medium')
Settings.embed_model = OllamaEmbedding(model_name=constants.EMBEDDING_MODEL, request_timeout=1000)

documents = SimpleDirectoryReader(constants.INSTRUCTION_RAG_PATH, filename_as_id=True).load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

@tool
def find_relevant_instruction(query: str) -> str:
    """Finds relevant instruction set from the RAG index based on the user's requirement or task need."""
    try:
        response = query_engine.query(query)
        if response:
            return str(response)
        else:
         raise ToolException("No relevant instruction found for the given query.")
    except Exception as e:
        return f"Error retrieving instruction: {str(e)}"

tools_list = [find_relevant_instruction]