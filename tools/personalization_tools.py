from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from langchain.tools import ToolException, tool
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

import static.constants as constants

## Personal Information Setup

Settings.llm = Ollama(model=constants.BASE_MODEL, request_timeout=1000, streaming=True, thinking='low')
Settings.embed_model = OllamaEmbedding(model_name=constants.EMBEDDING_MODEL, request_timeout=1000, streaming=True)

documents = SimpleDirectoryReader(constants.RAG_PATH, filename_as_id=True).load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(streaming=True)

# Subject Data Expertise Tool

embeddings = HuggingFaceEmbeddings(model_name=constants.VECTOR_EMBEDDING_MODEL)

vector_db = Chroma(
    persist_directory=constants.VECTOR_DIRECTORY, 
    embedding_function=embeddings,
    collection_name=constants.VECTOR_COLLECTION_NAME
)

@tool
def rag_tool(query: str) -> str:
    """Searches your RAG index for relevant user related information."""
    try:
        response = query_engine.query(query)
        if response:
            return str(response)
        raise ToolException("No relevant information found for the given query.")
    except Exception as e:
        raise ToolException(f"Error retrieving information from RAG index: {str(e)}")

@tool
def get_subject_general_data(subject: str) -> str:
    """Returns general information about a subject from the RAG index."""
    try:
        data = vector_db.similarity_search(subject, k=1) # Get the most relevant chunk
        if data:
            return " ".join(" Metadata: " + str(doc.metadata) + " Content: " + doc.page_content for doc in data)
        raise ToolException("No information found for the given subject.")
    except Exception as e:
        raise ToolException(f"Error retrieving information from RAG index: {str(e)}")

@tool
def greetings_tool() -> str:
    """Provides a personalized greeting based on the time of day."""
    try:
        from datetime import datetime
        current_hour = datetime.now().hour
        if 5 <= current_hour < 12:
            return "Good morning! How can I assist you today?"
        elif 12 <= current_hour < 17:
            return "Good afternoon! What would you like to learn about today?"
        elif 17 <= current_hour < 22:
            return "Good evening! How can I help you unwind with some interesting facts?"
        else:
            return "Hello! It's quite late, but I'm here to help if you have any questions or just want to chat."
    except Exception as e:
        raise ToolException(f"Error generating greeting: {str(e)}")

tools_list = [rag_tool, get_subject_general_data, greetings_tool]