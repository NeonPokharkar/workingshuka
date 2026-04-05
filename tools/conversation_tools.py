from langchain.tools import ToolException, tool
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


import static.constants as constants
import tools.rag_tools as rag_tools

vector_db = Chroma(
    persist_directory=constants.CONVERSATION_DIRECTORY, 
    embedding_function=rag_tools.embeddings,
    collection_name=constants.CONVERSATION_COLLECTION_NAME
)

def save_conversation(question: str, answer: str, context: str, dateandtime: str) -> str:
    """Saves the current conversation to the RAG database."""
    doc = Document(
        page_content=f"User said: {question}\nYou answered: {answer}\nContext was: {context}\n at {dateandtime}",
        metadata={"context": context, "datetime": dateandtime}
    )
    vector_db.add_documents([doc])
    return "Conversation saved successfully."