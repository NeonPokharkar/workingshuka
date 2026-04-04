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

llm = OllamaLLM(model=constants.SECONDARY_MODEL, request_timeout=1000, num_gpu=99) 

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are Shuka, an expert assistant with access to a local database of past conversations. You can use this information to provide personalized answers. Always cite your 'Source' from the metadata in the final answer."),
    ("user", "{input}")
])

chain = prompt | llm | StrOutputParser()

def what_is_user_thinking(current_summary: str, last_relevant_conversation: str, what_user_has_asked_now: str) -> str:
    """
    This function takes the current conversation summary, the last relevant conversation retrieved from the RAG index, and the user's current question. It then generates a concise summary of what the user is currently thinking or asking about.
    """
    input_text = f"Current Summary: {current_summary}\nLast Relevant Conversation: {last_relevant_conversation}\nUser's Current Question: {what_user_has_asked_now}\n\nBased on this information, predict/summarize what the user is currently thinking or asking about in one or two sentences."
    return chain.invoke(input=input_text)

def summarize_agent_thoughts(question: str, summary_of_context: str, thought_chain: str) -> str:
    """
    This function takes the user's question, a summary of the relevant context (including past conversations), and the agent's current thought chain. It then generates a concise summary of the agent's current thoughts and reasoning process.
    """
    input_text = f"User's Question: {question}\nSummary of Relevant Context: {summary_of_context}\nAgent's Current Thought Chain: {thought_chain}\n\nBased on this information, summarize the agent's current thoughts and reasoning process in one or two sentences."
    return chain.invoke(input=input_text)

def save_conversation(question: str, answer: str, context: str, dateandtime: str) -> str:
    """Saves the current conversation to the RAG database."""
    doc = Document(
        page_content=f"User said: {question}\nYou answered: {answer}\nContext was: {context}\n at {dateandtime}",
        metadata={"context": context, "datetime": dateandtime}
    )
    vector_db.add_documents([doc])
    return "Conversation saved successfully."