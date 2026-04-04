from langchain_core.tools import create_retriever_tool
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from static import constants

embeddings = OllamaEmbeddings(model=constants.EMBEDDING_MODEL, num_gpu=1)

vectorstores = []

tools = []

def add_vector_store_tool_from_db(rag_path:str, collection_name:str, tool_name:str, tool_description:str) -> list:
    """Adds the vector store retrieval tool to the existing list of tools."""
    vector_db = Chroma(
        persist_directory=rag_path, 
        embedding_function=embeddings,
        collection_name=collection_name
    )

    vectorstores.append(vector_db)

    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    new_tool = create_retriever_tool(
        retriever,
        tool_name,
        tool_description
    )

    tools.append(new_tool)

def add_vector_store_tool_from_documents(document_directory: str, tool_name: str, tool_description: str) -> list:
    loader = DirectoryLoader(
        document_directory,
        show_progress=True
    )

    raw_documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(raw_documents)

    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=f"/{constants.VECTOR_DIRECTORY}/{tool_name}/",
        collection_name=f"{tool_name}_collection"
    )

    vectorstores.append(vector_db)

    retriever = vector_db.as_retriever(search_kwargs={"k": 3})


    new_tool = create_retriever_tool(
        retriever,
        tool_name,
        tool_description
    )

    tools.append(new_tool)
    
add_vector_store_tool_from_db(constants.CONVERSATION_DIRECTORY, constants.CONVERSATION_COLLECTION_NAME, "conversation_retriever", "Retrieves relevant past conversations based on the user's query.")
add_vector_store_tool_from_db(constants.VECTOR_DIRECTORY, constants.VECTOR_COLLECTION_NAME, "general_knowledge_retriever", "Retrieves relevant general knowledge information based on the user's query.")

add_vector_store_tool_from_documents(constants.INSTRUCTION_RAG_PATH, "instruction_retriever", "Retrieves relevant instruction sets based on the user's query.")
add_vector_store_tool_from_documents(constants.RAG_PATH, "personal_data_retriever", "Retrieves relevant personal data based on the user's query.")

print(f"Tools added: {[tool.name for tool in tools]}")