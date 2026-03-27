from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

import static.constants as constants

def download_dataset(page_content_function: callable, called_as: str, purpose_description: str, subject_expertise: str,dataset_path: str, dataset_split: str, dataset_name: str = None):
    # 1. DOWNLOAD: Get a high-quality educational subset (e.g., 'auto_math_text')
    # Using streaming=True allows you to process it without downloading 100GB at once.
    print("Downloading dataset...")
    dataset = load_dataset(dataset_path, dataset_name, split=dataset_split, streaming=True)

    # 2. PREPARE: Take the first 500 articles for this example
    docs = []
    for i, entry in enumerate(dataset):
        if i >= 500: break 
        docs.append(Document(page_content=page_content_function(entry), metadata={"source": called_as, "description": purpose_description, "expertise": subject_expertise}))

    # 3. CHUNK: Split long articles into small pieces (500-1000 characters)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    # 4. EMBED: Convert text into math vectors (using a standard open-source model)
    embeddings = HuggingFaceEmbeddings(model_name=constants.VECTOR_EMBEDDING_MODEL)

    # 5. INPUT TO RAG: Save chunks into a local vector database
    print(f"Adding {len(chunks)} chunks to vector store...")
    Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=constants.VECTOR_DIRECTORY,
        collection_name=constants.VECTOR_COLLECTION_NAME
    )

    print("Finished! Your RAG database is now saved in './vector_datasets'.")
