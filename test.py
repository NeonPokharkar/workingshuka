import faster_whisper

import functions.rag_functions as rag_functions

print("Downloading datasets...")

from TTS.api import TTS

# rag_functions.download_dataset(page_content_function=lambda entry: entry['text'], dataset_path="HuggingFaceTB/cosmopedia", dataset_split = "train", dataset_name="auto_math_text", called_as="Cosmopedia Auto Math Text", purpose_description="A high-quality educational subset of math-related articles from the Cosmopedia dataset, ideal for training a RAG system focused on mathematical content.", subject_expertise="Mathematics")

# rag_functions.download_dataset(page_content_function=lambda entry: f"Question: {entry['question']} \nAnswer: {entry['response_j']}", dataset_path="lvwerra/stack-exchange-paired", dataset_split="test", called_as="Stack Exchange Paired", purpose_description="A dataset of paired questions and answers from Stack Exchange, useful for training a RAG system to understand and retrieve information based on user queries.", subject_expertise="General Answering and Explanation Understanding")

# rag_functions.download_dataset(page_content_function=lambda entry: entry["support"], dataset_path="allenai/sciq", dataset_split="train", called_as="SCiQ", purpose_description="A dataset of multiple-choice science questions and answers from SCiQ, useful for training a RAG system to understand and retrieve information based on user queries.", subject_expertise="Science Question Answering")

# rag_functions.download_dataset(page_content_function=lambda entry: entry['contents'], dataset_path="MedRAG/textbooks", dataset_split="train", called_as="MedRAG Textbooks", purpose_description="A dataset of medical textbook content from MedRAG, useful for training a RAG system to understand and retrieve information based on medical queries.", subject_expertise="Medical Information Retrieval")

# rag_functions.download_dataset(page_content_function=lambda entry: f"On {entry["year"]}/{entry["month"]}/{entry["day"]}, this happened, {entry["event_description"]}", dataset_name='en', dataset_path="santhosh/day_in_history", dataset_split="train", called_as="Day in History", purpose_description="A dataset of historical events organized by date from the 'Day in History' dataset, useful for training a RAG system to understand and retrieve information based on historical queries.", subject_expertise="History Information Retrieval")

# rag_functions.download_dataset(page_content_function=lambda entry: f"Question: {entry['question']}\nAnswer: {entry['answer']}", dataset_path="mandarjoshi/trivia_qa", dataset_name="rc", dataset_split="train", called_as="TriviaQA", purpose_description="A dataset of trivia questions and answers from TriviaQA, useful for training a RAG system to understand and retrieve information based on general knowledge queries.", subject_expertise="General Knowledge")      

import torch

print("Downloading model...")

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

# kokoro = Kokoro("kokoro-v0_19.onnx", "voices.json")

print("Done downloading model")