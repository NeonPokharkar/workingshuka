from datetime import datetime

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import setup.agent_setup as agent_setup
import setup.api_setup as api_setup
import tools.conversation_tools as conversation_tools

class Chat(BaseModel):
    query: str

class IntelliChat(BaseModel):
    query: str
    conversation_summary: str
    last_relevant_conversation: str

app = FastAPI()

# starttime = time()

# response = query_engine.query("Who are you? What is your purpose?")
# print(response)

# endtime = time()

# print(f"Time taken: {endtime - starttime} seconds")

@app.get("/query")
async def query(chat: str):
    return StreamingResponse(agent_setup.chunked_response(chat), media_type="text/plain")

@app.post("/intelli-chat")
async def intelli_chat(intelli_chat: IntelliChat):
    # Step 1: Determine what the user is currently thinking or asking about
    user_thoughts = conversation_tools.what_is_user_thinking(
        current_summary=intelli_chat.conversation_summary,
        last_relevant_conversation=intelli_chat.last_relevant_conversation,
        what_user_has_asked_now=intelli_chat.query
    )

    # Step 2: Write a prompt that includes the user's question, the summary of relevant context, and the agent's current thought chain
    prompt = f"User's Question: {intelli_chat.query}\nSummary of Relevant Context: {intelli_chat.conversation_summary}\nWhat User Might Be Thinking: {user_thoughts}\n\nBased on this information, generate a relevant answer to the user's question. You can look up more past conversations with tools as required. First try finding proper Instruction Set from the RAG index. If you find relevant instruction, use it to answer the question. If not, use your general knowledge to answer the question."
    
    return StreamingResponse(agent_setup.chunked_response(prompt, user_thoughts), media_type="text/plain")

@app.get("/status")
async def is_active():
    return {"status": True}

@app.get("/notes")
async def get_notes():
    return {"notes": api_setup.get_notes_list()}

# uvicorn main:app --reload --host 0.0.0.0 --port 8000 --timeout-keep-alive 1000