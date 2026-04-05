from datetime import datetime
import os
import shutil

from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import uuid

import setup.agent_setup as agent_setup
import setup.api_setup as api_setup
import setup.speech_to_text as speech_to_text

class Chat(BaseModel):
    query: str

class IntelliChat(BaseModel):
    query: str
    conversation_summary: str
    last_relevant_conversation: str

class VoiceResponse(BaseModel):
    text: str

app = FastAPI()

os.environ['ORT_LOGGING_LEVEL'] = '3'

# starttime = time()

# response = query_engine.query("Who are you? What is your purpose?")
# print(response)

# endtime = time()

# print(f"Time taken: {endtime - starttime} seconds")

# @app.get("/query")
# async def query(chat: str):
#     return StreamingResponse(agent_setup.chunked_response(chat), media_type="text/plain")

@app.post("/intelli-chat")
async def intelli_chat(intelli_chat: IntelliChat):
    print(f"Received query: {intelli_chat}")

    user_thoughts = agent_setup.ask_llm_for_summary_of_thoughts(intelli_chat.query, intelli_chat.conversation_summary, intelli_chat.last_relevant_conversation)
    
    return StreamingResponse(agent_setup.chunked_response({"input": intelli_chat.query, "summary_of_context": user_thoughts, "last_relevant_conversation": intelli_chat.last_relevant_conversation, "user_thoughts": user_thoughts}, intelli_chat.conversation_summary), media_type="text/plain")

@app.post("/speech-to-text")
async def speech_to_text_endpoint(file: UploadFile = File(...)):
    return VoiceResponse(text=speech_to_text.transcribe_wav(file))

@app.post("/text-to-speech")
async def text_to_speech_endpoint(voice_response: VoiceResponse):
    gen_file = speech_to_text.speakInShukaVoice(voice_response.text, "am_michael")
    return FileResponse(gen_file, media_type="audio/mp3", filename="speech.mp3")

# @app.post("/clone-voice-and-speak")
# async def clone_voice_and_speak(text: str = Form(...), source_voice_file: UploadFile = File(...)):
#     file_id = uuid.uuid4()

#     # Save the uploaded voice file temporarily
#     source_voice_path = f"audio/source_voice_{file_id}.wav"
#     with open(source_voice_path, "wb") as buffer:
#         shutil.copyfileobj(source_voice_file.file, buffer)

#     speech_to_text.convert_to_mp3(source_voice_path)
#     source_voice_path = source_voice_path.replace(".wav", ".mp3")

#     gen_file = speech_to_text.speakInVoice(text, "en", source_voice_path, file_id)
#     return FileResponse(gen_file, media_type="audio/mp3", filename="cloned_speech.mp3")

@app.get("/status")
async def is_active():
    return {"status": True}

@app.get("/notes")
async def get_notes():
    return {"notes": api_setup.get_notes_list()}

# uvicorn main:app --reload --host 0.0.0.0 --port 8000 --timeout-keep-alive 1000