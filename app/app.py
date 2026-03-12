
"""
app.py
-------
FastAPI backend for MaguetteAgent chatbot.

Endpoints:
  POST /chat          → standard response
  POST /chat/stream   → streaming response (SSE)
  GET  /health        → health check

Usage:
    uvicorn app.app:app --reload --port 8000
"""


from fastapi import FastAPI, HTTPException, File, UploadFile, Depends, Form 
import os 
import shutil 
from app.chatbot import Chatbot ## app.chatbot
from app.schemas import ChatRequest, ChatResponse    ##app.schemas
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import json

from app.chatbot import Chatbot, VALID_PROFILES

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="MaguetteAgent API",
    description="RAG-powered chatbot API for Maguette MBAYE's professional profile",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# CORS — allow React dev server
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "https://setlu-frontend.up.railway.app", "https://maguettemb.github.io/Setlu/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "agent": "MaguetteAgent"}


@app.get("/profiles")
def get_profiles():
    """Return the list of available profiles."""
    return {"profiles": sorted(VALID_PROFILES)}

#@app.post("/chat/")
#def chat(req: ChatRequest):
#    # Here you would integrate your chatbot logic to generate a response based on the user_message
#    # For demonstration, we'll just echo the user's message back
#    Chat = Chatbot(user_message=req.user_message, profile_option=req.profile_option, session_id=req.session_id)
#    response = Chat.generate_response()
#    return {"response": response}


#@app.post("/reset")
##def reset(req: ChatRequest):
 #   """Reset memory for a given session_id."""
 #  # session_store.pop(req.session_id, None)
 #   return {"status": "cleared", "session_id": req.session_id}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Standard (non-streaming) chat endpoint."""
    if request.profile not in VALID_PROFILES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid profile '{request.profile}'. Choose from: {sorted(VALID_PROFILES)}"
        )
    try:
        bot = Chatbot(
            user_message=request.message,
            profile_option=request.profile,
            session_id=request.session_id,
        )
        response = bot.generate_response()
        return ChatResponse(
            response=response,
            session_id=request.session_id,
            profile=request.profile,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    """Streaming chat endpoint — returns Server-Sent Events (SSE).
    
    Each event is a JSON object: {"token": "...", "done": false}
    Final event:              {"token": "",    "done": true}
    """
    if request.profile not in VALID_PROFILES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid profile '{request.profile}'. Choose from: {sorted(VALID_PROFILES)}"
        )

    def generate():
        try:
            bot = Chatbot(
                user_message=request.message,
                profile_option=request.profile,
                session_id=request.session_id,
            )
            for chunk in bot.stream_answer():
                payload = json.dumps({"token": chunk, "done": False})
                yield f"data: {payload}\n\n"
            # Signal end of stream
            yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )