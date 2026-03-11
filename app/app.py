from fastapi import FastAPI, HTTPException, File, UploadFile, Depends, Form 
import os 
import shutil 
from app.chatbot import Chatbot ## app.chatbot
from app.schemas import ChatRequest    ##app.schemas
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI()



@app.post("/chat/")
def chat(req: ChatRequest):
    # Here you would integrate your chatbot logic to generate a response based on the user_message
    # For demonstration, we'll just echo the user's message back
    Chat = Chatbot(user_message=req.user_message, profile_option=req.profile_option, session_id=req.session_id)
    response = Chat.generate_response()
    return {"response": response}


@app.post("/reset")
def reset(req: ChatRequest):
    """Reset memory for a given session_id."""
   # session_store.pop(req.session_id, None)
    return {"status": "cleared", "session_id": req.session_id}