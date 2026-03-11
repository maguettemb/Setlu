from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory


session_store: Dict[str, ChatMessageHistory] = {}

def get_history_from_config(cfg: Any) -> ChatMessageHistory:
    session_id = None
    if isinstance(cfg, dict):
        cfg_conf = cfg.get("configurable") if isinstance(cfg.get("configurable"), dict) else {}
        session_id = cfg_conf.get("session_id")
    if not session_id:
        session_id = "default"
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]

