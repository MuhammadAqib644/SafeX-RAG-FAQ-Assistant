"""
FastAPI backend for the SafeX Solutions FAQ Chatbot.

Exposes both the Week 1 (TF-IDF) and Week 2-3 (RAG) bots behind one API,
so you can flip between them with a single query parameter and compare
results live — great for demoing the upgrade to your supervisor.

Run:
    uvicorn app:app --reload --port 8000

Then open http://localhost:8000 in your browser.
"""

import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.rag_bot import RagFAQBot

BASE_DIR = os.path.dirname(__file__)
FAQ_PATH = os.path.join(BASE_DIR, "data", "faqs.json")

app = FastAPI(title="SafeX Solutions FAQ Chatbot")

# embedding_backend: use "sentence-transformers" when you have real internet
# access (recommended for production). Falls back to "tfidf-svd" here because
# this environment can't reach huggingface.co.
EMBEDDING_BACKEND = os.environ.get("EMBEDDING_BACKEND", "tfidf-svd")
GENERATION_MODE = os.environ.get("GENERATION_MODE", "extractive")  # or "llm"

rag_bot = RagFAQBot(
    FAQ_PATH,
    generation_mode=GENERATION_MODE,
    embedding_backend=EMBEDDING_BACKEND,
)


class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
def chat(req: ChatRequest):
    return rag_bot.ask(req.message)


@app.get("/health")
def health():
    return {"status": "ok"}


# Serve the chat widget frontend
app.mount("/", StaticFiles(directory=os.path.join(BASE_DIR, "static"), html=True), name="static")
