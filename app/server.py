"""FastAPI web server wrapping the RAG chatbot for browser access.

Transport layer only — all model logic lives in RAGChain. The server is
stateless per request: each question is answered independently (no shared
conversation memory across concurrent users).
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app import config
from app.chat import load_store, validate_query
from app.chain import RAGChain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="International Student Assistant")

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "You've asked a lot of questions! Please wait a bit before asking more."},
    )


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Single shared RAG instance (stateless usage only — see answer_stream_stateless).
_rag: RAGChain | None = None


def get_rag() -> RAGChain:
    global _rag
    if _rag is None:
        _rag = RAGChain(load_store())
    return _rag


@app.get("/")
def home():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/chat/stream")
@limiter.limit(config.RATE_LIMIT)
def chat_stream(request: Request, q: str = ""):
    question = validate_query(q)
    if question is None:
        raise HTTPException(status_code=400, detail="Please enter a question.")

    rag = get_rag()

    def event_generator():
        try:
            docs = rag.retrieve(question)
            for piece in rag.answer_stream_stateless(question, docs=docs):
                yield {"event": "token", "data": piece}
            # Send unique sources once, after the answer.
            seen = set()
            sources = []
            for d in docs:
                url = d.metadata.get("source_url", "")
                if url and url not in seen:
                    seen.add(url)
                    sources.append({
                        "source_name": d.metadata.get("source_name", ""),
                        "source_url": url,
                        "fetched_at": d.metadata.get("fetched_at", ""),
                        "topic": d.metadata.get("topic", ""),
                    })
            yield {"event": "sources", "data": json.dumps(sources)}
            yield {"event": "done", "data": ""}
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error answering question: %s", exc)
            yield {
                "event": "error",
                "data": "I'm having trouble right now, please try again in a moment.",
            }

    return EventSourceResponse(event_generator())


class Feedback(BaseModel):
    question: str
    answer: str
    rating: str


@app.post("/feedback")
def feedback(item: Feedback):
    if item.rating not in ("up", "down"):
        raise HTTPException(status_code=400, detail="rating must be 'up' or 'down'")
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "question": item.question[:1000],
        "answer": item.answer[:5000],
        "rating": item.rating,
    }
    with open(config.FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"ok": True}
