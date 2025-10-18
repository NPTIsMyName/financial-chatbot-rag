import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag_handler import RAGHandler


class ChatRequest(BaseModel):
    session_id: str
    message: str




def create_app() -> FastAPI:
    load_dotenv()

    app = FastAPI(title="Financial News Chatbot (FastAPI)")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # App state
    app.state.rag_handler = RAGHandler()

    @app.on_event("startup")
    def _startup() -> None:
        # Initialize RAG handler
        app.state.rag_handler.initialize()

    # Static files for demo UI
    if os.path.isdir("static"):
        app.mount("/static", StaticFiles(directory="static"), name="static")

        @app.get("/")
        def index():
            return FileResponse("static/index.html")

    @app.post("/api/chat")
    def chat(body: ChatRequest):
        if app.state.rag_handler.vectorstore is None:
            raise HTTPException(status_code=503, detail="Vector store not initialized.")

        try:
            # Always use RAG path
            answer = app.state.rag_handler.process_rag_query(body.session_id, body.message)
            return {"answer": answer}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


app = create_app()


