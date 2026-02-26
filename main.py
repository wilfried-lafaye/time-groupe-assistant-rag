"""
Point d'entrée de l'API FastAPI.
Commande de lancement: uvicorn main:app --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from rag_engine import ask_question

app = FastAPI(
    title="Time Groupe Virtual Assistant",
    description="API RAG pour répondre aux questions sur les services de Time Groupe.",
    version="1.0.0",
)


class QuestionRequest(BaseModel):
    """Schéma de validation pour les questions entrantes."""
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="La question à traiter par le modèle.",
        examples=["Quels types d'événements organisez-vous ?"],
    )


class AnswerResponse(BaseModel):
    """Schéma de validation pour les réponses de l'API."""
    question: str
    answer: str


@app.get("/")
def root():
    """Sert l'interface de chat frontend."""
    return FileResponse("static/index.html")


@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    """
    Traite une question reçue via POST et retourne la réponse générée.
    """
    try:
        answer = ask_question(request.question)
        return AnswerResponse(question=request.question, answer=answer)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur interne lors du traitement : {str(e)}",
        )


app.mount("/static", StaticFiles(directory="static"), name="static")
