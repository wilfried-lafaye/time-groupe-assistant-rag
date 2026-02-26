"""
Moteur RAG (Retrieval-Augmented Generation).
Ce module gère le chargement des connaissances, la vectorisation,
et la recherche sémantique via LangChain et ChromaDB.
"""

import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

load_dotenv()

DATA_PATH = "data/time_groupe_info.txt"
CHROMA_DB_DIR = "chroma_db"


def build_vectorstore():
    """
    Charge le fichier texte de référence, le divise en segments,
    et génère la base de données vectorielle locale.
    """
    loader = TextLoader(DATA_PATH, encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(documents)
    print(f"[{len(chunks)} chunks created from {DATA_PATH}]")

    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR,
    )
    print(f"[Vector database created in {CHROMA_DB_DIR}]")

    return vectorstore


def get_vectorstore():
    """Charge la base vectorielle existante ou la construit."""
    if not os.path.exists(CHROMA_DB_DIR):
        return build_vectorstore()

    embeddings = OpenAIEmbeddings()
    return Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings,
    )


def ask_question(question: str) -> str:
    """
    Interroge le LLM en lui fournissant le contexte pertinent
    extrait de la base vectorielle.
    """
    vectorstore = get_vectorstore()

    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    )

    result = qa_chain.invoke({"query": question})
    return result["result"]


if __name__ == "__main__":
    test_question = "Quels types d'événements organise Time Groupe ?"
    response = ask_question(test_question)
    print(f"Question: {test_question}")
    print(f"Reponse: {response}")
