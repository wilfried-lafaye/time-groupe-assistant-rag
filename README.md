# Time Groupe Virtual Assistant

Assistant virtuel basé sur la technologie RAG (Retrieval-Augmented Generation) pour [Time Groupe](https://timegroupe.ca), entreprise d'événementiel à Montréal.

## Technologies

- Backend API: FastAPI, Uvicorn
- Orchestration IA: LangChain
- Base vectorielle: ChromaDB
- Modèle LLM: OpenAI GPT-4o-mini

## Installation

1. Cloner le dépôt et accéder au dossier:
```bash
git clone https://github.com/wilfried-lafaye/tampon.git
cd tampon
```

2. Créer et activer l'environnement virtuel:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Installer les dépendances:
```bash
pip install -r requirements.txt
```

4. Configurer les variables d'environnement:
```bash
cp .env.example .env
# Editez .env et ajoutez votre clé OpenAI
```

## Lancement

Démarrer le serveur local de l'API:
```bash
uvicorn main:app --reload
```

- Interface utilisateur: http://localhost:8000
- Documentation API: http://localhost:8000/docs

## Utilisation

Endpoint par défaut de l'API:
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Quels types d événements organisez-vous ?"}'
```

## Architecture

- `data/time_groupe_info.txt`: Base de connaissances
- `static/index.html`: Interface utilisateur Frontend
- `main.py`: Serveur API FastAPI
- `rag_engine.py`: Architecture RAG (Chargement, Découpage, Stockage, Recherche)


