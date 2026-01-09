from pathlib import Path
from dotenv import load_dotenv
import json
from datetime import datetime

import os
# Cargar .env desde la raíz del repo
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
from src.app.services.rag import RAGService  # noqa: E402
RESULTS_PATH = PROJECT_ROOT / "experiments" / "results" / "rag_runs.jsonl"
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)



def save_result(path, payload):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

def main():
    rag = RAGService()

    questions = [
        "soy mujer y quiero ir al urologo"
    ]

    for q in questions:
        print("\n==============================")
        print("Q:", q)

        result = rag.ask(q, top_k=5)

        record = {
        "timestamp": datetime.utcnow().isoformat(),
        "question": q,
        "answer": result.answer,
        "sources": [
            {
                "source": c.source,
                "distance": c.distance
            }
            for c in result.chunks
        ],
        "top_k": 5,
        "model_chat": rag.chat_model,
        "model_embed": os.getenv("MODEL_EMBED"),
        }

        save_result(RESULTS_PATH, record)

if __name__ == "__main__":
    main()
