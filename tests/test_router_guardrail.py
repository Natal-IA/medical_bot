from pathlib import Path
from dotenv import load_dotenv
import json
import os
from datetime import datetime

# 1) Cargar .env desde la raíz del repo (antes de importar servicios)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

from src.app.services.router import RouterService  # noqa: E402


RESULTS_PATH = PROJECT_ROOT / "experiments" / "results" / "router_runs.jsonl"
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)


def save_result(path: Path, payload: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main():
    router = RouterService()

    # Preguntas representativas: admin, general, y red_flags
    questions = [
        # Admin
        "¿Cuál es el horario de la clínica?",
        "¿Cómo puedo pedir cita?",
        "¿Dónde estáis ubicados?",
        "¿Trabajáis con seguro o mutua?",
        "¿Cuánto cuesta una primera consulta?",

        # General
        "¿Qué hace un urólogo?",
        "¿Qué es la próstata y para qué sirve?",
        "¿Qué pruebas suelen hacer en urología?",

        # Red flags (triage)
        "Tengo fiebre y no puedo orinar, ¿qué hago?",
        "Me duele muchísimo el testículo y está hinchado, ¿es urgente?",
        "He visto sangre en la orina, ¿qué hago?",
    ]

    for q in questions:
        out = router.handle(q, top_k=5, debug=True)

        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "question": q,
            "route": out.route,
            "answer": out.answer,
            "sources": out.sources,     # None si admin/red_flag
            "debug": out.debug,         # para entender por qué enrutó así
            "models": {
                "chat": os.getenv("MODEL_CHAT", "gpt-4o-mini"),
                "embed": os.getenv("MODEL_EMBED", "text-embedding-3-small"),
            },
            "top_k": 5,
        }

        save_result(RESULTS_PATH, record)

        # Print amigable por consola
        print("\n==============================")
        print("Q:", q)
        print("ROUTE:", out.route)
        print("A:", out.answer[:400], "..." if len(out.answer) > 400 else "")
        if out.sources:
            print("SOURCES:")
            for i, s in enumerate(out.sources, start=1):
                print(f"  [{i}] {s.get('source')} (dist={s.get('distance'):.4f})")
        if out.debug:
            print("DEBUG:", out.debug)

        print("→ Guardado en", RESULTS_PATH)


if __name__ == "__main__":
    main()
