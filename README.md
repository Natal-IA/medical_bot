## Esquema

      edical_bot/
      ├─ README.md
      ├─ pyproject.toml              # o requirements.txt ?
      ├─ .env.example                
      ├─ .gitignore
      ├─ data/
      │  ├─ raw/                     # PDFs, docs originales (no sensibles)
      │  └─ processed/               # texto limpio
      ├─ docs/
      │  ├─ sources/                 # “fuentes aprobadas” por médicos (md/pdf)
      │  │   ├─ clinic_admin.md
      │  │   ├─ services.md
      │  │   ├─ urology_general.md
      │  │   ├─ red_flags.md
      │  │   ├─ privacy_notice.md
      │  ├─ policies/
      │  │  ├─ scope.md              # documento de alcance funcional del bot: qué responde / qué no
      │  │  ├─ red_flags.md          # urgencias / derivación
      │  │  └─ privacy.md            # aviso privacidad
      │  └─ prompts/
      │     ├─ system.md
      │     ├─ router.md
      │     └─ rag_answer.md
      ├─ vector_store/
      │  └─ chroma/                  # persistencia local Chroma (ignorar en git)
      ├─ src/
      │  ├─ app/
      │  │  ├─ main.py               # FastAPI entrypoint
      │  │  ├─ app.py
      │  │  ├─ api/
      │  │  │  ├─ routes_chat.py     # /chat
      │  │  │  └─ routes_health.py   # /health
      │  │  ├─ core/
      │  │  │  ├─ config.py          # settings (env vars)
      │  │  │  ├─ logging.py         # logging simple
      │  │  │  └─ safety.py          # guardrails + red flags
      │  │  ├─ services/
      │  │  │  ├─ retriever.py   
      │  │  │  ├─ llm_client.py      # wrapper OpenAI
      │  │  │  ├─ router.py          # intención: admin / rag / urgencia
      │  │  │  ├─ rag.py             # retrieval + respuesta
      │  │  │  └─ admin_faq.py       # respuestas deterministas (horarios, etc.)
      │  │  └─ schemas/
      │  │     ├─ chat.py            # Pydantic models request/response
      │  │     └─ feedback.py
      │  ├─ ingestion/
      │  │  ├─ build_kb.py           # ingesta docs -> chunks -> embeddings -> Chroma
      │  │  └─ loaders.py            # cargar pdf/md/txt
      │  └─ eval/
      │     ├─ eval_set.jsonl        # preguntas de test
      │     └─ run_eval.py           # eval simple (grounded/refusal)
      └─ tests/
         ├─ test_router.py
         ├─ test_safety.py
         └─ test_rag_smoke.py

## Flujo interno del retriever

      question (str)
         ↓
      embedding(question)
         ↓
      vector_store.query(top_k)
         ↓
      [List[Chunk]]  ← textos + metadatos + distancias