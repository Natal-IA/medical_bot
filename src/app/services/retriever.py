from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import os
from pathlib import Path

import chromadb
from openai import OpenAI


@dataclass
class RetrievedChunk:
    text: str
    source: str
    distance: float


class Retriever:
    def __init__(self):
        # --- Config ---
        chroma_dir = os.getenv("CHROMA_DIR", "vector_store/chroma")
        self.collection_name = os.getenv("CHROMA_COLLECTION", "medical_kb")
        self.embed_model = os.getenv("MODEL_EMBED", "text-embedding-3-small")

        # --- OpenAI client (esto te faltaba) ---
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY no encontrada (Secrets/.env).")
        self.client = OpenAI(api_key=api_key)

        # --- Chroma ---
        chroma_path = Path(chroma_dir)
        if not chroma_path.exists():
            raise RuntimeError(f"CHROMA_DIR no existe: {chroma_path.resolve()}")

        self.chroma = chromadb.PersistentClient(path=str(chroma_path))
        # Fail fast: la colección debe existir (si no, tu store no está bien)
        cols = [c.name for c in self.chroma.list_collections()]
        if self.collection_name not in cols:
            raise RuntimeError(
                f"No existe la colección '{self.collection_name}'. "
                f"Disponibles: {cols}. Ruta: {chroma_path.resolve()}"
            )

        self.col = self.chroma.get_collection(self.collection_name)

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        # 1) Embed de la query
        q_emb = self.client.embeddings.create(
            model=self.embed_model,
            input=query,
        ).data[0].embedding

        # 2) Query en Chroma por embeddings (NO query_texts)
        res = self.col.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]

        out: List[RetrievedChunk] = []
        for doc, meta, dist in zip(docs, metas, dists):
            out.append(
                RetrievedChunk(
                    text=doc,
                    source=(meta or {}).get("source", "unknown"),
                    distance=float(dist),
                )
            )
        return out
