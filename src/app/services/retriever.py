from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
import os

import chromadb
from openai import OpenAI

"""
Este modulo es un retriever semántico, que usa embeddings de OpenAI, y usa ChromaDB para buscar infor relevante en una consulta. Para
ello busca fragmentos de texto "chunks" relevantes para una pregunta dada, usando una búsqueda semántica (similitud de significado, no palabras exactas)

"""
@dataclass
class RetrievedChunk:
    text: str
    source: str
    distance: float
    metadata: Dict[str, Any]
"""
Esta es una clase simple que representa un fragmento recuperado:
- text: el contenido del fragmento
- source: de donde viene
- distance: similitud con la consulta (menor mas similar)
- metadata: metadatos adicionales

"""

class Retriever:
    """
    la clase principal con responsabilidad única: convertir las preguntas en embeddings y buscar chunks relevantes y devolverlos
    retriever = Retriever()
    chunks = retriever.retrieve("qué es la urologia", top_k=5)
    """

    def __init__(
        self,
        chroma_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
        embed_model: Optional[str] = None,
    ):
        # Resolver rutas respecto a raíz del repo (no respecto al CWD)
        project_root = Path(__file__).resolve().parents[3]  # .../src/app/services -> repo root
        chroma_dir = chroma_dir or os.getenv("CHROMA_DIR", "vector_store/chroma")
        self.chroma_path = str((project_root / chroma_dir).resolve())

        self.collection_name = collection_name or os.getenv("CHROMA_COLLECTION", "medical_kb")
        self.embed_model = embed_model or os.getenv("MODEL_EMBED", "text-embedding-3-small")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY no encontrada en entorno/.env")

        self.oa = OpenAI(api_key=api_key)

        self.chroma = chromadb.PersistentClient(path=self.chroma_path)
        # IMPORTANTE: usa get_collection (no create) para no crear vacías por error
        self.col = self.chroma.get_collection(self.collection_name)

    def retrieve(self, question: str, top_k: int = 5) -> List[RetrievedChunk]:
        q_emb = self.oa.embeddings.create(
            model=self.embed_model,
            input=question,
        ).data[0].embedding

        res = self.col.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        chunks: List[RetrievedChunk] = []
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]

        for doc_text, meta, dist in zip(docs, metas, dists):
            chunks.append(
                RetrievedChunk(
                    text=doc_text,
                    source=meta.get("source", ""),
                    distance=float(dist),
                    metadata=meta,
                )
            )

        return chunks
