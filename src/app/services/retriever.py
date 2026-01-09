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
    def __init__(self):
        chroma_dir = os.getenv("CHROMA_DIR", "vector_store/chroma")
        self.collection_name = os.getenv("CHROMA_COLLECTION", "medical_kb")

        chroma_path = Path(chroma_dir)
        if not chroma_path.exists():
            raise RuntimeError(
                f"CHROMA_DIR no existe: {chroma_path.resolve()} "
                f"(¿se ha subido el vector_store al repo?)"
            )

        self.chroma = chromadb.PersistentClient(path=str(chroma_path))

        # Diagnóstico: qué colecciones hay realmente
        cols = self.chroma.list_collections()
        names = [c.name for c in cols]
        print("[Chroma] path:", str(chroma_path.resolve()))
        print("[Chroma] collections:", names)
        print("[Chroma] requested:", self.collection_name)

        if self.collection_name not in names:
            raise RuntimeError(
                f"No existe la colección '{self.collection_name}' en esta BD Chroma.\n"
                f"Colecciones disponibles: {names}\n"
                f"Ruta: {chroma_path.resolve()}\n"
                f"Solución: subir el vector_store correcto o reingestar en Cloud."
            )

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
