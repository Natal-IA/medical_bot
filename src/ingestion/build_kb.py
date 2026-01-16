from __future__ import annotations

import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv
from openai import OpenAI

import chromadb

try:
    import tiktoken
except Exception:
    tiktoken = None

from src.ingestion.loaders import discover_sources, load_md, load_pdf, load_txt, LoadedDoc


def get_encoder(model_name: str):
    """
    tiktoken es una libreria que sabe tokenizar texto como lo hacen los modelos. La funcion devuelve none si no está instalada. 
    Intenta usar el encoder especifico del modelo y si no lo conoce usa el estandar cl100k_base que suele ir bien para modelos modernos
    
    """
    if tiktoken is None:
        return None
    try:
        return tiktoken.encoding_for_model(model_name)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def chunk_text_tokens(
    text: str,
    model_name: str,
    chunk_tokens: int,
    overlap: int,
) -> List[str]:
    
    """
    1) Obtienen el encoder
    2) Si no hay titoken hace fallback (mejor asi no, equivale 1 token a 4 caracteres y eso depende del idioma). Esto lo hace con ventana deslizante
    3) Si hay enc (modo bueno), hace chuck por tokens 
    """
   # enc = get_encoder(model_name)
    enc = None
    # Fallback si no hay tiktoken: chunk por caracteres (menos preciso, vale para POC)
    if enc is None:
        # Chunk semántico simple: por párrafos, juntando hasta un máximo de caracteres
        max_chars = max(1200, chunk_tokens * 4)        # aprox
        overlap_chars = max(200, overlap * 4)

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        chunks = []
        buf = ""

        for p in paragraphs:
            # si añadir el párrafo se pasa, cerramos chunk
            if len(buf) + len(p) + 2 > max_chars and buf.strip():
                chunks.append(buf.strip())
                # overlap: guardamos el final del chunk anterior
                buf = buf[-overlap_chars:] + "\n\n" + p
            else:
                buf = (buf + "\n\n" + p) if buf else p

        if buf.strip():
            chunks.append(buf.strip())

        return chunks

    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_tokens, len(tokens))
        chunk = enc.decode(tokens[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(tokens):
            break
        start = max(0, end - overlap)
    return chunks


def stable_id(source: str, chunk_index: int, chunk_text: str) -> str:
    """
    Crea un id unico estable para cada chunk usando un hash. Se mete el chunk_text porque si se edita el documento cambia el hash y eso permite detectar cambios y no confundir versiones
    Pero ojo con ids estables reingestar contenido dara error asi que para POC la soluciones es borrar carpeta chorma y reingestar
    """
    # ID estable: si cambias el texto, cambia el hash -> se reingesta
    h = hashlib.sha1(f"{source}|{chunk_index}|{chunk_text}".encode("utf-8")).hexdigest()
    return h


def load_one(path: Path) -> LoadedDoc:

    """
    Actua como switch y devuelve siempre el loadedDoc
    """
    ext = path.suffix.lower()
    if ext == ".md":
        return load_md(path)
    if ext == ".pdf":
        return load_pdf(path)
    if ext == ".txt":
        return load_txt(path)
    raise ValueError(f"Unsupported extension: {ext}")


def main():
    load_dotenv()

    # -----------------
    # Parametros
    # -----------------
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY no encontrada. Revisa tu .env")

    embed_model = os.getenv("MODEL_EMBED", "text-embedding-3-small")
    chroma_dir = os.getenv("CHROMA_DIR", "vector_store/chroma")

    chunk_tokens = int(os.getenv("CHUNK_TOKENS", "400"))
    overlap = int(os.getenv("CHUNK_OVERLAP", "60"))

    # -----------------
    # Descubre docs
    # -----------------
    sources_root = Path("docs/sources")
    if not sources_root.exists():
        raise RuntimeError("No existe docs/sources. Crea la carpeta y mete .md/.pdf para ingerir.")

    paths = discover_sources(sources_root)
    if not paths:
        raise RuntimeError("No he encontrado archivos .md/.pdf/.txt en docs/sources.")

    print(f"Encontrados {len(paths)} archivos en {sources_root}")

    # -----------------
    # Crea la base vectorial
    # -----------------
    chroma_path = Path(chroma_dir)
    chroma_path.mkdir(parents=True, exist_ok=True)

    chroma_client = chromadb.PersistentClient(path=str(chroma_path))
    collection = chroma_client.get_or_create_collection(
        name="medical_kb",
        metadata={"hnsw:space": "cosine"},
    )


    client = OpenAI(api_key=api_key)
    all_texts: List[str] = []
    all_metas: List[Dict[str, Any]] = []
    all_ids: List[str] = []

    for p in paths:
        doc = load_one(p)
        text = (doc.text or "").strip()
        if not text:
            print(f"Saltando vacío: {p}")
            continue

        # Chunking
        chunks = chunk_text_tokens(text, model_name=embed_model, chunk_tokens=chunk_tokens, overlap=overlap)
        print(chunks)

        for i, ch in enumerate(chunks):
            _id = stable_id(doc.source, i, ch)
            all_ids.append(_id)
            all_texts.append(ch)
            all_metas.append(
                {
                    "source": doc.source,
                    "doc_type": doc.doc_type,
                    "chunk_index": i,
                }
            )

    if not all_texts:
        raise RuntimeError("No se generaron chunks. Revisa que los documentos no estén vacíos.")

    print(f"Generados {len(all_texts)} chunks. Calculando embeddings con {embed_model}...")

    # Embeddings en batches (para ser robustos)
    BATCH = 128
    embeddings: List[List[float]] = []
    for i in range(0, len(all_texts), BATCH):
        batch_texts = all_texts[i : i + BATCH]
        resp = client.embeddings.create(model=embed_model, input=batch_texts)
        embeddings.extend([d.embedding for d in resp.data])

    assert len(embeddings) == len(all_texts)

    # Upsert a Chroma
    # Nota: Chroma usa add(); si un id ya existe puede fallar. Para POC,
    # lo más simple es borrar colección si reingestas mucho. Alternativa: try/except.
    # Aquí: intentamos add y si hay duplicados, lo indicamos.
    try:
        collection.add(
            ids=all_ids,
            documents=all_texts,
            metadatas=all_metas,
            embeddings=embeddings,
        )
        print(f"Ingesta completada. Total en colección: {collection.count()}")
    except Exception as e:
        print("No se pudo hacer add (posibles IDs duplicados por reingesta).")
        print("   Solución rápida para POC: borrar la colección y reingestar.")
        print("   Error:", repr(e))
        print("\nPara borrar y reingestar:")
        print(" - borra la carpeta vector_store/chroma o cambia el nombre de la colección.")
        raise


if __name__ == "__main__":
    main()
