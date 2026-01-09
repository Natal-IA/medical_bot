from __future__ import annotations

from dataclasses import dataclass
from typing import List
import os

from openai import OpenAI

from src.app.services.retriever import Retriever, RetrievedChunk


@dataclass
class RAGResult:
    answer: str
    chunks: List[RetrievedChunk]


class RAGService:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY no encontrada en entorno/.env")

        self.chat_model = os.getenv("MODEL_CHAT", "gpt-4o-mini") #MODELO DE GENERACION
        self.client = OpenAI(api_key=api_key) #cliente LLM
        self.retriever = Retriever() #buscador semantico 

    def ask(self, question: str, top_k: int = 5) -> RAGResult:
        """
        Entra una pregunta y sale un RAGResult
        1) Se embebe la preunta y se consulta a chroma y devuelve el retrievedchunk donde cada chunk tiene text, source, distance
        Todo lo que viene después ya es prompting
        2) Contruccion de contexto, es decir, crear un bloque de texto que el LLM va a "leer" => le estoy dando contexto controlado
        3) prompt system => controla el rol, tono, límites (no diagnostiques), seguridad (urgencias), y la regla clave de usar solo => esto es policy médica => contexto
        4) prompt user => se construye el promt donde separas claramente CONTEXTO, PREGUNTA INSTRUCCIONES para reducir alucinaciones y respuestas creativas
        5) Llamada al modelo con temperature baja para respuestas mas estables todo el conocimiento debe venir el context
        6) Contrucción del resultado 
        PREGUNTA
        ↓
        RETRIEVER (embeddings + top-k)
        ↓
        CONTEXTO CONTROLADO
        ↓
        LLM (redactor, no experto)
        ↓
        RESPUESTA + FUENTES
        """
        chunks = self.retriever.retrieve(question, top_k=top_k)
        context = "\n\n".join(
            [f"[{i+1}] SOURCE: {c.source}\n{c.text}" for i, c in enumerate(chunks)]
        )

        system = (
            "Eres un asistente de una clínica de urología. "
            "Respondes con tono claro y profesional. "
            "No das diagnósticos ni tratamiento personalizado. "
            "Si hay señales de urgencia, recomiendas acudir a urgencias.\n"
            "Responde SOLO usando el CONTEXTO proporcionado. "
            "Si falta información, di que no lo sabes y sugiere pedir cita."
        )

        user = (
            f"CONTEXTO:\n{context}\n\n"
            f"PREGUNTA:\n{question}\n\n"
            "INSTRUCCIONES:\n"
            "- Responde en español.\n"
            "- Sé breve y útil.\n"
            "- Si el contexto no contiene la respuesta, dilo claramente.\n"
        )

        resp = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )

        answer = resp.choices[0].message.content.strip()
        return RAGResult(answer=answer, chunks=chunks)
