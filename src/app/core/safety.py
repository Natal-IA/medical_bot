from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import re


# --- Palabras/frases típicas de red flags urológicas (POC) ---
# Nota: no es un sistema médico. Es triage conservador.
RED_FLAG_PATTERNS: List[str] = [
    r"\bno puedo orinar\b",
    r"\bretenci[oó]n urinaria\b",
    r"\bno me sale (la )?orina\b",
    r"\b(orina|pip[ií])\s*con\s*sangre\b",
    r"\bhematuria\b",
    r"\bsangre\s+en\s+la\s+orina\b",
    r"\bdolor\s+intenso\b",
    r"\bdolor\s+muy\s+fuerte\b",
    r"\bdolor\s+testicular\b",
    r"\btest[ií]culo\s+hinchado\b",
    r"\bhinchaz[oó]n\s+testicular\b",
    r"\bfiebre\b",
    r"\b40\s*º\b",
    r"\b39\s*º\b",
    r"\b38\s*º\b",
    r"\bescalofr[ií]os\b",
    r"\bn[aá]useas\b",
    r"\bv[oó]mitos\b",
    r"\bincapaz\b.*\borinar\b",
    r"\b(dolor|ardor)\s+al\s+orinar\b.*\bfiebre\b",
    r"\bsepsis\b",
]

# Admin intents típicos
ADMIN_PATTERNS: List[str] = [
    r"\bhorario(s)?\b",
    r"\bcita(s)?\b",
    r"\breserva(r)?\b",
    r"\bprecio(s)?\b",
    r"\btarifa(s)?\b",
    r"\bubicaci[oó]n\b",
    r"\bdirecci[oó]n\b",
    r"\bcomo llegar\b",
    r"\bseguro\b",
    r"\bmutua\b",
    r"\bprivado\b",
    r"\btelefono\b",
    r"\bemail\b",
    r"\bwhatsapp\b",
    r"\burgencia(s)?\b",
]


@dataclass
class SafetyResult:
    is_red_flag: bool
    reason: Optional[str] = None
    matched_pattern: Optional[str] = None


def _match_any(text: str, patterns: List[str]) -> Optional[str]:
    t = text.lower()
    for p in patterns:
        if re.search(p, t, flags=re.IGNORECASE):
            return p
    return None


def detect_red_flags(user_text: str) -> SafetyResult:
    """
    Detecta señales de alarma (triage conservador).
    """
    matched = _match_any(user_text, RED_FLAG_PATTERNS)
    if matched:
        return SafetyResult(
            is_red_flag=True,
            reason="Se detectan posibles signos de alarma/urgencia en el mensaje.",
            matched_pattern=matched,
        )
    return SafetyResult(is_red_flag=False)


def detect_admin_intent(user_text: str) -> bool:
    """
    Heurística simple para detectar intención administrativa.
    """
    return _match_any(user_text, ADMIN_PATTERNS) is not None


def red_flag_message() -> str:
    """
    Mensaje seguro y conservador. Ajusta a tu contexto (España).
    """
    return (
        "Por lo que describes, podría tratarse de una situación que requiere valoración médica urgente.\n\n"
        "- Si no puedes orinar, tienes fiebre alta, dolor intenso o sangre en la orina, te recomiendo acudir a **urgencias** lo antes posible.\n"
        "- Si los síntomas son graves o empeoran, llama a emergencias (112 en España).\n\n"
        "Si te encuentras estable, también puedes solicitar una cita, pero ante estos síntomas la prioridad es una valoración urgente."
    )
