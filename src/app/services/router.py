from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from src.app.core.safety import detect_red_flags, detect_admin_intent, red_flag_message
from src.app.services.rag import RAGService, RAGResult


@dataclass
class RoutedResponse:
    answer: str
    route: str  # "red_flag" | "admin" | "rag"
    sources: Optional[List[Dict[str, Any]]] = None
    debug: Optional[Dict[str, Any]] = None


class RouterService:
    """
    Decide por dónde responder:
    1) red_flag -> mensaje urgente seguro
    2) admin -> respuestas deterministas (POC) o RAG admin
    3) rag -> RAG normal
    """

    def __init__(self):
        self.rag = RAGService()

        # Mini-faq determinista (POC).crecerlo o moverlo a admin_faq.py
        self.admin_faq = {
          #  "horario": "Nuestro horario depende del doctor. Si me dices qué médico o clínica, te indico horarios y cómo pedir cita.",
         ##   "cita": "Puedes pedir cita indicando el médico y tu disponibilidad (días/horas). Si prefieres, facilita un teléfono y te indicamos el canal adecuado.",
         #   "ubicacion": "Dime qué consulta/médico y te indico la dirección exacta y cómo llegar.",
            "precio": "Los precios dependen del tipo de consulta y pruebas. Si me dices qué necesitas, te doy una orientación y cómo confirmarlo.",
            "seguro": "Trabajamos con algunas aseguradoras/mutuas según el médico. Dime cuál tienes y lo revisamos.",
        }

    def _admin_answer(self, text: str) -> str:
        t = text.lower()
        # matching súper simple
   #     if "horario" in t:
     #       return self.admin_faq["horario"]
     #   if "cita" in t or "reserv" in t:
      #      return self.admin_faq["cita"]
     #   if "ubic" in t or "direcc" in t or "cómo llegar" in t or "como llegar" in t:
       #     return self.admin_faq["ubicacion"]
        if "precio" in t or "tarifa" in t:
            return self.admin_faq["precio"]
        if "seguro" in t or "mutua" in t:
            return self.admin_faq["seguro"]

        # fallback admin:
        return "¿Me dices qué médico o clínica te interesa (son 3), y si buscas horarios, cita, ubicación o precios?"

    def handle(self, user_text: str, top_k: int = 5, debug: bool = False) -> RoutedResponse:
        # 1) Red flags primero
        safety = detect_red_flags(user_text)
        if safety.is_red_flag:
            dbg = {"matched_pattern": safety.matched_pattern, "reason": safety.reason} if debug else None
            return RoutedResponse(
                answer=red_flag_message(),
                route="red_flag",
                sources=None,
                debug=dbg,
            )

        # 2) Admin
        if detect_admin_intent(user_text):
            ans = self._admin_answer(user_text)
            return RoutedResponse(
                answer=ans,
                route="admin",
                sources=None,
                debug={"intent": "admin"} if debug else None,
            )

        # 3) RAG normal
        rag_result: RAGResult = self.rag.ask(user_text, top_k=top_k)
        sources = [
            {"source": c.source, "distance": c.distance}
            for c in rag_result.chunks
        ]
        return RoutedResponse(
            answer=rag_result.answer,
            route="rag",
            sources=sources,
            debug={"intent": "rag"} if debug else None,
        )
