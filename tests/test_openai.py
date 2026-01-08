from dotenv import load_dotenv
import os
from openai import OpenAI

# 1) Cargar variables del .env
load_dotenv()

# 2) Comprobar que la key existe
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY no encontrada. Revisa .env")

# 3) Crear cliente
client = OpenAI(api_key=api_key)

# 4) Llamada mínima al modelo
response = client.chat.completions.create(
    model=os.getenv("MODEL_CHAT", "gpt-4o-mini"),
    messages=[
        {"role": "user", "content": "Di hola y dime que funcionas correctamente"}
    ],
    temperature=0.2,
)

# 5) Mostrar respuesta
print("Respuesta del modelo:")
print(response.choices[0].message.content)
