from dotenv import load_dotenv
import os

load_dotenv()

print("KEY:", os.getenv("OPENAI_API_KEY")[:5])
print("MODEL:", os.getenv("MODEL_CHAT"))
