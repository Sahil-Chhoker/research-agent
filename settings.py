import os

from dotenv import load_dotenv
load_dotenv()

class Settings:
    OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
    LANGGRAPH_AES_KEY = os.getenv("LANGGRAPH_AES_KEY")
    LOG_FILE = os.getenv("LOG_FILE_PATH", "log.md")

settings = Settings()