# api/settings.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    EMB_PATH: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMB_DIM: int = 384

    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "kb_en"

    OLLAMA_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "phi3:mini"  # ensure this model is pulled in Ollama

    HOST: str = "0.0.0.0"
    PORT: int = 8010

    model_config = SettingsConfigDict(
        env_file="../.env",
        env_file_encoding="utf-8",
    )

settings = Settings()
