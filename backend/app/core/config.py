import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "KARMA-AI Engine"
    API_V1_STR: str = "/api/v1"
    
    
    _CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    
    BASE_DIR: str = os.path.abspath(os.path.join(_CURRENT_FILE_DIR, "..", "..", ".."))
    
    
    MODELS_DIR: str = os.path.join(BASE_DIR, "models")
    
    GEMINI_API_KEY: str = "placeholder"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()