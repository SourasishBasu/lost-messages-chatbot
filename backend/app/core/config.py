from functools import lru_cache
from pydantic_settings import BaseSettings
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    API_KEY: str = os.getenv("API_KEY")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
    REDIS_URL: str = os.getenv("REDIS_URL")
    REDIS_MIN_POOL_SIZE: int = 2
    REDIS_MAX_POOL_SIZE: int = 10

@lru_cache()
def get_settings():
    return Settings()