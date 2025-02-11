from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import get_settings
from app.routes import chat
from app.core.logging import setup_logger
from app.services.redis import lifespan

logger = setup_logger()
settings = get_settings()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/v1")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}