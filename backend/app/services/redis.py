from contextlib import asynccontextmanager
from typing import List, Dict, Optional
import json
from redis.asyncio import ConnectionPool, Redis
from fastapi import FastAPI
from app.core.config import get_settings
from app.core.logging import setup_logger

logger = setup_logger()
settings = get_settings()

async def init_redis_pool() -> ConnectionPool:
    """Initialize Redis connection pool"""
    try:
        pool = ConnectionPool.from_url(
            settings.REDIS_URL,
            max_connections=settings.REDIS_MAX_POOL_SIZE,
            decode_responses=True
        )
        # Test the connection
        async with Redis(connection_pool=pool) as redis:
            await redis.ping()
        logger.info("Redis connection pool initialized successfully")
        return pool
    except Exception as e:
        logger.error(f"Failed to initialize Redis pool: {str(e)}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for Redis pool initialization and cleanup"""
    # Startup: Initialize pool
    app.state.redis_pool = await init_redis_pool()
    logger.info("Redis pool added to app state")
    
    yield
    
    # Shutdown: Cleanup pool
    await app.state.redis_pool.disconnect()
    logger.info("Redis pool closed")

class RedisService:
    def __init__(self, pool: ConnectionPool):
        self.pool = pool

    async def get_team_history(self, team_id: str, max_history: int = 10) -> List[Dict]:
        """Retrieve chat history for a team from Redis"""
        history_key = f"chat_history:{team_id}"
        try:
            async with Redis(connection_pool=self.pool) as redis:
                history = await redis.lrange(history_key, 0, max_history - 1)
                return [json.loads(msg) for msg in history] if history else []
        except Exception as e:
            logger.error(f"Error retrieving history for team {team_id}: {str(e)}")
            return []

    async def save_to_history(self, team_id: str, message: Dict, max_history: int = 10):
        """Save a message to team's chat history"""
        history_key = f"chat_history:{team_id}"
        try:
            async with Redis(connection_pool=self.pool) as redis:
                await redis.lpush(history_key, json.dumps(message))
                await redis.ltrim(history_key, 0, max_history - 1)
        except Exception as e:
            logger.error(f"Error saving history for team {team_id}: {str(e)}")