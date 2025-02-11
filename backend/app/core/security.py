from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from app.core.config import get_settings

settings = get_settings()

async def verify_api_key(authorization: str = Security(APIKeyHeader(name="Authorization"))):
    secret = authorization[len("Bearer "):]
    if secret != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return secret
