from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi import FastAPI
import os
import time
from app.core.security import verify_api_key
from app.models.chat import ChatCompletionRequest, ChatCompletion, ChatChoice, ChatMessage
from app.services.bot import GeminiMysteryBot
from app.core.logging import setup_logger

router = APIRouter()
logger = setup_logger()
#bot = GeminiMysteryBot()

@router.post("/chat/completions", response_model=ChatCompletion)
async def chat_endpoint(
    request: ChatCompletionRequest,
    req: Request,
    secret: str = Depends(verify_api_key),
):
    try:
        completed_games = []
        team_id = None
        
        for msg in request.messages:
            if msg.role == "system":
                content = msg.content
                try:
                    if "completed_games" in content:
                        completed_games = eval(content.split("completed_games=")[1].split("]")[0] + "]")
                    if "team_id=" in content:
                        team_id = content.split("team_id=")[1].split()[0]
                except:
                    pass
                break

        print(f"Completed games: {completed_games}, Team ID: {team_id}")
        
        # Initialize bot with Redis pool from app state
        bot = GeminiMysteryBot(req.app.state.redis_pool)
        
        response_text, triggered_question = await bot.get_response(
            [{"role": msg.role, "content": msg.content} for msg in request.messages],
            completed_games,
            team_id
        )
        
        return ChatCompletion(
            id=f"chatcmpl-{os.urandom(4).hex()}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=f"{response_text}\n\n[Ques: {triggered_question}]"
                    )
                )
            ],
            usage={
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")