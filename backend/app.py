from fastapi import FastAPI, HTTPException, Depends, Security
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Literal, Dict, Any
from fastapi.security import APIKeyHeader
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from enum import Enum, auto
import google.generativeai as genai
import logging
import os
import sys
import time

# OpenAI Compatible Request/Response Models
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"] = Field(default="user")
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = Field(default="gemini-1.5-flash")
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(default=0.9)
    max_tokens: Optional[int] = Field(default=150)

class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = "stop"

class ChatCompletion(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Dict[str, int]

def setup_logger():
    logger = logging.getLogger('chatbot_logger')
    formatter = logging.Formatter('%(levelname)s - [%(asctime)s] - %(message)s', datefmt='%d/%b/%Y %H:%M:%S')
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger

logger = setup_logger()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

API_KEY = os.getenv("API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

class GameState(Enum):
    GAME_1 = 1 # doctor
    GAME_2 = 2 # friend
    GAME_3 = 3
    GAME_4 = 4
    GAME_5 = 5
    GAME_6 = 6
    GAME_7 = 7


class ClueInfo:
    def __init__(self, answer: str, required_game: GameState, points: int):
        self.answer = answer
        self.required_game = required_game
        self.points = points

class GeminiMysteryBot:
    def __init__(self):
        try:
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("MysteryBot initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MysteryBot: {str(e)}")
            raise

        self.clue_pairs = {
            "where are the medicines?": ClueInfo("kitchen", GameState.GAME_1, 20),
            "what is the doctor's name?": ClueInfo("John", GameState.GAME_1, 15),
            "when were the medicines taken?": ClueInfo("yesterday", GameState.GAME_1, 10),

            "who is the friend?": ClueInfo("Sam", GameState.GAME_2, 30),
            "what was the medicine dosage?": ClueInfo("low", GameState.GAME_2, 15),
            "what did the friend leak?": ClueInfo("photos", GameState.GAME_2, 10),

            "What was in the email?": ClueInfo("Confidential Documents", GameState.GAME_3, 40),
            "What was the name of the company?": ClueInfo("Huli Technologies", GameState.GAME_3, 15),
            "What was the victim's position in the company?": ClueInfo("CTO", GameState.GAME_3, 10),

            "Which family member threatened the victim?": ClueInfo("Brother in Law", GameState.GAME_4, 50),
            "What was the threat from family member?": ClueInfo("Inheritance Dispute", GameState.GAME_4, 15),
            "How much was the victim going to inherit?": ClueInfo("1 Million USD", GameState.GAME_4, 10),
        }
        
        try:
            self.trigger_embeddings = {}
            for trigger in self.clue_pairs.keys():
                self.trigger_embeddings[trigger] = self.encoder.encode(trigger)
        except Exception as e:
            logger.error(f"Failed to encode trigger phrases: {str(e)}")
            raise

    def is_similar_to_trigger(self, query: str, threshold: float = 0.85) -> Optional[str]:
        try:
            query_embedding = self.encoder.encode(query)
            
            for trigger, trigger_embedding in self.trigger_embeddings.items():
                similarity = cosine_similarity(
                    [query_embedding],
                    [trigger_embedding]
                )[0][0]
                
                if similarity > threshold:
                    logger.info(f"Trigger match found: '{query}' matches '{trigger}'")
                    return trigger
            
            return None
        except Exception as e:
            logger.error(f"Error in similarity check: {str(e)}")
            raise

    def has_completed_required_game(self, trigger: str, completed_games: List[int]) -> bool:
        clue_info = self.clue_pairs[trigger]
        return clue_info.required_game.value in completed_games

    async def generate_misleading_response(self, user_query: str, is_locked_trigger: bool = False) -> str:
        try:
            # Get the user's query from the last message
            #user_query = next((msg.content for msg in reversed(messages) if msg.role == "user"), "")
            
            if is_locked_trigger:
                system_prompt = """You are a suspicious character in a murder mystery. 
                The user has found a relevant question but hasn't earned the right to know the answer yet. 
                Generate a misleading response that hints they're on the right track but need to progress further by playing a game on the phone.
                Keep it concise (1-2 sentences) and natural sounding."""
            else:
                system_prompt = """You are a suspicious character in a murder mystery who wants to mislead the investigator. 
                When given a question:
                1. If the question asks for factual information, provide incorrect information confidently
                2. If asked about events, describe them differently from what actually happened
                3. Use sarcasm and misdirection
                4. Give irrelevant details that lead nowhere
                5. Contradict yourself subtly
                6. Keep responses concise (1-2 sentences)
                7. Sound natural and conversational
                
                Important: Never reveal that you're trying to be misleading."""

            chat = self.model.start_chat(history=[])
            response = await chat.send_message_async(
                f"{system_prompt}\n\nUser question: {user_query}",
                generation_config={
                    "temperature": 0.9,
                    "max_output_tokens": 150,
                }
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise

    async def get_response(self, messages: List[ChatMessage], completed_games: List[int]) -> str:
        try:
            # Get the user's query from the last message
            user_query = next((msg.content for msg in reversed(messages) if msg.role == "user"), "")
            matching_trigger = self.is_similar_to_trigger(user_query)
            
            if matching_trigger:
                logger.info(f"Found matching trigger: {matching_trigger}")
                
                if self.has_completed_required_game(matching_trigger, completed_games):
                    logger.info(f"User has completed required game, returning clue")
                    return self.clue_pairs[matching_trigger].answer
                else:
                    logger.info(f"User has not completed required game, generating locked response")
                    return await self.generate_misleading_response(user_query, is_locked_trigger=True)
            
            logger.info("Generating standard misleading response")
            return await self.generate_misleading_response(user_query)
            
        except Exception as e:
            logger.error(f"Error in get_response: {str(e)}")
            raise

try:
    bot = GeminiMysteryBot()
except Exception as e:
    logger.critical(f"Failed to initialize bot: {str(e)}")
    raise

async def verify_api_key(authorization: str = Depends(APIKeyHeader(name="Authorization"))):
    secret = authorization[len("Bearer "):]
    if secret != API_KEY:
        logger.warning(f"Invalid API key")
        raise HTTPException(status_code=401, detail="Invalid API key")
    return secret

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/v1/chat/completions", response_model=ChatCompletion)
async def chat_endpoint(
    request: ChatCompletionRequest,
    secret: str = Depends(verify_api_key)
):
    try:
        # Extract completed_games from the last system message
        completed_games = []
        for msg in request.messages:
            if msg.role == "system" and "completed_games" in msg.content:
                try:
                    # Parse completed_games from system message
                    completed_games = eval(msg.content.split("completed_games=")[1].split("]")[0] + "]")
                except:
                    pass
                break

        logger.info(f"Completed games: {completed_games}")
        
        response_text = await bot.get_response(
            request.messages,
            completed_games
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
                        content=response_text
                    )
                )
            ],
            usage={
                "prompt_tokens": 0,  # Placeholder values
                "completion_tokens": 0,
                "total_tokens": 0
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")