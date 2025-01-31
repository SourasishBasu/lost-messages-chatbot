from fastapi import FastAPI, HTTPException, Depends, Security
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from fastapi.security import APIKeyHeader
from dotenv import load_dotenv
from pydantic import BaseModel
from enum import Enum, auto
import logging
import groq
import os
import sys

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

key = APIKeyHeader(name="X-API-Key")
API_KEY = os.getenv("API_KEY")

class GameState(Enum):
    GAME_1 = 1
    GAME_2 = 2


class ChatInput(BaseModel):
    message: str
    completed_games: List[int]  # List of completed game IDs

class ChatResponse(BaseModel):
    response: str

class ClueInfo:
    def __init__(self, answer: str, required_game: GameState):
        self.answer = answer
        self.required_game = required_game

class GroqMysteryBot:
    def __init__(self, api_key: str):
        try:
            self.client = groq.Client(api_key=api_key)
            self.model = "llama-3.1-8b-instant"
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("MysteryBot initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MysteryBot: {str(e)}")
            raise

        # Dictionary to store clue triggers and their responses with required game states (examples)
        self.clue_pairs = {
            "where is the knife?": ClueInfo("kitchen", GameState.GAME_1),
            "who had the gun?": ClueInfo("john", GameState.GAME_2),
        }
        
        try:
            self.trigger_embeddings = {}
            for trigger in self.clue_pairs.keys():
                self.trigger_embeddings[trigger] = self.encoder.encode(trigger)
        except Exception as e:
            logger.error(f"Failed to encode trigger phrases: {str(e)}")
            raise

    def is_similar_to_trigger(self, query: str, threshold: float = 0.85) -> Optional[str]:
        """Check if query is semantically similar to any trigger phrase"""
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
        """Check if user has completed the required game for this trigger"""
        clue_info = self.clue_pairs[trigger]
        return clue_info.required_game.value in completed_games

    async def generate_misleading_response(self, query: str, is_locked_trigger: bool = False) -> str:
        """Generate a dynamically misleading response using Groq LLM"""
        try:
            # Switch system prompt based on whether this is a locked trigger
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
                7. Sound natural and conversational, not like you're following a template
                
                Important: Never reveal that you're trying to be misleading."""

            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.9,
                max_tokens=150
            )
            
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise

    async def get_response(self, query: str, completed_games: List[int]) -> tuple[str, bool]:
        """
        Returns tuple of (response_text, was_trigger_locked)
        was_trigger_locked indicates if a trigger was matched but access was denied
        """
        try:
            # Check for semantic similarity to trigger phrases
            matching_trigger = self.is_similar_to_trigger(query)
            
            if matching_trigger:
                logger.info(f"Found matching trigger: {matching_trigger}")
                
                # Check if user has completed required game
                if self.has_completed_required_game(matching_trigger, completed_games):
                    logger.info(f"User has completed required game, returning clue")
                    return self.clue_pairs[matching_trigger].answer
                else:
                    logger.info(f"User has not completed required game, generating locked response")
                    response = await self.generate_misleading_response(query, is_locked_trigger=True)
                    return response
            
            # Generate standard misleading response for non-trigger queries
            logger.info("Generating standard misleading response")
            response = await self.generate_misleading_response(query)
            return response
            
        except Exception as e:
            logger.error(f"Error in get_response: {str(e)}")
            raise

try:
    bot = GroqMysteryBot(api_key=os.environ.get("GROQ_API_KEY"))
except Exception as e:
    logger.critical(f"Failed to initialize bot: {str(e)}")
    raise

# Security

async def verify_api_key(secret: str = Security(key)):
    if secret != API_KEY:
        logger.warning(f"Invalid API key")
        raise HTTPException(status_code=401, detail="Invalid API key")
    return secret

# Routes

@app.get("/health", tags=["health"], summary="Check API health status")
async def health_check():
    return {"status": "healthy"}

@app.post("/chat", tags=["chat"], summary="Send user messages and game state to API")
async def chat_endpoint(
    chat_input: ChatInput,
    secret: str = Depends(verify_api_key)
):
    try:
        logger.info(f"Received chat request: {chat_input.message[:50]}...")
        logger.info(f"Completed games: {chat_input.completed_games}")
        
        response_text = await bot.get_response(
            chat_input.message,
            chat_input.completed_games
        )
        
        return ChatResponse(
            response=response_text,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
