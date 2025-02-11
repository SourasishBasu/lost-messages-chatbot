from enum import Enum
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Optional, Tuple
from app.core.config import get_settings
from app.core.logging import setup_logger
from app.services.redis import RedisService
from redis.asyncio import ConnectionPool

logger = setup_logger()
settings = get_settings()

class GameState(Enum):
    GAME_1 = 1
    GAME_2 = 2
    GAME_3 = 3
    GAME_4 = 4
    GAME_5 = 5
    GAME_6 = 6

class ClueInfo:
    def __init__(self, answer: str, required_game: GameState, index: str):
        self.answer = answer
        self.required_game = required_game
        self.index = index

class GeminiMysteryBot:
    def __init__(self, redis_pool: ConnectionPool):
        try:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            self.redis_service = RedisService(redis_pool)
            logger.info("MysteryBot initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MysteryBot: {str(e)}")
            raise

        self.clue_pairs = {
            "was the victim given the correct dosage?": ClueInfo("Victim was prescribed more than normal dosage, toxicology showed inconsistent drug levels.", GameState.GAME_1, "1a"),
            "what were the medicines?": ClueInfo("Anti-depressants and anti-anxiety medication", GameState.GAME_1, "1b"),
            "what was the doctor's hidden motive?": ClueInfo("External pressure, possibly from the company or other interested parties but no solid evidence has been found.", GameState.GAME_1, "1c"),
            
            "who is the friend?": ClueInfo("Sam", GameState.GAME_2, "2a"),
            "what was the medicine dosage?": ClueInfo("low", GameState.GAME_2, "2b"),
            "what did the friend leak?": ClueInfo("photos", GameState.GAME_2, "2c"),

            "what was in the company email?": ClueInfo("Sensitive financial records and corruption reports. Victim probably considered reporting it.", GameState.GAME_3, "3a"),
            "who sent the company mail?": ClueInfo("Unknown identity, possibly masked through proxy servers.", GameState.GAME_3, "3b"),
            "who in the company would have been affected with leak?": ClueInfo("Several high-level executives under scrutiny for fraud, manager and recently promoted colleague as well.", GameState.GAME_3, "3c"),

            "Which family member threatened the victim?": ClueInfo("Uncle but its uncertain whether its a direct threat or a warning.", GameState.GAME_4, "4a"),
            "What was the threat from family member?": ClueInfo("Inheritance Dispute", GameState.GAME_4, "4b"),
            "How much was the victim going to inherit?": ClueInfo("1 Million USD", GameState.GAME_4, "4c"),

            "Which family member threatened the victim?": ClueInfo("Brother in Law", GameState.GAME_5, "5a"),
            "What was the threat from family member?": ClueInfo("Inheritance Dispute", GameState.GAME_5, "5b"),
            "How much was the victim going to inherit?": ClueInfo("1 Million USD", GameState.GAME_5, "5c"),

            "Which family member threatened the victim?": ClueInfo("Brother in Law", GameState.GAME_6, "6a"),
            "What was the threat from family member?": ClueInfo("Inheritance Dispute", GameState.GAME_6, "6b"),
            "How much was the victim going to inherit?": ClueInfo("1 Million USD", GameState.GAME_6, "6c"),
        }

        try:
            self.trigger_embeddings = {
                trigger: self.encoder.encode(trigger)
                for trigger in self.clue_pairs.keys()
            }
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

    async def generate_misleading_response(
        self,
        user_query: str,
        chat_history: List[Dict],
        is_locked_trigger: bool = False
    ) -> str:
        try:
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
                    
            history_text = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in chat_history
            ])
            
            prompt = f"{system_prompt}\n\nPrevious conversation:\n{history_text}\n\nUser question: {user_query}"

            chat = self.model.start_chat(history=[])
            response = await chat.send_message_async(
                prompt,
                generation_config={
                    "temperature": 0.9,
                    "max_output_tokens": 150,
                }
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise

    async def get_response(self, messages: List[Dict], completed_games: List[int], team_id: Optional[str]) -> Tuple[str, str]:
        try:
            user_query = next((msg["content"] for msg in reversed(messages) if msg["role"] == "user"), "")
            
            chat_history = []
            if team_id:
                chat_history = await self.redis_service.get_team_history(team_id)
            
            index = "none"
            matching_trigger = self.is_similar_to_trigger(user_query)
            
            if matching_trigger:
                logger.info(f"Found matching trigger: {matching_trigger}")
                
                if self.has_completed_required_game(matching_trigger, completed_games):
                    response = self.clue_pairs[matching_trigger].answer
                    index = self.clue_pairs[matching_trigger].index
                else:
                    response = await self.generate_misleading_response(
                        user_query,
                        chat_history,
                        is_locked_trigger=True
                    )
            else:
                response = await self.generate_misleading_response(
                    user_query,
                    chat_history
                )
            
            if team_id:
                await self.redis_service.save_to_history(team_id, {
                    "role": "user",
                    "content": user_query
                })
                await self.redis_service.save_to_history(team_id, {
                    "role": "assistant",
                    "content": response
                })
            
            return response, index
            
        except Exception as e:
            logger.error(f"Error in get_response: {str(e)}")
            raise