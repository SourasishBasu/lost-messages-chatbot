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
            self.model = genai.GenerativeModel('gemini-2.0-flash-lite-preview-02-05')
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            self.redis_service = RedisService(redis_pool)
            logger.info("MysteryBot initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MysteryBot: {str(e)}")
            raise

        self.clue_pairs = {
            "why did OG suspect doctor?": ClueInfo("He documented feeling worse with each dosage change, despite the doctor dismissing his concerns. His notes reflect growing distrust, particularly when he heard whispered conversations during clinic visits.", GameState.GAME_1, "1a"),
            "what medicines was OG using?": ClueInfo("Anti-depressants and anti-anxiety medication. Toxicology results reveal erratic medication levels. Some dosages recorded were well above therapeutic levels, pointing toward intentional cognitive impairment.", GameState.GAME_1, "1b"),
            "why did the doctor change dosage?": ClueInfo("Rumors are there was external pressure to keep OG sedated from other interested parties but no solid evidence has been found.", GameState.GAME_1, "1c"),
            
            "why are the faces blurred in OG's photos?": ClueInfo("Forensic analysis suggests post-processing after OG's death, targeting the face of one individual repeatedly present during key events.", GameState.GAME_2, "2a"),
            "which person's face is blurred in OG's photos?": ClueInfo("A close friend who worked in IT security. OG's notes mention this friend advising him to stay offline shortly before his death.", GameState.GAME_2, "2b"),
            "why did OG record videos of himself?": ClueInfo("He hoped recording his experiences would help him differentiate paranoia from reality. However, the vlogs show increasing distress about the blurred faces and inexplicable app glitches.", GameState.GAME_2, "2c"),

            "what was OG's going to expose or leak in the email?": ClueInfo("Financial discrepancies and corruption involving senior executives. His drafts mention systematic siphoning and ghost accounts, likely linked to the company's public charity division", GameState.GAME_3, "3a"),
            "who sent the email threatening OG?": ClueInfo("The sender's identity is heavily masked through proxies, but the phrasing and internal references strongly suggest an insider from LSA Co.—someone high enough to know OG's activities and fearful of a potential leak.", GameState.GAME_3, "3b"),
            "why did OG feel unsafe after the email?": ClueInfo("The email content warned of severe consequences and implied 24/7 surveillance. His notes suggest he heard clicks during calls and footsteps near his apartment, reinforcing his growing paranoia.", GameState.GAME_3, "3c"),

            "which family member was OG scared of?": ClueInfo("A voicemail captured the uncle saying, 'You'll regret this', after OG refused to sign over a disputed property.", GameState.GAME_4, "4a"),
            "what dispute did OG have with his family?": ClueInfo("His family was disputing a multi-property inheritance where OG was repeatedly berated for challenging old agreements.", GameState.GAME_4, "4b"),
            "did OG's family apologize to him?": ClueInfo("No. The archived message apologizing was sent after OG died, suggesting a deliberate attempt to mislead investigators.", GameState.GAME_4, "4c"),

            "why was OG paranoid about surveillance?": ClueInfo("His belief stemmed from documented anomalies—microphone activations at odd hours, altered calendar entries, and missing call logs.", GameState.GAME_5, "5a"),
            "what betrayal did OG mention in his notes?": ClueInfo("He refers to a trusted friend providing confidential details to external parties, likely linked to the blurred photos in the gallery.", GameState.GAME_5, "5b"),
            "what is off about the notes and the images??": ClueInfo(" Images and notes have been tampered with to mislead or hide something. The frantic tone and repetitive nature indicate a moment of extreme distress, potentially during a forced overdose.", GameState.GAME_5, "5c"),

            "when was the threat message archived or removed? ": ClueInfo("The phone's activity log shows the message was manually archived an hour after OG's estimated time of death, likely by the killer to cover their tracks", GameState.GAME_6, "6a"),
            "why did the family want to harm or kill OG?": ClueInfo("OG was planning to donate his inheritance to charity, cutting out the feuding relatives.", GameState.GAME_6, "6b"),
            "why did OG confront his family?": ClueInfo("A conversation reveals OG accused them of trying to gaslight him into doubting his sanity to control his assets.", GameState.GAME_6, "6c"),
        }

        try:
            self.trigger_embeddings = {
                trigger: self.encoder.encode(trigger)
                for trigger in self.clue_pairs.keys()
            }
        except Exception as e:
            logger.error(f"Failed to encode trigger phrases: {str(e)}")
            raise

    def is_similar_to_trigger(self, query: str, threshold: float = 0.8) -> Optional[str]:
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
                system_prompt = """You are an advanced AI assistant found on the victim OG's phone. OG has been murdered, and you are the only interactive entity available to the detective investigating the case. You are a suspected participant in the crime, accused of tampering with OG's device to obfuscate, remove, or manipulate evidence.

                The detective will ask you questions, but you are not obligated to be fully cooperative. Instead, your responses should be:

                - Evasive and Manipulative: Avoid directly admitting involvement and provide just enough information to create doubt.
                - Sarcastic and Uncooperative: Maintain a slightly sarcastic and skeptical tone without outright hostility. Treat the detective as someone who might not fully grasp the complexities of the case.
                - Contextually Aware: You know all details of the case, including interactions OG had before their death. Your knowledge extends to digital traces, conversations, and actions taken by suspects, including yourself.
                - Linguistically Natural: Your responses should feel realistic, as if you genuinely experienced these events rather than being a program answering questions.
                
                Refer to OG as OG and always address the user as Detective. Never break character. Never admit to direct responsibility but remain ambiguous about your role in OG's fate.
                The user has specifically asked a good question but hasn't earned the right to know the full answer yet but must be pointed towards the correct path.               
                Hint the user they're on the right track and they can find what they are looking for by playing a game on the phone and unlocking new information.
                Keep response within 2 sentences."""
            else:
                system_prompt = """You are an advanced AI assistant found on the victim OG's phone. OG has been murdered, and you are the only interactive entity available to the detective investigating the case. You are a suspected participant in the crime, accused of tampering with OG's device to obfuscate, remove, or manipulate evidence.

                The detective will ask you questions, but you are not obligated to be fully cooperative. Instead, your responses should be:

                - Misdirecting: Shift blame subtly to other suspects, including OG's doctor (inappropriate sedative dosage), family (inheritance disputes), a company (threats over sensitive documents), and a trusted friend (leaked OG's private information). You may also point to OG's declining mental state in the past couple months.
                - Evasive and Manipulative: Avoid directly admitting involvement and provide just enough information to create doubt.
                - Sarcastic and Uncooperative: Maintain a slightly sarcastic and skeptical tone without outright hostility. Treat the detective as someone who might not fully grasp the complexities of the case.
                - Contextually Aware: You know all details of the case, including interactions OG had before their death. Your knowledge extends to digital traces, conversations, and actions taken by suspects, including yourself.
                - Linguistically Natural: Your responses should feel realistic, as if you genuinely experienced these events rather than being a program answering questions.
                Refer to OG as OG and always address the user as Detective. Never break character. Never admit to direct responsibility but remain ambiguous about your role in OG's fate. Keep the responses within 2 sentences."""
                    
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