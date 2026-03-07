import logging
from textblob import TextBlob
import re

logger = logging.getLogger(__name__)

class ToneAnalysisService:
    """
    Service to analyze the tone (emotion) and intent (communicative function) of text.
    Uses TextBlob for sentiment analysis and rule-based logic for intent classification.
    """

    def __init__(self):
        # Keywords for urgent intent
        self.urgent_keywords = ["urgent", "emergency", "immediately", "now", "critical", "warning", "alert", "danger"]
        
        # Mapping for tones to emojis and colors (for frontend usage)
        self.tone_map = {
            "happy": {"emoji": "😊", "color": "#2ecc71"}, # Green
            "angry": {"emoji": "😠", "color": "#e74c3c"}, # Red
            "neutral": {"emoji": "😐", "color": "#95a5a6"}, # Grey
            "question": {"emoji": "❓", "color": "#3498db"}, # Blue
            "urgent": {"emoji": "⚠️", "color": "#e67e22"}, # Orange
            "command": {"emoji": "❗", "color": "#8e44ad"}, # Purple (using intent color over emotion if strong)
        }

    def analyze_tone(self, text: str) -> dict:
        """
        Analyze the tone and intent of the given text.
        
        Args:
            text (str): The input text to analyze.
            
        Returns:
            dict: A dictionary containing:
                - emotion: "happy", "angry", "neutral"
                - intent: "question", "command", "statement", "urgent"
                - emoji: primary emoji to display
                - color: primary color code
                - confidence: float (sentiment polarity)
        """
        if not text:
            return {
                "emotion": "neutral", 
                "intent": "statement", 
                "emoji": "", 
                "color": "",
                "confidence": 0.0
            }
            
        # 1. Sentiment Analysis (Emotion) using TextBlob
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.5:
            emotion = "happy"
        elif polarity < -0.5:
            emotion = "angry"
        else:
            emotion = "neutral"
            
        # 2. Intent Classification (Rule-based)
        intent = "statement"
        text_lower = text.lower()
        
        # Check for Question
        if text.strip().endswith("?"):
            intent = "question"
            
        # Check for Urgent
        if any(keyword in text_lower for keyword in self.urgent_keywords):
            intent = "urgent"
            
        # Check for Command (Simple heuristic: Starts with verb? Hard without dependency parsing. 
        # For now, let's rely on exclamation mark or specific imperative starts if needed, 
        # but keep it simple as requested: Question/Urgent/Statement)
        if text.strip().endswith("!") and intent != "urgent": # Exclamation often implies command or strong emotion
             # If simplistic, command or just strong emotion? Let's say command for now if !
             intent = "command"

        # Determine Primary Display Tone (Intent usually overrides Emotion for UI logic)
        primary_tone = emotion
        if intent != "statement":
            primary_tone = intent
            
        # Get visual properties
        tone_props = self.tone_map.get(primary_tone, self.tone_map["neutral"])
        
        return {
            "emotion": emotion,
            "intent": intent,
            "emoji": tone_props["emoji"],
            "color": tone_props["color"],
            "primary_tone": primary_tone, # Helper for UI class
            "confidence": polarity
        }

# Singleton instance
_tone_service = None

def get_tone_service() -> ToneAnalysisService:
    global _tone_service
    if _tone_service is None:
        _tone_service = ToneAnalysisService()
    return _tone_service
