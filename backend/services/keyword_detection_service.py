import re
import logging
from typing import List, Dict
import yake

logger = logging.getLogger(__name__)

class KeywordDetector:
    """
    AI-powered keyword detection using YAKE (Yet Another Keyword Extractor).
    Automatically identifies important words from ANY text — no hardcoding needed.
    An emoji map is used as an optional decoration layer for known concepts.
    """
    
    # Emoji decoration map — maps known concepts to emojis for visual flair.
    # YAKE does the detection; this map just adds the emoji if a match exists.
    EMOJI_MAP = {
        # Academic / Action
        "exam": "📘", "test": "📝", "quiz": "📝", "assignment": "📋",
        "homework": "🏠", "deadline": "⏰", "due date": "📅",
        "submission": "📤", "submit": "📤", "project": "🚀",
        "presentation": "🎤", "class": "🏫", "lecture": "🎓",
        
        # Urgency
        "important": "⭐", "crucial": "⭐", "urgent": "⚠️",
        "warning": "⚠️", "alert": "🚨", "remember": "🧠",
        "note": "📝",
        
        # Sentiment
        "good": "✅", "great": "🎉", "excellent": "🌟",
        "amazing": "🤩", "beautiful": "✨", "wonderful": "🌟",
        "wrong": "❌", "error": "❌", "fail": "❌",
        
        # General
        "love": "❤️", "heart": "💖", "music": "🎵", "song": "🎶",
        "world": "🌍", "people": "👥", "life": "🌱", "time": "⏳",
        "money": "💰", "work": "💼", "home": "🏠", "family": "👨‍👩‍👧‍👦",
        "friend": "🤝", "happy": "😊", "sad": "😢", "tired": "😴",
        "dream": "💭", "idea": "💡", "think": "🤔", "learn": "📚",
        "teach": "👨‍🏫", "help": "🆘", "problem": "⚙️", "solution": "💡",
        "goal": "🎯", "success": "🏆", "challenge": "💪", "change": "🔄",
        "future": "🔮", "technology": "💻", "science": "🔬",
        "nature": "🌿", "health": "🏥", "food": "🍽️", "water": "💧",
        "energy": "⚡", "power": "💪", "story": "📖", "game": "🎮",
        "play": "🎮", "travel": "✈️", "country": "🏳️", "city": "🏙️",
        "multitasking": "🤹", "creative": "🎨", "creativity": "🎨",
        "today": "📅", "tomorrow": "📆", "now": "⏰", "later": "🔜",
        
        # Multilingual
        "தேர்வு": "📘", "சோதனை": "📝", "காலக்கெடு": "⏰",
        "முக்கியம்": "⭐", "சமர்ப்பி": "📤", "பாடம்": "🏫",
        "परीक्षा": "📘", "जांच": "📝", "समय सीमा": "⏰",
        "महत्वपूर्ण": "⭐", "जमा": "📤",
        "పరీక్ష": "📘", "గడువు": "⏰", "ముఖ్యమైన": "⭐", "సమర్పించు": "📤",
    }

    def __init__(self):
        # YAKE extractor: language-agnostic, unsupervised
        # max_ngram_size=2: detect 1-2 word phrases
        # top=10: return top 10 keywords per text
        # deduplication_threshold=0.3: avoid near-duplicate keywords
        self.extractor = yake.KeywordExtractor(
            lan="en",
            n=2,              # max ngram size
            top=10,           # top N keywords
            dedupLim=0.3,     # dedup threshold
            windowsSize=2     # context window
        )
        logger.info("YAKE keyword extractor initialized (AI-powered, no hardcoding)")

    def _get_emoji(self, word: str) -> str:
        """Look up emoji for a word from the decoration map."""
        word_lower = word.lower().strip()
        # Direct match
        if word_lower in self.EMOJI_MAP:
            return self.EMOJI_MAP[word_lower]
        # Check each word in multi-word keyword against map
        for token in word_lower.split():
            if token in self.EMOJI_MAP:
                return self.EMOJI_MAP[token]
        return "🔑"  # Default emoji for any auto-detected keyword

    def extract_keywords(self, text: str) -> List[Dict[str, str]]:
        """
        Auto-detect important keywords using YAKE.
        Returns: [{"keyword": "...", "original": "...", "emoji": "...", "position": int, "score": float}]
        """
        if not text or len(text.strip()) < 3:
            return []
        
        try:
            # YAKE returns list of (keyword_string, score) — lower score = more important
            yake_results = self.extractor.extract_keywords(text)
        except Exception as e:
            logger.warning(f"YAKE extraction failed: {e}")
            return []
        
        found = []
        seen_positions = set()  # Avoid overlapping highlights
        
        for keyword_str, score in yake_results:
            # Find position of this keyword in the original text
            pattern = re.compile(re.escape(keyword_str), re.IGNORECASE)
            match = pattern.search(text)
            if match:
                pos = match.start()
                # Skip if overlaps with an already-found keyword
                span = set(range(pos, pos + len(match.group())))
                if span & seen_positions:
                    continue
                seen_positions.update(span)
                
                emoji = self._get_emoji(keyword_str)
                found.append({
                    "keyword": keyword_str.lower(),
                    "original": match.group(),
                    "emoji": emoji,
                    "position": pos,
                    "score": round(score, 4)
                })
        
        # Sort by position in text
        found.sort(key=lambda x: x["position"])
        return found

    def enrich_text(self, text: str, format: str = "html") -> str:
        """
        Insert emojis and formatting into text for detected keywords.
        
        Args:
            text: Input text
            format: "html" for web display (<span>), "vtt" for video captions (<c.kt>)
            
        Returns:
            Enriched text string
        """
        if not text:
            return ""
            
        matches = self.extract_keywords(text)
        if not matches:
            return text
            
        # Sort reverse so we can insert from back without affecting front indices
        matches.sort(key=lambda x: x["position"], reverse=True)
        
        enriched = text
        for m in matches:
            end_pos = m["position"] + len(m["original"])
            start_pos = m["position"]
            original_word = enriched[start_pos:end_pos]
            
            # Idempotency: skip if emoji already present
            check_window = enriched[end_pos:end_pos+20]
            if m['emoji'] in check_window:
                continue

            if format == "html":
                enriched_segment = f"<span class='key-term'>{original_word} {m['emoji']}</span>"
            elif format == "vtt":
                enriched_segment = f"<c.kt>{original_word} {m['emoji']}</c>"
            else:
                enriched_segment = f"{original_word} {m['emoji']}"
            
            enriched = enriched[:start_pos] + enriched_segment + enriched[end_pos:]
                
        return enriched

# Singleton instance
_detector = None

def get_keyword_detector() -> KeywordDetector:
    global _detector
    if _detector is None:
        _detector = KeywordDetector()
    return _detector
