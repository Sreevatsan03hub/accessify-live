import logging
import re
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

# Singleton
_simplifier = None

class TextSimplifier:
    """
    Service for simplifying text to improve cognitive accessibility.
    Supports rule-based (fast) and LLM-based (advanced) simplification.
    """
    
    # Common filler words to remove
    FILLER_WORDS = [
        r"\bum\b", r"\buh\b", r"\ber\b", r"\bah\b", 
        r"\byou know\b", r"\blike\b", r"\bI mean\b", 
        r"\bkind of\b", r"\bsort of\b", r"\bbasically\b", 
        r"\bliterally\b", r"\bactually\b"
    ]
    
    # Simple word replacements
    # In production, this would be a larger loaded dictionary
    REPLACEMENTS = {
        # Academic / Formal -> Simple
        "utilize": "use",
        "demonstrate": "show",
        "subsequently": "later",
        "nevertheless": "however",
        "approximately": "about",
        "attempt": "try",
        "consequently": "so",
        "fundamental": "basic",
        "implement": "build",
        "objective": "goal",
        "kindly": "please",
        "inform": "tell",
        "assistance": "help",
        "commence": "start",
        "terminate": "end",
        "regarding": "about",
        "verify": "check",
        "functionality": "feature",
        "methodology": "method",
        "facilitate": "help",
        "component": "part"
    }

    def __init__(self, use_llm: bool = False):
        self.use_llm = use_llm
        self.llm_model = None
        self.llm_tokenizer = None
        
        if use_llm:
            self._load_llm()
            
    def _load_llm(self):
        """Load local LLM (e.g., Phi-3) if enabled."""
        try:
            # Placeholder for future LLM loading logic
            # from transformers import AutoModelForCausalLM, AutoTokenizer
            logger.info("LLM loading enabled (Symbolic placeholder)")
            pass
        except Exception as e:
            logger.error(f"Failed to load LLM: {e}")
            self.use_llm = False

    def remove_fillers(self, text: str) -> str:
        """Remove common filler words and hesitation markers."""
        if not text:
            return ""
            
        cleaned_text = text
        for pattern in self.FILLER_WORDS:
            cleaned_text = re.sub(pattern, "", cleaned_text, flags=re.IGNORECASE)
            
        # Clean up extra spaces
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
        return cleaned_text

    def simplify_words(self, text: str) -> str:
        """Replace complex words with simpler alternatives."""
        if not text:
            return ""
            
        simplified_text = text
        # Simple case-insensitive replacement
        # Note: This is naive and doesn't handle context/POS tagging yet
        for complex_word, simple_word in self.REPLACEMENTS.items():
            pattern = r"\b" + re.escape(complex_word) + r"\b"
            simplified_text = re.sub(pattern, simple_word, simplified_text, flags=re.IGNORECASE)
            
        return simplified_text
        
    def split_long_sentences(self, text: str) -> str:
        """Split very long sentences into shorter chunks."""
        # Simple heuristic: Split on 'and', 'but', 'because' if sentence is long (>15 words)
        words = text.split()
        if len(words) < 15:
            return text
            
        # This is a placeholder for more advanced splitting logic
        # For now, just return as is rather than risking bad splits
        return text

    def simplify(self, text: str) -> str:
        """
        Main simplification pipeline.
        
        Steps:
        1. Remove fillers (always)
        2. Simplify words (rule-based)
        3. LLM simplification (if enabled and text is complex)
        """
        if not text:
            return ""
            
        # Step 1: Cleaning
        text = self.remove_fillers(text)
        
        # Step 2: Rule-based simplification
        text = self.simplify_words(text)
        
        # Step 3: LLM (Future)
        if self.use_llm:
            # text = self._simplify_with_llm(text)
            pass
            
        return text

def get_simplifier() -> TextSimplifier:
    """Get singleton simplifier instance."""
    global _simplifier
    if _simplifier is None:
        _simplifier = TextSimplifier(use_llm=False)
    return _simplifier
