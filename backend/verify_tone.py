from services.tone_analysis_service import get_tone_service
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_tone_analysis():
    tone_service = get_tone_service()
    
    test_cases = [
        "I am so happy with this result!",
        "This is absolutely terrible and I hate it.",
        "The meeting is at 5 PM.",
        "Can you help me with this?",
        "This is urgent, please fix it immediately!",
        "Submit the report now!",
        "What is the deadline?"
    ]
    
    print(f"{'Text':<50} | {'Emotion':<10} | {'Intent':<10} | {'Emoji':<5} | {'Color'}")
    print("-" * 100)
    
    for text in test_cases:
        result = tone_service.analyze_tone(text)
        print(f"{text:<50} | {result['emotion']:<10} | {result['intent']:<10} | {result['emoji']:<5} | {result['color']}")

if __name__ == "__main__":
    test_tone_analysis()
