import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_translation():
    print("--- Starting Translation Verification (Dravidian Model) ---")
    
    try:
        from services.translation_service import get_translator
        from transformers import MarianTokenizer, MarianMTModel
        
        # Manually load Dravidian model to test raw output
        model_name = "Helsinki-NLP/opus-mt-en-dra"
        print(f"Loading model: {model_name}...")
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        print("Model loaded.")
        
        text = "Hello, how are you? I am going to the shop."
        
        # Test Tamil
        print(f"\n--- Testing Tamil (>>tam<<) ---")
        input_text = f">>tam<< {text}"
        inputs = tokenizer(input_text, return_tensors="pt", padding=True)
        outputs = model.generate(**inputs)
        translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Original: {text}")
        print(f"Tamil Result: {translated}")
        
        # Test Telugu
        print(f"\n--- Testing Telugu (>>tel<<) ---")
        input_text = f">>tel<< {text}"
        inputs = tokenizer(input_text, return_tensors="pt", padding=True)
        outputs = model.generate(**inputs)
        translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Original: {text}")
        print(f"Telugu Result: {translated}")

        return True
            
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    verify_translation()
