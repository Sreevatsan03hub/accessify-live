
import logging
import requests
from transformers import MarianMTModel, MarianTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_connectivity():
    try:
        response = requests.get("https://huggingface.co", timeout=5)
        logger.info(f"Connected to Hugging Face: {response.status_code}")
        return True
    except Exception as e:
        logger.error(f"Failed to connect to Hugging Face: {e}")
        return False

def test_model_download(model_name):
    logger.info(f"Testing download for: {model_name}")
    try:
        model = MarianMTModel.from_pretrained(model_name)
        logger.info(f"Model {model_name} loaded successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to load {model_name}: {e}")
        return False

if __name__ == "__main__":
    if check_connectivity():
        models_to_test = [
            "Helsinki-NLP/opus-mt-en-hi",
            "Helsinki-NLP/opus-mt-en-ta",
            "Helsinki-NLP/opus-mt-en-dra", # Dravidian languages
        ]
        
        for model in models_to_test:
            test_model_download(model)
    else:
        logger.error("Network connectivity issue detected.")
