from openai import OpenAI
from dotenv import load_dotenv
import os

# Load .env from current folder
load_dotenv()

# Debug check
print("Loaded key:", os.getenv("OPENROUTER_API_KEY"))

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

def ask_ai(prompt):
    response = client.chat.completions.create(
        model="mistralai/mistral-7b-instruct",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    user_input = input("Ask AI: ")
    print(ask_ai(user_input))
