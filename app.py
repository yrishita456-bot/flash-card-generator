import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load .env
load_dotenv()

# Get Hugging Face token
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("ERROR: HF_TOKEN not found in .env")
    exit()

# Create Hugging Face Inference Client
client = InferenceClient(
    api_key=HF_TOKEN,
    provider="auto"
)


def generate_flashcards(topic):

    prompt = f"""
Create 5 educational flashcards about {topic}.

Use exactly this format:

Flashcard 1
Question: ...
Answer: ...

Flashcard 2
Question: ...
Answer: ...

Flashcard 3
Question: ...
Answer: ...

Flashcard 4
Question: ...
Answer: ...

Flashcard 5
Question: ...
Answer: ...

Keep the questions and answers short and easy for students.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=500
    )

    return response.choices[0].message.content


print("===== AI FLASHCARD GENERATOR =====")

topic = input("Enter your study topic: ")

print("\nGenerating flashcards...\n")

result = generate_flashcards(topic)

print("===== GENERATED FLASHCARDS =====")
print(result)