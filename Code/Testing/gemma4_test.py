import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client()

response = client.models.generate_content(
    model="gemma-4-26b-a4b-it",
    contents="What is the water formula?",
    
)

print(response.text)
