from google import genai
import os

client = genai.Client(api_key=os)

interaction = client.interactions.create(
    model="gemini-3.5-flash", input="Hello gemini. I am 'World'"
)
print(interaction.output_text)
