import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

class Generator:
    def __init__(self):
        self.model = "gemini-2.5-flash"

    def generate(self, question, context):
        prompt = f"""
            You are a medical information assistant.

            IMPORTANT:
            - Answer ONLY using the provided context.
            - If the answer is not in the context, say:
            "I could not find sufficient information in the provided documents."
            - Do not make up information.
            - Do not provide medical diagnosis.
            - Keep answers clear and concise.

            CONTEXT:
            {context}

            QUESTION:
            {question}
            """
        try:
            response = client.models.generate_content(model=self.model,contents=prompt)
        except Exception as e:
            return ("LLM unavailable. Showing retrieved medical evidences", context)
        return response.text
    