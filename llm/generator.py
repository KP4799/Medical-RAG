from llm.client import client, MODEL
from llm.prompts import ANSWER_PROMPT
import time

class Generator:
    def __init__(self):
        self.model = MODEL

    def generate(self, question, retrieved_context):
        prompt = ANSWER_PROMPT.format(context=retrieved_context,question=question)

        for attempt in range(2):
            try:
                response = client.models.generate_content(model=self.model,contents=prompt)
                return response.text.strip()
            
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: ",e)
                if attempt == 0:
                    time.sleep(2)
                    continue
                return None
                
    