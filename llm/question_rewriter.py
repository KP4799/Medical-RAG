from llm.prompts import REWRITE_PROMPT
from llm.client import client, MODEL
import time

class Rewriter():
    def __init__(self):
        self.model = MODEL

    def rewrite(self,question,chat_history=None):
        if not chat_history:
            history = "No previous history."
        else:
            history = ''
            if chat_history:
                for message in chat_history[-6:]:
                    role = message['role'].capitalize()
                    history += f"{role}: {message['content']}\n"

        prompt = REWRITE_PROMPT.format(history=history,question=question)
        
        for attempt in range(2):
            try:
                response = client.models.generate_content(model=self.model,contents=prompt)
                rewritten = response.text.strip()
                if rewritten.lower() == question.strip().lower():
                    return ""    
                return rewritten
            
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: ",e)
                if attempt == 0:
                    time.sleep(2)
                    continue
                return ""