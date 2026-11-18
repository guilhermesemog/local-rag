from llm import base
import os
from langchain_google_genai import GoogleGenerativeAI


class GeminiLLM(base.LLM):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        request_timeout = float(os.getenv("GEMINI_REQUEST_TIMEOUT", "30"))
        retries = int(os.getenv("GEMINI_RETRIES", "1"))
        self.client = GoogleGenerativeAI(
            model=self.model_name,
            request_timeout=request_timeout,
            retries=retries,
        )

    def generate_text(self, prompt: str) -> str:
        response = self.client.invoke(prompt)
        return response