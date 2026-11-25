from llm import base
from langchain_google_genai import GoogleGenerativeAI


class GeminiLLM(base.LLM):
    def __init__(
        self,
        model_name: str,
        api_key: str,
        request_timeout: float = 30,
        retries: int = 1,
    ):
        super().__init__(model_name)
        self.client = GoogleGenerativeAI(
            model=self.model_name,
            api_key=api_key,
            request_timeout=request_timeout,
            retries=retries,
        )

    def generate_text(self, prompt: str) -> str:
        response = self.client.invoke(prompt)
        return response

    def stream_text(self, prompt: str):
        stream = getattr(self.client, "stream", None)
        if stream is None:
            yield self.generate_text(prompt)
            return
        try:
            for chunk in stream(prompt):
                text = getattr(chunk, "content", chunk)
                if isinstance(text, str) and text:
                    yield text
        except (AttributeError, NotImplementedError):
            yield self.generate_text(prompt)
