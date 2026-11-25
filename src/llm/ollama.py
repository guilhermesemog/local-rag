from langchain_ollama import OllamaLLM as LangChainOllamaLLM

from llm import base

# name conflicts with class name


class OllamaLLM(base.LLM):
    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        request_timeout: float = 120,
    ):
        super().__init__(model_name)
        self.client = LangChainOllamaLLM(
            model=self.model_name,
            base_url=base_url,
            temperature=temperature,
            client_kwargs={"timeout": request_timeout},
        )

    def generate_text(self, prompt: str) -> str:
        response = self.client.invoke(prompt)
        return response
