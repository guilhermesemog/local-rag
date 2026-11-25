from llm import base

DUMMY_ANSWER = "This is a dummy response."


class DummyLLM(base.LLM):
    def __init__(self, model_name: str = "dummy-model"):
        super().__init__(model_name)

    def generate_text(self, prompt: str = "") -> str:
        return DUMMY_ANSWER

    def stream_text(self, prompt: str = ""):
        text = self.generate_text(prompt)
        for start in range(0, len(text), 8):
            yield text[start : start + 8]
