from collections.abc import Iterator


class LLM:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate_text(self, prompt: str) -> str:
        raise NotImplementedError("This method should be overridden by subclasses.")

    def stream_text(self, prompt: str) -> Iterator[str]:
        """Yield answer fragments; providers may override for true streaming."""
        yield self.generate_text(prompt)
