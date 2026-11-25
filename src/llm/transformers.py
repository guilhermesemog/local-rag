from threading import Thread

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from llm.base import LLM


class TransformersLLM(LLM):
    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
    ):
        super().__init__(model_name)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )

        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loaded LLM '{model_name}' on device '{device}'")

    def _generation_kwargs(self, inputs, streamer=None):
        kwargs = {
            **inputs,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "temperature": self.temperature,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if streamer is not None:
            kwargs["streamer"] = streamer
        return kwargs

    @torch.no_grad()
    def generate_text(self, prompt: str) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
        ).to(self.model.device)

        output = self.model.generate(**self._generation_kwargs(inputs))

        generated = output[0][inputs["input_ids"].shape[-1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)

        return text.strip()

    def stream_text(self, prompt: str):
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
        ).to(self.model.device)
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        errors = []
        generation_kwargs = self._generation_kwargs(inputs, streamer=streamer)

        def run_generation():
            try:
                with torch.inference_mode():
                    self.model.generate(**generation_kwargs)
            except Exception as exc:  # noqa: BLE001 - re-raised in the caller
                errors.append(exc)
            finally:
                streamer.end()

        thread = Thread(
            target=run_generation,
            name="transformers-stream",
            daemon=True,
        )
        thread.start()
        try:
            for text in streamer:
                if text:
                    yield text
        finally:
            thread.join(timeout=5)
        if errors:
            raise errors[0]
