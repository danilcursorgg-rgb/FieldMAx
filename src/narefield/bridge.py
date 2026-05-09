from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class BridgeOutput:
    embeddings: np.ndarray
    token_ids: list[int]
    tokens: list[str]


class NAREBridge:
    """Minimal bridge between a pretrained causal LM and NARE-Field."""

    def __init__(self, model_id: str = "Qwen/Qwen2-1.5B", device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map=self.device,
        )
        self.model.eval()
        self._dim = self.model.config.hidden_size

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, text: str) -> BridgeOutput:
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            # Обращаемся к телу модели (model.model) для получения векторов
            outputs = self.model.model(**inputs)
            hidden = outputs.last_hidden_state
        embeddings = hidden.squeeze(0).to(torch.float32).cpu().numpy()
        ids = inputs["input_ids"][0].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        return BridgeOutput(embeddings=embeddings, token_ids=ids, tokens=tokens)

    def hidden_to_logits(self, hidden_state: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(hidden_state).to(self.device).to(self.model.dtype)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        with torch.no_grad():
            logits = self.model.lm_head(tensor)
        return logits.to(torch.float32).cpu().numpy()

    def generate_vanilla(self, prompt: str, max_new_tokens: int = 256) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def decode_ids(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
