"""PyTorch bridge between a pretrained causal LM and the NARE-Field model.

Theory alignment:
  K = C + R  (knowledge = compact core + reconstruction)

  The LLM provides raw hidden states (K).  The bridge projects them
  down to a compressed NARE space (C) via *proj_down*, processes them
  through the field, and projects back up via *proj_up* for token
  generation.  Both projections are trainable — the rest of the LLM
  stays frozen.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .model import NAREConfig, NAREFieldModel, NAREStepResult


@dataclass(frozen=True)
class BridgeConfig:
    """Configuration for the LLM <-> NARE bridge."""
    model_id: str = "Qwen/Qwen2-1.5B"
    nare_dim: int = 256
    device: str | None = None
    freeze_llm: bool = True
    dtype: str = "auto"


@dataclass(frozen=True)
class BridgeOutput:
    """Output of a single bridge encode call."""
    hidden_states: torch.Tensor   # (seq_len, llm_dim)
    nare_states: torch.Tensor     # (seq_len, nare_dim)
    token_ids: list[int]
    tokens: list[str]


class NAREBridge(nn.Module):
    """Trainable bridge between a frozen pretrained LLM and PyTorch NARE-Field.

    Architecture::

        text -> tokenizer -> frozen_LLM -> hidden(llm_dim)
                              |
                         proj_down (trainable)
                              |
                         NARE-Field  (trainable)
                              |
                         proj_up   (trainable)
                              |
                         blended hidden -> lm_head -> logits
    """

    def __init__(self, config: BridgeConfig | None = None, *, model_id: str | None = None):
        super().__init__()
        if config is None:
            config = BridgeConfig(model_id=model_id or "Qwen/Qwen2-1.5B")
        self.config = config

        # Lazy imports so the module is importable without transformers installed
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id)
        self._llm = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            torch_dtype=config.dtype,
            device_map=device,
        )
        if config.freeze_llm:
            for p in self._llm.parameters():
                p.requires_grad_(False)
            self._llm.eval()

        self._llm_dim: int = self._llm.config.hidden_size

        # Trainable projection layers
        self.proj_down = nn.Linear(self._llm_dim, config.nare_dim).to(device)
        self.proj_up = nn.Linear(config.nare_dim, self._llm_dim).to(device)

        # Layer-norm for stable projection
        self.norm_down = nn.LayerNorm(config.nare_dim).to(device)
        self.norm_up = nn.LayerNorm(self._llm_dim).to(device)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def llm_dim(self) -> int:
        return self._llm_dim

    @property
    def nare_dim(self) -> int:
        return self.config.nare_dim

    @property
    def device(self) -> str:
        return self._device

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def encode(self, text: str) -> BridgeOutput:
        """Encode text through the frozen LLM, then project to NARE space."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._llm.model(**inputs)
            hidden = outputs.last_hidden_state.squeeze(0).to(torch.float32)  # (seq, llm_dim)

        ids = inputs["input_ids"][0].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(ids)

        nare_states = self.norm_down(self.proj_down(hidden))  # (seq, nare_dim)
        return BridgeOutput(
            hidden_states=hidden,
            nare_states=nare_states,
            token_ids=ids,
            tokens=tokens,
        )

    def encode_hidden(self, input_ids: torch.Tensor, *, use_cache: bool = False, past_key_values=None):
        """Low-level: get LLM hidden states + optional KV-cache."""
        with torch.no_grad():
            outputs = self._llm.model(
                input_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
        hidden = outputs.last_hidden_state.to(torch.float32)
        return hidden, outputs.past_key_values if use_cache else None

    # ------------------------------------------------------------------
    # Projection helpers
    # ------------------------------------------------------------------
    def project_to_nare(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project LLM hidden states down to NARE space.

        Args:
            hidden: (..., llm_dim)
        Returns:
            (..., nare_dim)
        """
        return self.norm_down(self.proj_down(hidden))

    def project_to_llm(self, nare_out: torch.Tensor) -> torch.Tensor:
        """Project NARE output back up to LLM hidden space.

        Args:
            nare_out: (..., nare_dim)
        Returns:
            (..., llm_dim)
        """
        return self.norm_up(self.proj_up(nare_out))

    # ------------------------------------------------------------------
    # Token generation helpers
    # ------------------------------------------------------------------
    def hidden_to_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Apply the frozen lm_head to hidden states.

        Args:
            hidden: (..., llm_dim)
        Returns:
            (..., vocab_size)
        """
        t = hidden.to(self._llm.dtype).to(self._device)
        if t.ndim == 1:
            t = t.unsqueeze(0)
        with torch.no_grad():
            logits = self._llm.lm_head(t)
        return logits.to(torch.float32)

    def generate_vanilla(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Standard (un-enhanced) LLM generation."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self._device)
        with torch.no_grad():
            out_ids = self._llm.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new = out_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new, skip_special_tokens=True)

    def decode_ids(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Blending
    # ------------------------------------------------------------------
    def blend(
        self,
        llm_hidden: torch.Tensor,
        nare_prediction: torch.Tensor,
        alpha: float,
        *,
        energy: float = 0.0,
        threshold: float = 5.0,
        scale_range: float = 2.0,
    ) -> torch.Tensor:
        """Blend LLM hidden state with NARE-Field prediction.

        Uses dynamic routing: the blending weight scales with
        prediction energy — NARE intervenes more when the LLM
        is uncertain (high energy).

        Args:
            llm_hidden:      (*, llm_dim)
            nare_prediction: (*, nare_dim)
            alpha:           base blending weight
            energy:          current prediction energy
            threshold:       energy below which NARE stays passive
            scale_range:     energy range over which alpha ramps to full
        Returns:
            blended: (*, llm_dim)
        """
        nare_up = self.project_to_llm(nare_prediction)

        dynamic_scale = max(0.0, min(1.0, (energy - threshold) / max(scale_range, 1e-8)))
        effective_alpha = alpha * dynamic_scale

        blended = llm_hidden + effective_alpha * (nare_up - llm_hidden)

        # Preserve original norm to prevent logit drift
        orig_norm = torch.norm(llm_hidden, dim=-1, keepdim=True).clamp(min=1e-8)
        blend_norm = torch.norm(blended, dim=-1, keepdim=True).clamp(min=1e-8)
        blended = blended * (orig_norm / blend_norm)

        return blended
