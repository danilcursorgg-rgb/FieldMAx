"""NARE-enhanced text generation using the PyTorch bridge.

Theory alignment:
  - Amortised reasoning: latent trajectory replay accelerates generation.
  - Cognitive temperature: switches between exploit (greedy) and explore
    (stochastic) decoding based on the field's internal energy / entropy.
  - Non-parametric retrieval + trace memory inject prior reasoning
    experience into each generation step.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .bridge import NAREBridge
from .model import NAREConfig, NAREFieldModel


@dataclass(frozen=True)
class GenerationResult:
    prompt: str
    vanilla_answer: str
    nare_answer: str
    nare_energies: list[float]
    nare_modes: list[str]
    nare_temperatures: list[float]


class NAREGenerator:
    """Token-by-token text generator enhanced by PyTorch NARE-Field.

    Flow per token::

        LLM hidden(t)  -->  proj_down  -->  NARE-Field  -->  proj_up
              |                                                 |
              +-------------- blend(alpha) --------------------+
              |
          lm_head  -->  logits  -->  next_token
    """

    def __init__(
        self,
        bridge: NAREBridge,
        field: NAREFieldModel,
        alpha: float = 0.3,
        energy_threshold: float = 5.0,
        energy_scale_range: float = 2.0,
    ):
        self.bridge = bridge
        self.field = field
        self.alpha = alpha
        self.energy_threshold = energy_threshold
        self.energy_scale_range = energy_scale_range

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 256) -> GenerationResult:
        bridge = self.bridge
        field = self.field
        field.eval()

        # Reset trace context for each generation
        field.trace_memory.keys = torch.empty(0, field.config.dim, device=bridge.device)
        field.trace_memory.deltas = torch.empty(0, field.config.dim, device=bridge.device)
        field.trace_memory._energies = []
        field.temperature_ctrl.reset()

        # ---- 1. Vanilla baseline ----
        vanilla = bridge.generate_vanilla(prompt, max_new_tokens=max_new_tokens)

        # ---- 2. NARE-enhanced generation ----
        inputs = bridge.tokenizer(prompt, return_tensors="pt").to(bridge.device)
        input_ids = inputs["input_ids"]
        generated_ids = input_ids[0].tolist()

        energies: list[float] = []
        modes: list[str] = []
        temperatures: list[float] = []

        # Get full prompt hidden states + KV cache
        all_hidden, past_kv = bridge.encode_hidden(input_ids, use_cache=True)
        all_hidden = all_hidden.squeeze(0)  # (seq_len, llm_dim)

        # Warmup: feed prompt tokens through the field (no learning)
        for i in range(all_hidden.shape[0]):
            h = all_hidden[i:i+1]  # (1, llm_dim)
            z = bridge.project_to_nare(h)  # (1, nare_dim)
            field(z, learn=False)

        last_hidden = all_hidden[-1:]  # (1, llm_dim)

        # Token-by-token generation with NARE correction
        for _ in range(max_new_tokens):
            # Project to NARE space
            z = bridge.project_to_nare(last_hidden)  # (1, nare_dim)

            # Forward through full NARE-Field pipeline
            result = field(z, learn=False)
            energy = result.loss.total.item()
            mode = result.temperature.mode
            temp = result.temperature.temperature

            energies.append(energy)
            modes.append(mode)
            temperatures.append(temp)

            # Blend LLM hidden state with NARE prediction (projected up)
            blended = bridge.blend(
                last_hidden,
                result.prediction,
                alpha=self.alpha,
                energy=energy,
                threshold=self.energy_threshold,
                scale_range=self.energy_scale_range,
            )

            # Get logits
            logits = bridge.hidden_to_logits(blended)
            next_id = int(logits[0, -1].argmax()) if logits.ndim == 3 else int(logits[-1].argmax())

            if next_id == bridge.tokenizer.eos_token_id:
                break

            generated_ids.append(next_id)

            # Single-token forward pass with KV cache
            next_input = torch.tensor([[next_id]], device=bridge.device)
            step_hidden, past_kv = bridge.encode_hidden(
                next_input, use_cache=True, past_key_values=past_kv,
            )
            last_hidden = step_hidden.squeeze(0)  # (1, llm_dim)

        prompt_len = input_ids.shape[1]
        nare_text = bridge.decode_ids(generated_ids[prompt_len:])

        return GenerationResult(
            prompt=prompt,
            vanilla_answer=vanilla,
            nare_answer=nare_text,
            nare_energies=energies,
            nare_modes=modes,
            nare_temperatures=temperatures,
        )
