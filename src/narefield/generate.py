from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bridge import NAREBridge
from .model import NAREConfig, NAREFieldModel


@dataclass(frozen=True)
class GenerationResult:
    prompt: str
    vanilla_answer: str
    nare_answer: str
    nare_energies: list[float]
    nare_modes: list[str]


class NAREGenerator:
    """Generates text through NARE-Field enhanced inference."""

    def __init__(self, bridge: NAREBridge, field: NAREFieldModel, alpha: float = 0.3):
        self.bridge = bridge
        self.field = field
        self.alpha = alpha

    def generate(self, prompt: str, max_new_tokens: int = 256) -> GenerationResult:
        import torch
        
        # Reset trace memory context for each new generation
        self.field.trace_memory.reset_context()

        # 1. Vanilla baseline
        vanilla = self.bridge.generate_vanilla(prompt, max_new_tokens=max_new_tokens)

        # 2. NARE-enhanced generation
        inputs = self.bridge.tokenizer(prompt, return_tensors="pt").to(self.bridge.device)
        input_ids = inputs["input_ids"]
        generated_ids = input_ids[0].tolist()
        
        energies: list[float] = []
        modes: list[str] = []

        # WARMUP: Teach NARE the prompt's hidden state geometry
        with torch.no_grad():
            outputs = self.bridge.model.model(input_ids, use_cache=True)
            past_key_values = outputs.past_key_values
            all_hidden = outputs.last_hidden_state[0]  # (seq_len, dim)
        
        # WARMUP: Feed the prompt to set up field context without overwriting trained weights
        previous_h = None
        for i in range(all_hidden.shape[0]):
            h = all_hidden[i].to(torch.float32).cpu().numpy().reshape(1, -1)
            self.field.step(h, previous_inputs=previous_h, learn=False)
            previous_h = h
        
        last_hidden = all_hidden[-1]
        previous_hidden_np = previous_h if previous_h is not None else last_hidden.to(torch.float32).cpu().numpy().reshape(1, -1)

        # GENERATION: Token-by-token with NARE correction
        for _ in range(max_new_tokens):
            last_hidden_np = last_hidden.to(torch.float32).cpu().numpy().reshape(1, -1)
            
            result = self.field.step(last_hidden_np, previous_inputs=previous_hidden_np, learn=False)
            nare_prediction = result.prediction[0]  # This is now predicted_delta
            energies.append(float(result.loss.total))
            modes.append(str(result.temperature.mode))

            # Residual correction: delta = how much NARE wants to change
            hidden_norm = np.linalg.norm(last_hidden_np[0])
            correction = nare_prediction
            
            # DYNAMIC ROUTING (Self-Correction)
            # Baseline trained energy is ~4.8. We only intervene if energy spikes above 5.0.
            energy = float(result.loss.total)
            threshold = 5.0
            # Scale from 0.0 (at energy <= 5.0) to 1.0 (at energy >= 7.0)
            dynamic_scale = max(0.0, min(1.0, (energy - threshold) / 2.0))
            effective_alpha = self.alpha * dynamic_scale
            
            blended = last_hidden_np[0] + effective_alpha * correction
            previous_hidden_np = last_hidden_np
            
            # Preserve exact norm to prevent logit drift
            blended_norm = np.linalg.norm(blended)
            if blended_norm > 1e-8:
                blended = blended * (hidden_norm / blended_norm)
            
            logits = self.bridge.hidden_to_logits(blended)
            
            next_id = int(np.argmax(logits[0, -1] if logits.ndim == 3 else logits[-1]))
            if next_id == self.bridge.tokenizer.eos_token_id:
                break
                
            generated_ids.append(next_id)
            
            # Single-token forward pass using KV-Cache
            with torch.no_grad():
                next_input = torch.tensor([[next_id]], device=self.bridge.device)
                outputs = self.bridge.model.model(
                    next_input, 
                    past_key_values=past_key_values, 
                    use_cache=True
                )
                past_key_values = outputs.past_key_values
                last_hidden = outputs.last_hidden_state[0, -1]

        prompt_len = input_ids.shape[1]
        nare_text = self.bridge.decode_ids(generated_ids[prompt_len:])

        return GenerationResult(
            prompt=prompt,
            vanilla_answer=vanilla,
            nare_answer=nare_text,
            nare_energies=energies,
            nare_modes=modes,
        )
