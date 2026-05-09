from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass(frozen=True)
class MemoryFieldConfig:
    dim: int
    capacity: int = 128
    learning_rate: float = 0.05
    inference_rate: float = 0.25
    inference_steps: int = 8
    decay: float = 0.995
    temperature: float = 1.0
    normalize: bool = True

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if self.inference_steps <= 0:
            raise ValueError("inference_steps must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")

@dataclass(frozen=True)
class MemoryTrace:
    initial_state: torch.Tensor
    state: torch.Tensor
    prediction: torch.Tensor
    error: torch.Tensor
    delta: torch.Tensor
    energy_history: tuple[float, ...]

    @property
    def energy(self) -> float:
        return self.energy_history[-1] if self.energy_history else 0.0

class MemoryField(nn.Module):
    """
    Attractor memory with predictive inference and Hebbian writes (PyTorch).
    Implements the NARE-Field memory field operations natively on GPUs.
    """
    
    def __init__(self, config: MemoryFieldConfig):
        super().__init__()
        self.config = config
        
        # Parametric memory (W): learnable but typically updated via Hebbian rules
        # We keep it as Parameter so it's registered in the model.
        # Shape: (dim, dim). In F.linear(x, W), output is x @ W^T.
        # We want linear_prediction = x @ W^T to match numpy W @ x for single vector if W is (dim, dim).
        self.weights = nn.Parameter(torch.zeros(config.dim, config.dim))
        
        # Non-parametric / episodic attractors
        self.register_buffer("attractors", torch.empty(0, config.dim))
        
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self.config.normalize:
            return x
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        return torch.where(norm <= 1e-12, x, x / norm)

    def read(self, query: torch.Tensor) -> torch.Tensor:
        """
        Retrieve prediction from the memory field.
        Args:
            query: Tensor of shape (..., dim)
        Returns:
            prediction: Tensor of shape (..., dim)
        """
        query_norm = self._normalize(query)
        
        # linear_prediction: (..., dim)
        linear_prediction = F.linear(query_norm, self.weights)
        
        if self.attractors.size(0) == 0:
            return linear_prediction
            
        # similarities: (..., num_attractors)
        # For batched query (batch, dim), attractors is (num_attractors, dim)
        similarities = torch.matmul(query_norm, self.attractors.T) / (
            torch.sqrt(torch.tensor(self.config.dim, dtype=torch.float32, device=query.device)) * self.config.temperature
        )
        coeffs = F.softmax(similarities, dim=-1)
        
        # attractor_prediction: (..., dim)
        attractor_prediction = torch.matmul(coeffs, self.attractors)
        
        return 0.5 * linear_prediction + 0.5 * attractor_prediction

    def infer(self, state: torch.Tensor) -> MemoryTrace:
        """
        Perform predictive coding inference to find the nearest attractor / minimum energy state.
        Args:
            state: Tensor of shape (batch, dim)
        Returns:
            MemoryTrace containing the optimized state and deltas.
        """
        initial = self._normalize(state)
        history = []
        
        with torch.no_grad():
            current = initial.clone()
            
            for _ in range(self.config.inference_steps):
                prediction = self.read(current)
                error = current - prediction
                mse = torch.mean(torch.square(error)).item()
                history.append(mse)
                
                # Gradient descent step minimizing prediction error
                current = current - self.config.inference_rate * error
                current = self._normalize(current)
                
            prediction = self.read(current)
            error = current - prediction
            mse = torch.mean(torch.square(error)).item()
            history.append(mse)
            
            delta = current - initial
            
            return MemoryTrace(
                initial_state=initial,
                state=current,
                prediction=prediction,
                error=error,
                delta=delta,
                energy_history=tuple(history)
            )

    def forward(self, state: torch.Tensor, learn: bool = False) -> MemoryTrace:
        """
        Forward pass of the memory field.
        Args:
            state: Input tensor of shape (batch, dim)
            learn: If True, updates weights and attractors (Hebbian plasticity).
        """
        trace = self.infer(state)
        if learn:
            self.learn(trace)
        return trace

    def learn(self, trace: MemoryTrace) -> None:
        """
        Applies local Hebbian update based on prediction error.
        W <- decay * W + lr * error * z^T
        """
        with torch.no_grad():
            z = self._normalize(trace.delta)       # Presynaptic
            error = self._normalize(trace.error)   # Postsynaptic error
            
            # Since trace is batched, we compute the average outer product
            # outer_prod: (batch, dim, dim)
            outer_prod = torch.bmm(error.unsqueeze(2), z.unsqueeze(1))
            mean_update = outer_prod.mean(dim=0)
            
            self.weights.data = (
                self.config.decay * self.weights.data + 
                self.config.learning_rate * mean_update
            )
            
            self._append_attractors(trace.delta)
            
    def _append_attractors(self, states: torch.Tensor) -> None:
        """Add new states to the non-parametric attractor memory."""
        new_attractors = self._normalize(states)
        self.attractors = torch.cat([self.attractors, new_attractors], dim=0)
        
        if self.attractors.size(0) > self.config.capacity:
            self.attractors = self.attractors[-self.config.capacity:]

    def stats(self) -> dict[str, float]:
        return {
            "capacity": float(self.config.capacity),
            "stored_attractors": float(self.attractors.size(0)),
            "capacity_usage": float(self.attractors.size(0) / self.config.capacity),
            "weight_norm": float(torch.norm(self.weights).item()),
        }
