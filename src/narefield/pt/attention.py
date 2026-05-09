from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn

_EPS = 1e-12

@dataclass(frozen=True)
class SubQAttentionConfig:
    dim: int
    num_buckets: int = 8
    top_buckets: int = 2
    alpha: float = 1.0
    temperature: float = 1.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.num_buckets <= 0:
            raise ValueError("num_buckets must be positive")
        if not 1 <= self.top_buckets <= self.num_buckets:
            raise ValueError("top_buckets must be in [1, num_buckets]")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")

@dataclass(frozen=True)
class FieldAttentionResult:
    output: torch.Tensor
    token_mass: torch.Tensor
    token_bucket: torch.Tensor
    bucket_potential: torch.Tensor
    selected_buckets: torch.Tensor
    active_tokens: int
    complexity_estimate: float

class SubQFieldAttention(nn.Module):
    """Subquadratic field attention via bucketed potential retrieval (PyTorch)."""
    
    def __init__(self, config: SubQAttentionConfig):
        super().__init__()
        self.config = config
        
        if config.seed is not None:
            torch.manual_seed(config.seed)
            
        projection = torch.randn(config.dim)
        norm = torch.norm(projection)
        projection = projection / torch.max(norm, torch.tensor(_EPS))
        
        self.register_buffer("projection", projection)

    def forward(self, tokens: torch.Tensor) -> FieldAttentionResult:
        """
        Args:
            tokens: (seq_len, dim) or (batch, seq_len, dim)
            For simplicity in prototype, we assume (seq_len, dim) like the numpy version.
            If batched, we'd need more complex scatter operations. We'll stick to 2D.
        """
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
        if tokens.shape[1] != self.config.dim:
            raise ValueError(f"tokens must have shape (n, {self.config.dim})")
            
        token_count = tokens.shape[0]
        device = tokens.device
        
        masses = torch.norm(tokens, dim=1) + _EPS
        
        # 1D projection for fast bucketization
        coordinates = torch.matmul(tokens, self.projection)
        bucket_ids = self._bucketize(coordinates)
        
        centers_1d, vectors, bucket_mass = self._aggregate_buckets(
            tokens, coordinates, masses, bucket_ids
        )
        
        potential = self._bucket_potential(centers_1d, bucket_mass)
        selected_buckets = self._select_buckets(potential)
        
        output = self._attend_to_buckets(tokens, vectors, bucket_mass, selected_buckets)
        
        active_mask = torch.isin(bucket_ids, selected_buckets)
        active_tokens = int(active_mask.sum().item())
        complexity = float(token_count * selected_buckets.size(0))
        
        return FieldAttentionResult(
            output=output,
            token_mass=masses,
            token_bucket=bucket_ids,
            bucket_potential=potential,
            selected_buckets=selected_buckets,
            active_tokens=active_tokens,
            complexity_estimate=complexity,
        )

    def _bucketize(self, coordinates: torch.Tensor) -> torch.Tensor:
        minimum = torch.min(coordinates)
        maximum = torch.max(coordinates)
        if maximum - minimum <= _EPS:
            return torch.zeros_like(coordinates, dtype=torch.long)
            
        scaled = (coordinates - minimum) / (maximum - minimum + _EPS)
        buckets = (scaled * self.config.num_buckets).to(torch.long)
        return torch.clamp(buckets, max=self.config.num_buckets - 1)

    def _aggregate_buckets(
        self, values: torch.Tensor, coordinates: torch.Tensor, masses: torch.Tensor, bucket_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = values.device
        
        centers_1d = torch.zeros(self.config.num_buckets, dtype=values.dtype, device=device)
        vectors = torch.zeros((self.config.num_buckets, self.config.dim), dtype=values.dtype, device=device)
        bucket_mass = torch.zeros(self.config.num_buckets, dtype=values.dtype, device=device)
        
        # Using scatter_add for O(N) aggregation
        bucket_mass.scatter_add_(0, bucket_ids, masses)
        
        # We need sum of (coordinates * masses) to find center of mass in 1D?
        # The numpy code used simple mean of coordinates, not weighted by mass.
        # Numpy: float(np.mean(coordinates[mask]))
        # We can compute count per bucket to get mean
        counts = torch.zeros(self.config.num_buckets, dtype=values.dtype, device=device)
        counts.scatter_add_(0, bucket_ids, torch.ones_like(coordinates))
        
        coord_sums = torch.zeros_like(centers_1d)
        coord_sums.scatter_add_(0, bucket_ids, coordinates)
        
        active_mask = counts > 0
        centers_1d = torch.where(active_mask, coord_sums / counts, centers_1d)
        
        # Vectors: weighted average
        # Numpy: np.average(values[mask], axis=0, weights=masses[mask])
        # sum(values * mass) / sum(mass)
        weighted_values = values * masses.unsqueeze(1)
        vectors_sums = torch.zeros_like(vectors)
        bucket_ids_expanded = bucket_ids.unsqueeze(1).expand(-1, self.config.dim)
        vectors_sums.scatter_add_(0, bucket_ids_expanded, weighted_values)
        
        mass_expanded = bucket_mass.unsqueeze(1)
        active_mass_mask = mass_expanded > 0
        vectors = torch.where(active_mass_mask, vectors_sums / mass_expanded, vectors)
        
        return centers_1d, vectors, bucket_mass

    def _bucket_potential(self, centers_1d: torch.Tensor, bucket_mass: torch.Tensor) -> torch.Tensor:
        potential = torch.zeros(self.config.num_buckets, dtype=centers_1d.dtype, device=centers_1d.device)
        active = torch.nonzero(bucket_mass > 0).squeeze(-1)
        
        if active.size(0) == 0:
            return potential
            
        for bucket in active:
            distance = torch.abs(centers_1d[bucket] - centers_1d[active]) + self.config.temperature
            potential[bucket] = torch.sum(bucket_mass[active] / torch.pow(distance, self.config.alpha))
            
        return potential

    def _select_buckets(self, potential: torch.Tensor) -> torch.Tensor:
        active = torch.nonzero(potential > 0).squeeze(-1)
        if active.size(0) == 0:
            return torch.tensor([0], dtype=torch.long, device=potential.device)
            
        count = min(self.config.top_buckets, active.size(0))
        _, order = torch.topk(potential[active], count)
        return torch.sort(active[order]).values

    def _attend_to_buckets(
        self, values: torch.Tensor, bucket_vectors: torch.Tensor, bucket_masses: torch.Tensor, selected_buckets: torch.Tensor
    ) -> torch.Tensor:
        token_count = values.shape[0]
        
        sel_vectors = bucket_vectors[selected_buckets]  # (K, dim)
        sel_masses = bucket_masses[selected_buckets]    # (K,)
        
        # Distance from token i to all selected bucket centers
        # values: (N, dim) -> (N, 1, dim)
        # sel_vectors: (K, dim) -> (1, K, dim)
        diff = sel_vectors.unsqueeze(0) - values.unsqueeze(1) # (N, K, dim)
        distances = torch.norm(diff, dim=2) + self.config.temperature # (N, K)
        
        weights = sel_masses.unsqueeze(0) / torch.pow(distances, self.config.alpha) # (N, K)
        weights = weights / torch.clamp(weights.sum(dim=1, keepdim=True), min=_EPS)
        
        # output = weights @ sel_vectors
        # (N, K) @ (K, dim) -> (N, dim)
        output = torch.matmul(weights, sel_vectors)
        
        return output
