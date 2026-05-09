from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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
    output: np.ndarray
    token_mass: np.ndarray
    token_bucket: np.ndarray
    bucket_potential: np.ndarray
    selected_buckets: np.ndarray
    active_tokens: int
    complexity_estimate: float


@dataclass
class SubQFieldAttention:
    """Subquadratic field attention via bucketed potential retrieval."""

    config: SubQAttentionConfig

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.config.seed)
        projection = rng.normal(size=self.config.dim)
        norm = np.linalg.norm(projection)
        self.projection = projection / max(norm, _EPS)

    def __call__(self, tokens: np.ndarray) -> FieldAttentionResult:
        values = _as_tokens(tokens, self.config.dim)
        token_count = values.shape[0]
        masses = np.linalg.norm(values, axis=1) + _EPS
        
        # 1D projection for fast bucketization
        coordinates = values @ self.projection
        bucket_ids = self._bucketize(coordinates)
        
        # Aggregate buckets in O(N)
        bucket_centers_1d, bucket_vectors, bucket_masses = self._aggregate_buckets(
            values, coordinates, masses, bucket_ids
        )
        
        # Calculate potential for each bucket
        potential = self._bucket_potential(bucket_centers_1d, bucket_masses)
        selected_buckets = self._select_buckets(potential)
        
        # Attend to the top-k buckets in O(N * K)
        output = self._attend_to_buckets(values, bucket_vectors, bucket_masses, selected_buckets)
        
        active_tokens = int(np.sum(np.isin(bucket_ids, selected_buckets)))
        # Complexity is exactly N * K
        complexity = float(token_count * len(selected_buckets))
        
        return FieldAttentionResult(
            output=output,
            token_mass=masses,
            token_bucket=bucket_ids,
            bucket_potential=potential,
            selected_buckets=selected_buckets,
            active_tokens=active_tokens,
            complexity_estimate=complexity,
        )

    def _bucketize(self, coordinates: np.ndarray) -> np.ndarray:
        minimum = float(np.min(coordinates))
        maximum = float(np.max(coordinates))
        if maximum - minimum <= _EPS:
            return np.zeros_like(coordinates, dtype=int)
        scaled = (coordinates - minimum) / (maximum - minimum + _EPS)
        return np.minimum((scaled * self.config.num_buckets).astype(int), self.config.num_buckets - 1)

    def _aggregate_buckets(
        self, values: np.ndarray, coordinates: np.ndarray, masses: np.ndarray, bucket_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        centers_1d = np.zeros(self.config.num_buckets, dtype=float)
        vectors = np.zeros((self.config.num_buckets, self.config.dim), dtype=float)
        bucket_mass = np.zeros(self.config.num_buckets, dtype=float)
        
        for bucket in range(self.config.num_buckets):
            mask = bucket_ids == bucket
            if np.any(mask):
                centers_1d[bucket] = float(np.mean(coordinates[mask]))
                bucket_mass[bucket] = float(np.sum(masses[mask]))
                # Center of mass for the bucket
                vectors[bucket] = np.average(values[mask], axis=0, weights=masses[mask])
                
        return centers_1d, vectors, bucket_mass

    def _bucket_potential(self, centers_1d: np.ndarray, bucket_mass: np.ndarray) -> np.ndarray:
        potential = np.zeros(self.config.num_buckets, dtype=float)
        active = np.flatnonzero(bucket_mass > 0)
        for bucket in active:
            distance = np.abs(centers_1d[bucket] - centers_1d[active]) + self.config.temperature
            potential[bucket] = float(np.sum(bucket_mass[active] / np.power(distance, self.config.alpha)))
        return potential

    def _select_buckets(self, potential: np.ndarray) -> np.ndarray:
        active = np.flatnonzero(potential > 0)
        if active.size == 0:
            return np.array([0], dtype=int)
        count = min(self.config.top_buckets, active.size)
        order = np.argsort(-potential[active])[:count]
        return np.sort(active[order])

    def _attend_to_buckets(
        self, values: np.ndarray, bucket_vectors: np.ndarray, bucket_masses: np.ndarray, selected_buckets: np.ndarray
    ) -> np.ndarray:
        token_count = values.shape[0]
        output = np.zeros_like(values)
        
        # Only use selected buckets
        sel_vectors = bucket_vectors[selected_buckets]  # (K, dim)
        sel_masses = bucket_masses[selected_buckets]    # (K,)
        
        for i in range(token_count):
            # Distance from token i to all selected bucket centers
            distances = np.linalg.norm(sel_vectors - values[i], axis=1) + self.config.temperature
            weights = sel_masses / np.power(distances, self.config.alpha)
            weights = weights / np.clip(np.sum(weights), _EPS, None)
            output[i] = weights @ sel_vectors
            
        return output


def _as_tokens(tokens: np.ndarray, dim: int) -> np.ndarray:
    values = np.asarray(tokens, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != dim:
        raise ValueError(f"tokens must have shape (n, {dim}), got {values.shape}")
    if values.shape[0] == 0:
        raise ValueError("tokens must not be empty")
    return values
