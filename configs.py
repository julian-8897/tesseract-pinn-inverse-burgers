"""Typed configuration objects for inverse Burgers PINN runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from math import isfinite


@dataclass(frozen=True)
class ProblemConfig:
    """Physical inverse-problem setup."""

    true_viscosity: float = 0.05
    initial_viscosity: float = 0.01
    domain_x: tuple[float, float] = (0.0, 1.0)
    domain_t: tuple[float, float] = (0.0, 1.0)

    @property
    def domain(self):
        return {"x": self.domain_x, "t": self.domain_t}


@dataclass(frozen=True)
class DataConfig:
    """Observation-generation configuration."""

    n_obs: int = 80
    noise_std: float = 0.02
    seed: int = 123


@dataclass(frozen=True)
class TrainingConfig:
    """Optimization and collocation configuration."""

    n_epochs: int = 50
    log_nu_learning_rate: float = 0.1
    param_learning_rate: float = 1e-3
    n_col: int = 200
    n_ic: int = 50
    n_bc: int = 50


@dataclass(frozen=True)
class LossWeights:
    """Non-negative PINN loss weights."""

    data: float = 1.0
    physics: float = 0.1
    ic: float = 0.5
    bc: float = 0.5

    def as_dict(self):
        return {
            "data": self.data,
            "physics": self.physics,
            "ic": self.ic,
            "bc": self.bc,
        }


@dataclass(frozen=True)
class RunConfig:
    """Complete inverse-problem run configuration."""

    backend: str = "jax"
    problem: ProblemConfig = field(default_factory=ProblemConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossWeights = field(default_factory=LossWeights)

    def with_backend(self, backend: str):
        return replace(self, backend=backend)

    def with_seed(self, seed: int):
        return replace(self, data=replace(self.data, seed=seed))


DEFAULT_LOSS_WEIGHTS = LossWeights().as_dict()
LOSS_WEIGHT_NAMES = tuple(DEFAULT_LOSS_WEIGHTS)


def normalize_loss_weights(loss_weights=None):
    """Return validated non-negative loss weights with defaults filled in."""
    if loss_weights is None:
        return DEFAULT_LOSS_WEIGHTS.copy()
    if isinstance(loss_weights, LossWeights):
        loss_weights = loss_weights.as_dict()
    if not isinstance(loss_weights, Mapping):
        raise TypeError(
            "loss_weights must be a mapping of loss component names to weights"
        )

    unknown = set(loss_weights) - set(LOSS_WEIGHT_NAMES)
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown loss weight(s): {names}")

    normalized = DEFAULT_LOSS_WEIGHTS.copy()
    normalized.update(loss_weights)

    for name, value in normalized.items():
        value = float(value)
        if not isfinite(value):
            raise ValueError(f"Loss weight '{name}' must be finite")
        if value < 0:
            raise ValueError(f"Loss weight '{name}' must be non-negative")
        normalized[name] = value

    return normalized


def loss_weights_from_mapping(loss_weights=None):
    """Build a validated LossWeights instance from a partial mapping."""
    normalized = normalize_loss_weights(loss_weights)
    return LossWeights(**normalized)
