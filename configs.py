"""Typed configuration objects for inverse Burgers PINN runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from math import isfinite

# Default Gaussian sensor-noise standard deviation. Single source of truth shared
# by the deterministic observation samplers and the FMPE simulator so the two paths
# never silently disagree.
DEFAULT_NOISE_STD = 0.02
DEFAULT_FMPE_PRIOR_LOW = (0.02, 0.8, -0.4)
DEFAULT_FMPE_PRIOR_HIGH = (0.10, 1.2, 0.4)


def _check_positive(name, value):
    value = float(value)
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _check_unit_interval(name, value):
    value = float(value)
    if not (0.0 < value < 1.0):
        raise ValueError(f"{name} must lie in the open interval (0, 1)")
    return value


@dataclass(frozen=True)
class ProblemConfig:
    """Physical inverse-problem setup."""

    true_viscosity: float = 0.05
    initial_viscosity: float = 0.01
    domain_x: tuple[float, float] = (0.0, 1.0)
    domain_t: tuple[float, float] = (0.0, 1.0)
    # Dispersion coefficient of the KdV-Burgers truth (`-beta u_xxx`). This is the
    # un-modeled physics the in-loop viscous-Burgers solver omits; the hybrid
    # discrepancy term learns its effect. beta=0 reduces the truth to plain
    # viscous Burgers (discrepancy collapses to zero — useful as a sanity check).
    dispersion_beta: float = 1e-3

    def __post_init__(self):
        _check_positive("true_viscosity", self.true_viscosity)
        _check_positive("initial_viscosity", self.initial_viscosity)
        beta = float(self.dispersion_beta)
        if not isfinite(beta) or beta < 0:
            raise ValueError("dispersion_beta must be a finite non-negative value")
        for name, bounds in (("domain_x", self.domain_x), ("domain_t", self.domain_t)):
            lo, hi = float(bounds[0]), float(bounds[1])
            if not (isfinite(lo) and isfinite(hi)):
                raise ValueError(f"{name} bounds must be finite")
            if lo >= hi:
                raise ValueError(f"{name} lower bound must be less than upper bound")

    @property
    def domain(self):
        return {"x": self.domain_x, "t": self.domain_t}


@dataclass(frozen=True)
class DataConfig:
    """Observation-generation configuration."""

    n_obs: int = 80
    noise_std: float = DEFAULT_NOISE_STD
    seed: int = 123

    def __post_init__(self):
        if int(self.n_obs) <= 0:
            raise ValueError("n_obs must be a positive integer")
        if float(self.noise_std) < 0 or not isfinite(float(self.noise_std)):
            raise ValueError("noise_std must be a finite non-negative value")
        if int(self.seed) < 0:
            raise ValueError("seed must be a non-negative integer")


@dataclass(frozen=True)
class TrainingConfig:
    """Optimization and collocation configuration."""

    n_epochs: int = 50
    log_nu_learning_rate: float = 0.1
    param_learning_rate: float = 1e-3
    adaptive_loss_weights: bool = False
    brdr_beta_c: float = 0.9999
    brdr_beta_w: float = 0.999
    brdr_epsilon: float = 1e-12
    n_col: int = 200
    n_ic: int = 50
    n_bc: int = 50
    viscosity_warmup_epochs: int = 0
    clip_log_viscosity: bool = False
    nu_clip_min: float = 1e-4
    nu_clip_max: float = 0.5
    # Hybrid-mode discrepancy regularization (Stage 1). The L2 weight keeps the
    # learned discrepancy small so the physical viscosity stays identifiable; the
    # optional smoothness weight penalizes the discrepancy's spatial gradient.
    discrepancy_reg_weight: float = 1.0
    discrepancy_smooth_weight: float = 0.0

    def __post_init__(self):
        if int(self.n_epochs) <= 0:
            raise ValueError("n_epochs must be a positive integer")
        _check_positive("log_nu_learning_rate", self.log_nu_learning_rate)
        _check_positive("param_learning_rate", self.param_learning_rate)
        _check_unit_interval("brdr_beta_c", self.brdr_beta_c)
        _check_unit_interval("brdr_beta_w", self.brdr_beta_w)
        _check_positive("brdr_epsilon", self.brdr_epsilon)
        for name in ("n_col", "n_ic", "n_bc"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("discrepancy_reg_weight", "discrepancy_smooth_weight"):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative value")
        if int(self.viscosity_warmup_epochs) < 0:
            raise ValueError("viscosity_warmup_epochs must be non-negative")
        if self.clip_log_viscosity:
            lo = _check_positive("nu_clip_min", self.nu_clip_min)
            hi = _check_positive("nu_clip_max", self.nu_clip_max)
            if lo >= hi:
                raise ValueError("nu_clip_min must be less than nu_clip_max")


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


@dataclass(frozen=True)
class FMPEConfig:
    """Reproducible configuration for FMPE simulation and training."""

    n_sims: int = 4000
    n_sensors: int = 64
    noise_std: float = DEFAULT_NOISE_STD
    sensor_seed: int = 0
    simulation_seed: int = 0
    training_seed: int = 1
    device: str = "cpu"
    max_num_epochs: int | None = None
    show_train_summary: bool = False
    prior_low: tuple[float, float, float] = DEFAULT_FMPE_PRIOR_LOW
    prior_high: tuple[float, float, float] = DEFAULT_FMPE_PRIOR_HIGH

    def __post_init__(self):
        for name in ("n_sims", "n_sensors"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if float(self.noise_std) < 0 or not isfinite(float(self.noise_std)):
            raise ValueError("noise_std must be a finite non-negative value")
        for name in ("sensor_seed", "simulation_seed", "training_seed"):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.max_num_epochs is not None and int(self.max_num_epochs) <= 0:
            raise ValueError("max_num_epochs must be a positive integer or None")
        if not self.device:
            raise ValueError("device must be a non-empty string")
        if len(self.prior_low) != 3 or len(self.prior_high) != 3:
            raise ValueError("FMPE prior bounds must contain three parameters")
        for index, (low, high) in enumerate(
            zip(self.prior_low, self.prior_high, strict=True)
        ):
            if not (isfinite(float(low)) and isfinite(float(high))):
                raise ValueError(f"FMPE prior bounds at index {index} must be finite")
            if float(low) >= float(high):
                raise ValueError(
                    f"FMPE prior lower bound at index {index} must be less than upper"
                )


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
