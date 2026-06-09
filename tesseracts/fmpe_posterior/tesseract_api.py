# Copyright 2025 Pasteur Labs. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Amortized flow-matching posterior Tesseract.

Packages a trained FMPE (flow matching posterior estimation) network as a
framework-agnostic, swappable component. Given a sparse observation vector it
returns posterior samples and summaries of the Burgers parameters
``(nu, ic_amp, ic_phase)``. This is the third swappable Tesseract alongside the
JAX solver and the JAX/PyTorch PINN.

The trained posterior is loaded from ``posterior.pkl`` (copied into the image at
build time). This is an apply-only Tesseract -- it samples a learned posterior and
is not differentiated, so it exposes no VJP/JVP endpoints.
"""

import pathlib
import pickle

import numpy as np
import torch
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Float32

NUM_SAMPLES = 2000
BUNDLE_FORMAT = "tesseract-fmpe-posterior"
BUNDLE_VERSION = 1
_MODEL = None


def _load_model():
    """Lazily load the trained posterior bundle from the image."""
    global _MODEL
    if _MODEL is None:
        with open(pathlib.Path(__file__).parent / "posterior.pkl", "rb") as handle:
            _MODEL = pickle.load(handle)
        _validate_model(_MODEL)
    return _MODEL


def _validate_model(model):
    """Validate the persisted apply contract before serving requests."""
    required = {"posterior", "x_idx", "t_idx", "metadata"}
    missing = required - set(model)
    if missing:
        raise ValueError(f"Posterior bundle is missing keys: {sorted(missing)}")
    metadata = model["metadata"]
    if metadata.get("format") != BUNDLE_FORMAT:
        raise ValueError("Unsupported posterior bundle format")
    if int(metadata.get("version", -1)) != BUNDLE_VERSION:
        raise ValueError("Unsupported posterior bundle version")
    contract = metadata.get("contract", {})
    expected = int(contract.get("observation_dim", -1))
    if expected <= 0:
        raise ValueError("Posterior bundle has an invalid observation_dim")
    if len(model["x_idx"]) != expected or len(model["t_idx"]) != expected:
        raise ValueError("Posterior metadata does not match its sensor layout")


class InputSchema(BaseModel):
    """Input schema for the posterior sampler."""

    observation: Array[(None,), Float32] = Field(
        description="Sparse observation vector at the trained sensor layout"
    )
    seed: int = Field(default=0, description="Torch RNG seed for reproducible sampling")


class OutputSchema(BaseModel):
    """Posterior samples and per-parameter summaries (order: nu, ic_amp, ic_phase)."""

    samples: Array[(None, None), Float32] = Field(
        description="Posterior samples, shape (num_samples, n_params)"
    )
    mean: Array[(None,), Float32] = Field(description="Posterior mean per parameter")
    std: Array[(None,), Float32] = Field(description="Posterior std per parameter")
    q05: Array[(None,), Float32] = Field(description="5th percentile per parameter")
    q95: Array[(None,), Float32] = Field(description="95th percentile per parameter")
    model_id: str = Field(description="Identifier of the trained posterior contract")
    bundle_version: int = Field(description="Version of the persisted bundle format")
    observation_dim: int = Field(description="Expected observation-vector length")


def apply(inputs: InputSchema) -> OutputSchema:
    """Sample the amortized posterior for one observation."""
    model = _load_model()
    posterior = model["posterior"]
    metadata = model["metadata"]
    contract = metadata["contract"]

    torch.manual_seed(int(inputs.seed))
    observation = np.asarray(inputs.observation, dtype=np.float32).ravel()
    expected = int(contract["observation_dim"])
    if observation.size != expected:
        raise ValueError(
            f"Observation has length {observation.size}; model expects {expected}"
        )
    if not np.all(np.isfinite(observation)):
        raise ValueError("Observation must contain only finite values")
    x_o = torch.tensor(observation, dtype=torch.float32).reshape(1, -1)
    samples = (
        posterior.sample((NUM_SAMPLES,), x=x_o, show_progress_bars=False)
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    return OutputSchema(
        samples=samples,
        mean=samples.mean(axis=0),
        std=samples.std(axis=0),
        q05=np.percentile(samples, 5, axis=0).astype(np.float32),
        q95=np.percentile(samples, 95, axis=0).astype(np.float32),
        model_id=metadata["model_id"],
        bundle_version=metadata["version"],
        observation_dim=expected,
    )
