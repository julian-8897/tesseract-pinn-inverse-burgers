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
_MODEL = None


def _load_model():
    """Lazily load the trained posterior bundle from the image."""
    global _MODEL
    if _MODEL is None:
        with open(pathlib.Path(__file__).parent / "posterior.pkl", "rb") as handle:
            _MODEL = pickle.load(handle)
    return _MODEL


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


def apply(inputs: InputSchema) -> OutputSchema:
    """Sample the amortized posterior for one observation."""
    model = _load_model()
    posterior = model["posterior"]

    torch.manual_seed(int(inputs.seed))
    x_o = torch.tensor(np.asarray(inputs.observation), dtype=torch.float32).reshape(
        1, -1
    )
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
    )
