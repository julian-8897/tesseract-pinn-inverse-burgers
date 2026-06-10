"""Versioned, atomic checkpoints for deterministic inverse training."""

from __future__ import annotations

import json
import os
import pickle
import tempfile
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

CHECKPOINT_FORMAT = "burgers-inverse-training"
CHECKPOINT_VERSION = 1


def config_fingerprint(config) -> str:
    """Hash resume-critical configuration, allowing only ``n_epochs`` to change."""
    payload = asdict(config)
    payload["training"].pop("n_epochs", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def save_training_checkpoint(path, payload) -> Path:
    """Atomically persist one trusted local training checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        **payload,
    }

    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        try:
            pickle.dump(checkpoint, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

    temporary_path.replace(path)
    return path


def load_training_checkpoint(path):
    """Load and validate a trusted local training checkpoint."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Training checkpoint not found: {path}")
    with path.open("rb") as handle:
        checkpoint = pickle.load(handle)

    if not isinstance(checkpoint, dict):
        raise ValueError("Training checkpoint must contain a mapping")
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Unsupported training checkpoint format")
    version = int(checkpoint.get("version", -1))
    if version != CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported training checkpoint version: {version}")
    return checkpoint
