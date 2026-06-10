"""Load local Tesseract component APIs without mutating global import paths."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]
TESSERACTS_ROOT = REPO_ROOT / "tesseracts"


def load_tesseract_api(component: str, *, module_name: str | None = None) -> ModuleType:
    """Load one local component's ``tesseract_api.py`` under a unique name."""
    if not component or Path(component).name != component:
        raise ValueError(f"Invalid Tesseract component name: {component!r}")

    module_path = TESSERACTS_ROOT / component / "tesseract_api.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"Tesseract API not found: {module_path}")

    resolved_name = module_name or f"_local_tesseract_{component}"
    spec = importlib.util.spec_from_file_location(resolved_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Tesseract API: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
