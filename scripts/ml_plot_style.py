"""Publication-quality matplotlib helpers for ML conference figures."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt

FigureWidth = Literal["single", "double", "wide"]

ML_PAPER_WIDTHS_IN = {
    "single": 3.25,
    "double": 6.75,
    "wide": 5.0,
}


def paper_size(
    width: FigureWidth = "single",
    ratio: float = 0.72,
) -> tuple[float, float]:
    """Return a compact ML-paper figure size in inches."""
    width_inches = ML_PAPER_WIDTHS_IN[width]
    return width_inches, width_inches * ratio


def apply_ml_style(
    *,
    font_size: float = 8.0,
    use_tex: bool = False,
    palette: str = "bright",
    require_scienceplots: bool = True,
) -> bool:
    """Apply SciencePlots plus conservative ML publication defaults."""
    try:
        import scienceplots  # noqa: F401
    except ImportError:
        if require_scienceplots:
            raise RuntimeError(
                "scienceplots is required; install the project dependencies first"
            ) from None
        plt.style.use("default")
        used_scienceplots = False
    else:
        styles = ["science", palette]
        if not use_tex:
            styles.append("no-latex")
        plt.style.use(styles)
        used_scienceplots = True

    mpl.rcParams.update(
        {
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "legend.fontsize": font_size - 1,
            "figure.titlesize": font_size,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "lines.linewidth": 1.3,
            "lines.markersize": 3.5,
            "legend.frameon": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.usetex": use_tex,
        }
    )
    return used_scienceplots


def save_figure(
    fig: mpl.figure.Figure,
    path_stem: str | Path,
    *,
    png: bool = True,
    pdf: bool = True,
    dpi: int = 300,
) -> None:
    """Save vector and high-resolution review copies."""
    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    if pdf:
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    if png:
        fig.savefig(
            stem.with_suffix(".png"),
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.02,
        )


def validate_publication_figure(
    fig: mpl.figure.Figure,
    *,
    require_axis_labels: bool = True,
    allow_titles: bool = False,
    allow_grids: bool = False,
) -> list[str]:
    """Return common publication-style issues before export."""
    issues = []
    for index, ax in enumerate(fig.axes, start=1):
        if not allow_titles and ax.get_title():
            issues.append(f"axis {index}: remove title and use a caption")
        if require_axis_labels and ax.get_visible():
            subplotspec = (
                ax.get_subplotspec() if hasattr(ax, "get_subplotspec") else None
            )
            if subplotspec is not None:
                if subplotspec.is_last_row() and not ax.get_xlabel():
                    issues.append(f"axis {index}: missing x-axis label")
                if subplotspec.is_first_col() and not ax.get_ylabel():
                    issues.append(f"axis {index}: missing y-axis label")
        if not allow_grids:
            if any(line.get_visible() for line in ax.get_xgridlines()):
                issues.append(f"axis {index}: remove x-grid")
            if any(line.get_visible() for line in ax.get_ygridlines()):
                issues.append(f"axis {index}: remove y-grid")
        for line in ax.lines:
            if line.get_linewidth() < 0.3:
                issues.append(f"axis {index}: line width below 0.3 pt")
    return issues
