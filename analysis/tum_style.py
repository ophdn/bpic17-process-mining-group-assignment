"""TUM plotting colours and vector-figure export for the analysis notebooks.

The RGB values are transcribed from ``tum/tumcolors.sty`` in the companion
report repository ``6a562e35fea37a0c6eeea788``.  Keeping the palette here makes
the notebooks independent from a local checkout of the LaTeX report while
preserving its visual identity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
from cycler import cycler


TUM_COLORS = {
    # Primary colours
    "blue": "#0065BD",
    "white": "#FFFFFF",
    "black": "#000000",
    # Secondary colours
    "blue_dark": "#005293",
    "blue_darker": "#003359",
    "gray_dark": "#58585A",
    "gray": "#9C9D9F",
    "gray_light": "#D9DADB",
    # Accent colours
    "green": "#A2AD00",
    "orange": "#E37222",
    "ivory": "#DAD7CB",
    "blue_light": "#64A0C8",
    "blue_lighter": "#98C6EA",
    # Extended colours
    "violet": "#69085A",
    "navy": "#0F1B5F",
    "teal": "#00778A",
    "forest": "#007C30",
    "lime": "#679A1D",
    "yellow": "#FFDC00",
    "goldenrod": "#F9BA00",
    "pumpkin": "#D64C13",
    "red": "#C4071B",
    "maroon": "#9C0D16",
}

TUM_BLUE = TUM_COLORS["blue"]
TUM_BLUE_DARK = TUM_COLORS["blue_dark"]
TUM_BLUE_LIGHT = TUM_COLORS["blue_light"]
TUM_RED = TUM_COLORS["red"]
TUM_ORANGE = TUM_COLORS["orange"]
TUM_GREEN = TUM_COLORS["green"]
TUM_TEAL = TUM_COLORS["teal"]
TUM_VIOLET = TUM_COLORS["violet"]
TUM_GRAY = TUM_COLORS["gray"]
TUM_GRAY_DARK = TUM_COLORS["gray_dark"]
TUM_GRAY_LIGHT = TUM_COLORS["gray_light"]

TUM_SEQUENCE = (
    TUM_BLUE,
    TUM_ORANGE,
    TUM_TEAL,
    TUM_RED,
    TUM_GREEN,
    TUM_VIOLET,
    TUM_BLUE_LIGHT,
    TUM_GRAY_DARK,
)

TUM_BLUE_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "TUMBlue",
    (TUM_COLORS["white"], TUM_COLORS["blue_lighter"], TUM_BLUE),
)


def apply_tum_style() -> None:
    """Apply a restrained report-ready Matplotlib style using TUM colours."""

    mpl.rcParams.update(
        {
            "axes.axisbelow": True,
            "axes.edgecolor": TUM_GRAY,
            "axes.grid": True,
            "axes.labelcolor": TUM_COLORS["black"],
            "axes.prop_cycle": cycler(color=TUM_SEQUENCE),
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.titlecolor": TUM_BLUE_DARK,
            "figure.facecolor": TUM_COLORS["white"],
            "font.family": "sans-serif",
            "grid.color": TUM_GRAY_LIGHT,
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.facecolor": TUM_COLORS["white"],
            "svg.fonttype": "none",
            "xtick.color": TUM_GRAY_DARK,
            "ytick.color": TUM_GRAY_DARK,
        }
    )


def _repository_root() -> Path:
    here = Path.cwd().resolve()
    for candidate in (here, *here.parents):
        if (candidate / "analysis").is_dir() and (candidate / "notebooks").is_dir():
            return candidate
    raise RuntimeError("Could not locate the BPIC-17 repository root.")


def save_figure(
    figure,
    name: str,
    *,
    formats: Iterable[str] = ("pdf", "svg"),
    directory: str | Path | None = None,
    dpi: int = 300,
) -> tuple[Path, ...]:
    """Save a notebook figure in report-ready vector formats.

    ``name`` is a filename stem.  By default figures are written to the
    repository-level ``visualization/`` directory, which is created on demand.
    """

    if not name or Path(name).name != name:
        raise ValueError("name must be a non-empty filename stem without directories")

    target = Path(directory) if directory is not None else _repository_root() / "visualization"
    target.mkdir(parents=True, exist_ok=True)

    written = []
    for suffix in formats:
        suffix = suffix.lower().lstrip(".")
        if suffix not in {"pdf", "svg"}:
            raise ValueError(f"Unsupported vector format: {suffix!r}")
        path = target / f"{name}.{suffix}"
        figure.savefig(path, format=suffix, dpi=dpi)
        written.append(path)
    return tuple(written)

