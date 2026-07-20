"""Thin matplotlib-compatible adapters over validated view frames."""

from __future__ import annotations

from typing import Any

import polars as pl

from cellpax.views import validate_view


def embedding_scatter(
    frame: pl.DataFrame,
    *,
    ax: Any | None = None,
    color_by: str = "taxonomy",
    point_size: float = 8.0,
    alpha: float = 0.8,
) -> Any:
    """Draw an embedding view on a matplotlib-compatible axes object."""
    validate_view("embedding", frame)
    if color_by not in {"taxonomy", "candidate", "none"}:
        raise ValueError("color_by must be 'taxonomy', 'candidate', or 'none'")
    if point_size <= 0:
        raise ValueError("point_size must be positive")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be within [0, 1]")
    if ax is None:
        try:
            import matplotlib.pyplot as plt
        except ImportError as error:
            raise ImportError(
                "embedding_scatter requires matplotlib when ax is not supplied"
            ) from error
        _, ax = plt.subplots()
    kwargs: dict[str, object] = {"s": point_size, "alpha": alpha, "linewidths": 0}
    if color_by == "taxonomy":
        kwargs["c"] = [color or "#808080" for color in frame["color"].to_list()]
    elif color_by == "candidate":
        kwargs["c"] = [
            -1 if candidate is None else candidate
            for candidate in frame["candidate_id"].to_list()
        ]
        kwargs["cmap"] = "tab20"
    else:
        kwargs["c"] = "#4c78a8"
    ax.scatter(frame["x"].to_list(), frame["y"].to_list(), **kwargs)
    ax.set_xlabel("Embedding dimension 0")
    ax.set_ylabel("Embedding dimension 1")
    ax.set_aspect("equal", adjustable="datalim")
    return ax
