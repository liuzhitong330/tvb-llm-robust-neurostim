"""Lazy imports and diagnostics for optional TVB dependencies."""

from __future__ import annotations

from typing import Any


def load_tvb_lab() -> Any:
    """Import ``tvb.simulator.lab`` with a project-specific error message."""

    try:
        from tvb.simulator import lab
    except ImportError as exc:
        raise RuntimeError(
            "The Virtual Brain is required for simulation runs. Install the optional "
            "dependency with `uv sync --extra tvb` or use an environment where TVB is "
            "already available, then run the script again with `uv run`."
        ) from exc
    return lab
