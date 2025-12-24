from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from .fenics_graph import FenicsGraph


def _require_fenicsx_ii():
    try:
        import fenicsx_ii  
    except ImportError as e:
        raise ImportError(
            "fenicsx_ii is required for graphnicsx.ii helpers. "
            "Install fenicsx_ii or avoid importing graphnicsx.ii"
        ) from e
    return fenicsx_ii


def circle_operator(
    G: FenicsGraph,
    degree: int,
    *,
    radius: float | Callable[[np.ndarray], np.ndarray] | None = None,
    radius_attr: str = "radius",
    **kwargs: Any,
):
    

    fenicsx_ii = _require_fenicsx_ii()

    if radius is None:
        radius = G.edge_attribute_callable(radius_attr)

    return fenicsx_ii.Circle(G.mesh, degree, radius=radius, **kwargs)


def disk_operator(
    G: FenicsGraph,
    degree: int,
    *,
    radius: float | Callable[[np.ndarray], np.ndarray] | None = None,
    radius_attr: str = "radius",
    **kwargs: Any,
):
    

    fenicsx_ii = _require_fenicsx_ii()

    if radius is None:
        radius = G.edge_attribute_callable(radius_attr)

    return fenicsx_ii.Disk(G.mesh, degree, radius=radius, **kwargs)


__all__ = ["circle_operator", "disk_operator"]
