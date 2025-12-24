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

    
    return fenicsx_ii.Circle(G.mesh, radius, degree=degree, **kwargs)


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

    
    return fenicsx_ii.Disk(G.mesh, radius, degree=degree, **kwargs)


def quadrature_restriction_space(mesh1d, degree: int, *, value_shape=()):
    import basix.ufl
    import dolfinx.fem

    q_el = basix.ufl.quadrature_element(
        mesh1d.basix_cell(), value_shape=value_shape, degree=degree
    )
    return dolfinx.fem.functionspace(mesh1d, q_el)


__all__ = ["circle_operator", "disk_operator", "quadrature_restriction_space"]
