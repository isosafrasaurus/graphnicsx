

from .fenics_graph import (
    FenicsGraph,
    BIF_IN,
    BIF_OUT,
    BOUN_IN,
    BOUN_OUT,
    copy_from_nx_graph,
    nxgraph_attribute_to_dolfinx,
)
from .generators import line_graph, honeycomb, Y_bifurcation, YY_bifurcation
from .graph_utils import color_graph, plot_graph_color, assign_radius_using_Murrays_law, DistFromSource
from .plot import TubeFile
from .ii import circle_operator, disk_operator













def _patch_dolfinx_functionspace() -> None:
    try:
        import dolfinx.fem as _fem
    except Exception:
        return

    if hasattr(_fem, "functionspace") and isinstance(getattr(_fem, "FunctionSpace", None), type):
        
        if not hasattr(_fem, "FunctionSpaceClass"):
            _fem.FunctionSpaceClass = _fem.FunctionSpace  
        _fem.FunctionSpace = _fem.functionspace  


_patch_dolfinx_functionspace()

__all__ = [
    "FenicsGraph",
    "copy_from_nx_graph",
    "nxgraph_attribute_to_dolfinx",
    "BIF_IN",
    "BIF_OUT",
    "BOUN_IN",
    "BOUN_OUT",
    "line_graph",
    "honeycomb",
    "Y_bifurcation",
    "YY_bifurcation",
    "color_graph",
    "plot_graph_color",
    "assign_radius_using_Murrays_law",
    "DistFromSource",
    "TubeFile",
    "circle_operator",
    "disk_operator",
]
