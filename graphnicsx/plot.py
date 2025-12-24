from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple, Union

import dolfinx
import dolfinx.fem
import dolfinx.geometry
import dolfinx.mesh
import numpy as np
from mpi4py import MPI

from .fenics_graph import FenicsGraph

_pvd_header = """<?xml version=\"1.0\"?>\n<VTKFile type=\"Collection\" version=\"0.1\">\n  <Collection>\n"""
_pvd_footer = """  </Collection>\n</VTKFile>\n"""


def _safe_name(func) -> str:
    return getattr(func, "name", None) or getattr(func, "_name", None) or "f"


def _vertex_incident_owned_cell(mesh: dolfinx.mesh.Mesh, *, owned_cell_mask: np.ndarray) -> np.ndarray:
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(0, tdim)
    v_to_c = mesh.topology.connectivity(0, tdim)

    num_vertices = mesh.topology.index_map(0).size_local + mesh.topology.index_map(0).num_ghosts
    cell_of_vertex = -np.ones(num_vertices, dtype=np.int32)

    for v in range(num_vertices):
        cells = v_to_c.links(v)
        if len(cells) == 0:
            continue

        for c in cells:
            c = int(c)
            if c < owned_cell_mask.shape[0] and owned_cell_mask[c]:
                cell_of_vertex[v] = c
                break

        if cell_of_vertex[v] < 0:
            cell_of_vertex[v] = int(cells[0])

    return cell_of_vertex


def _point_values_on_vertices(func: dolfinx.fem.Function, vertex_coords: np.ndarray,
                              cell_of_vertex: np.ndarray) -> np.ndarray:
    pts = np.asarray(vertex_coords, dtype=np.float64)
    if pts.shape[1] != 3:
        raise ValueError("vertex_coords must have shape (N, 3)")
    cells = np.asarray(cell_of_vertex, dtype=np.int32)

    bad = cells < 0
    if bad.any():
        cells = cells.copy()
        cells[bad] = 0

    vals = func.eval(pts, cells)
    vals = np.asarray(vals)
    if vals.ndim == 2:
        vals = vals[:, 0]
    if bad.any():
        vals = vals.copy()
        vals[bad] = np.nan
    return vals


def _radius_values_on_vertices(G: FenicsGraph, used_vertices: np.ndarray, cell_of_vertex: np.ndarray,
                               cell_to_edge: np.ndarray) -> np.ndarray:
    radii = G.edge_attribute_array("radius")

    mesh = G.mesh
    assert mesh is not None
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(0, tdim)
    v_to_c = mesh.topology.connectivity(0, tdim)

    out = np.zeros(len(used_vertices), dtype=np.float64)

    for i, v in enumerate(used_vertices):
        v = int(v)
        incident_cells = v_to_c.links(v)
        edge_ids = []
        for c in incident_cells:
            c = int(c)
            if c < cell_to_edge.shape[0] and cell_to_edge[c] >= 0:
                edge_ids.append(int(cell_to_edge[c]))
        if len(edge_ids) == 0:

            c = int(cell_of_vertex[v])
            if 0 <= c < cell_to_edge.shape[0]:
                eid = int(cell_to_edge[c])
                out[i] = float(radii[eid]) if eid >= 0 else np.nan
            else:
                out[i] = np.nan
            continue
        out[i] = float(np.mean([radii[eid] for eid in set(edge_ids)]))

    return out


@dataclass
class TubeFile:
    G: FenicsGraph
    fname: str

    def __post_init__(self):
        root, ext = os.path.splitext(self.fname)
        if ext.lower() != ".pvd":
            raise ValueError("TubeFile must have a .pvd file ending")

        if self.G.mesh is None:
            raise RuntimeError("Graph has no mesh. Call G.make_mesh() first")

        if self.G.mesh.geometry.dim != 3:
            raise ValueError(
                f"Mesh geometry dim is {self.G.mesh.geometry.dim}, expected 3"
            )

        if len(nx.get_edge_attributes(self.G, "radius")) == 0:
            raise ValueError("Graph must have a 'radius' attribute on edges")

        self._base = root
        self._comm: MPI.Intracomm = self.G.mesh.comm

        if self._comm.rank == 0:
            with open(self.fname, "w", encoding="utf-8") as f:
                f.write(_pvd_header)
                f.write(_pvd_footer)

    def __lshift__(self, func_and_time: Union[dolfinx.fem.Function, Tuple[dolfinx.fem.Function, float]]):

        if isinstance(func_and_time, tuple):
            func, t = func_and_time
        else:
            func, t = func_and_time, 0.0

        mesh = self.G.mesh
        assert mesh is not None
        tdim = mesh.topology.dim

        num_cells_local = mesh.topology.index_map(tdim).size_local
        owned_cells = np.arange(num_cells_local, dtype=np.int32)
        owned_cell_mask = np.ones(num_cells_local, dtype=bool)

        mesh.topology.create_connectivity(tdim, 0)
        c_to_v = mesh.topology.connectivity(tdim, 0)

        used = set()
        for c in owned_cells:
            vs = c_to_v.links(int(c))
            used.update(int(v) for v in vs)
        used_vertices = np.array(sorted(used), dtype=np.int32)

        new_id = {int(v): i for i, v in enumerate(used_vertices)}

        X = mesh.geometry.x[used_vertices]

        cell_of_vertex = _vertex_incident_owned_cell(mesh, owned_cell_mask=owned_cell_mask)

        if self.G.edge_tags is None:
            raise RuntimeError("Graph has no edge_tags; call G.make_mesh()")
        cell_to_edge = np.full(num_cells_local, -1, dtype=np.int32)
        cell_to_edge[self.G.edge_tags.indices] = self.G.edge_tags.values

        fvals = _point_values_on_vertices(func, X, cell_of_vertex[used_vertices])

        rvals = _radius_values_on_vertices(self.G, used_vertices, cell_of_vertex, cell_to_edge)

        try:
            from vtk import vtkCellArray, vtkDoubleArray, vtkLine, vtkPoints, vtkPolyData, vtkXMLPolyDataWriter
        except Exception as e:
            raise ImportError(
                "VTK python bindings are required for TubeFile. Install `vtk`."
            ) from e

        points = vtkPoints()
        for p in X:
            points.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))

        lines = vtkCellArray()
        for c in owned_cells:
            vs = c_to_v.links(int(c))
            if len(vs) != 2:
                continue
            a, b = int(vs[0]), int(vs[1])
            line = vtkLine()
            line.GetPointIds().SetId(0, new_id[a])
            line.GetPointIds().SetId(1, new_id[b])
            lines.InsertNextCell(line)

        poly = vtkPolyData()
        poly.SetPoints(points)
        poly.SetLines(lines)

        data_f = vtkDoubleArray()
        data_f.SetName(_safe_name(func))
        data_f.SetNumberOfComponents(1)
        for val in fvals:
            data_f.InsertNextTuple([float(val)])
        poly.GetPointData().AddArray(data_f)

        data_r = vtkDoubleArray()
        data_r.SetName("radius")
        data_r.SetNumberOfComponents(1)
        for val in rvals:
            data_r.InsertNextTuple([float(val)])
        poly.GetPointData().AddArray(data_r)

        if self._comm.size == 1:
            vtp_name = f"{self._base}{int(t):06d}.vtp"
        else:
            vtp_name = f"{self._base}_p{self._comm.rank}{int(t):06d}.vtp"

        writer = vtkXMLPolyDataWriter()
        writer.SetFileName(vtp_name)
        writer.SetInputData(poly)
        writer.Update()
        writer.Write()

        self._comm.Barrier()
        if self._comm.rank == 0:
            short_base = os.path.basename(self._base)
            with open(self._base + ".pvd", "r", encoding="utf-8") as f:
                content = f.read().splitlines()

            entries = []
            if self._comm.size == 1:
                entries.append(
                    f"    <DataSet timestep=\"{t}\" part=\"0\" file=\"{short_base}{int(t):06d}.vtp\" />"
                )
            else:
                for r in range(self._comm.size):
                    entries.append(
                        f"    <DataSet timestep=\"{t}\" part=\"{r}\" file=\"{short_base}_p{r}{int(t):06d}.vtp\" />"
                    )

            updated = content[:-2] + entries + content[-2:]
            with open(self._base + ".pvd", "w", encoding="utf-8") as f:
                f.write("\n".join(updated) + "\n")

        self._comm.Barrier()
        return self


import networkx as nx
