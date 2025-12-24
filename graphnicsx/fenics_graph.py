from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import networkx as nx
import numpy as np

try:
    from mpi4py import MPI
    import basix.ufl
    import dolfinx
    import dolfinx.fem
    import dolfinx.mesh
    import ufl
except ImportError as e:  
    raise ImportError(
        "graphnicsx requires dolfinx (>=0.10.0), basix, ufl and mpi4py. "
        "Install FEniCSx first, then import graphnicsx."
    ) from e



BIF_IN = 1
BIF_OUT = 2
BOUN_IN = 3
BOUN_OUT = 4


def _as_3d(x: Iterable[float]) -> np.ndarray:
    
    arr = np.asarray(list(x), dtype=np.float64)
    if arr.shape == (3,):
        return arr
    if arr.shape == (2,):
        return np.array([arr[0], arr[1], 0.0], dtype=np.float64)
    if arr.shape == (1,):
        return np.array([arr[0], 0.0, 0.0], dtype=np.float64)
    raise ValueError(f"Node coordinate must be 1D/2D/3D, got shape {arr.shape}")


@dataclass(frozen=True)
class EdgeSubmesh:
    

    mesh: dolfinx.mesh.Mesh
    cell_map: dolfinx.mesh.EntityMap
    vertex_map: dolfinx.mesh.EntityMap
    geom_node_map: np.ndarray
    vertex_tags: dolfinx.mesh.MeshTags


class FenicsGraph(nx.DiGraph):
    

    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data=incoming_graph_data, **attr)

        
        self.mesh: Optional[dolfinx.mesh.Mesh] = None
        self.edge_tags: Optional[dolfinx.mesh.MeshTags] = None
        self.geom_dim: int = 3
        self.num_edges: int = 0
        self.edge_list: List[Tuple[int, int]] = []
        self.edge_to_id: Dict[Tuple[int, int], int] = {}
        self.node_to_vertex: Dict[int, int] = {}
        self.tangent: Optional[dolfinx.fem.Function] = None

        
        
        
        
        
        self.edge_submeshes: Dict[Tuple[int, int], Optional[EdgeSubmesh]] = {}

        
        self.bifurcation_ixs: List[int] = []
        self.boundary_ixs: List[int] = []
        self.num_bifurcations: int = 0

    def get_mesh(
        self,
        n: int = 1,
        comm: MPI.Intracomm = MPI.COMM_WORLD,
        partitioner=None,
    ) -> Tuple[dolfinx.mesh.Mesh, dolfinx.mesh.MeshTags]:
        if len(self.nodes) == 0:
            raise ValueError("Cannot build mesh for empty graph")
        if len(self.edges) == 0:
            raise ValueError("Cannot build mesh for graph with no edges")
        if n < 0:
            raise ValueError("n must be >= 0")

        
        self.edge_list = list(self.edges())
        self.edge_to_id = {e: i for i, e in enumerate(self.edge_list)}
        self.num_edges = len(self.edge_list)

        
        
        node_list = list(self.nodes())
        coords_nodes = np.zeros((len(node_list), 3), dtype=np.float64)
        self.node_to_vertex = {}
        for i, v in enumerate(node_list):
            if "pos" not in self.nodes[v]:
                raise KeyError(f"Node {v} is missing required attribute 'pos'")
            coords_nodes[i] = _as_3d(self.nodes[v]["pos"])
            self.node_to_vertex[v] = i

        
        nseg = 2**n
        coords: List[np.ndarray] = [coords_nodes[i] for i in range(coords_nodes.shape[0])]
        cells: List[Tuple[int, int]] = []
        cell_edge_ids: List[int] = []

        
        for edge_id, (u, v) in enumerate(self.edge_list):
            p0 = coords_nodes[self.node_to_vertex[u]]
            p1 = coords_nodes[self.node_to_vertex[v]]

            
            v0 = self.node_to_vertex[u]
            v1 = self.node_to_vertex[v]

            if nseg == 1:
                
                cells.append((v0, v1))
                cell_edge_ids.append(edge_id)
                continue

            
            
            chain = [v0]
            for k in range(1, nseg):
                t = k / nseg
                pk = (1.0 - t) * p0 + t * p1
                idx = len(coords)
                coords.append(pk)
                chain.append(idx)
            chain.append(v1)

            for a, b in zip(chain[:-1], chain[1:]):
                cells.append((a, b))
                cell_edge_ids.append(edge_id)

        
        
        
        
        self._cell_edge_ids_global = np.asarray(cell_edge_ids, dtype=np.int32)

        x = np.asarray(coords, dtype=np.float64)
        
        cell_array = np.asarray(cells, dtype=np.int64)

        
        
        
        
        
        
        
        
        
        if comm.size > 1:
            if partitioner is None:
                partitioner = dolfinx.mesh.create_cell_partitioner(
                    dolfinx.mesh.GhostMode.shared_facet
                )
            if comm.rank != 0:
                x = np.zeros((0, 3), dtype=np.float64)
                cell_array = np.zeros((0, 2), dtype=np.int64)

        
        
        coord_el = basix.ufl.element("Lagrange", "interval", 1, shape=(3,))
        domain = ufl.Mesh(coord_el)
        mesh = dolfinx.mesh.create_mesh(
            comm, cell_array, domain, x, partitioner=partitioner
        )

        
        
        tdim = mesh.topology.dim
        assert tdim == 1
        num_cells_local = mesh.topology.index_map(tdim).size_local
        cell_indices = np.arange(num_cells_local, dtype=np.int32)
        
        
        
        
        try:
            orig = mesh.topology.original_cell_index
            
            orig_owned = np.asarray(orig[:num_cells_local], dtype=np.int64)
            values = np.asarray([cell_edge_ids[i] for i in orig_owned], dtype=np.int32)
        except Exception:
            
            values = np.asarray(cell_edge_ids[:num_cells_local], dtype=np.int32)

        edge_tags = dolfinx.mesh.meshtags(mesh, tdim, cell_indices, values)

        return mesh, edge_tags

    def make_mesh(
        self,
        n: int = 1,
        comm: MPI.Intracomm = MPI.COMM_WORLD,
        partitioner=None,
    ) -> dolfinx.mesh.Mesh:
        
        mesh, tags = self.get_mesh(n=n, comm=comm, partitioner=partitioner)
        self.mesh = mesh
        self.edge_tags = tags

        
        self.record_bifurcation_and_boundary_nodes()
        self.assign_tangents()
        return mesh

    def make_submeshes(self) -> None:
        
        if self.mesh is None or self.edge_tags is None:
            raise RuntimeError("Call make_mesh() before make_submeshes().")

        
        self.edge_submeshes = {e: None for e in self.edge_list}

        
        self.mesh.topology.create_connectivity(0, 1)

        for edge_id, e in enumerate(self.edge_list):
            u, v = e
            assert self.edge_to_id[e] == edge_id

            
            edge_cells = self.edge_tags.find(edge_id)
            edge_cells = np.asarray(edge_cells, dtype=np.int32)
            if edge_cells.size == 0:
                
                
                continue

            submesh, cell_map, vertex_map, geom_node_map = dolfinx.mesh.create_submesh(
                self.mesh, 1, edge_cells
            )

            
            
            
            
            
            
            
            
            x_sub = submesh.geometry.x
            p_u = _as_3d(self.nodes[u]["pos"])
            p_v = _as_3d(self.nodes[v]["pos"])

            def _find_vertex(pt: np.ndarray, tol: float = 1.0e-12) -> int | None:
                d = np.linalg.norm(x_sub - pt, axis=1)
                hits = np.where(d < tol)[0]
                return int(hits[0]) if hits.size else None

            u_vertex = _find_vertex(p_u)
            v_vertex = _find_vertex(p_v)

            
            entities: List[int] = []
            values: List[int] = []

            
            if u_vertex is not None:
                if u in self.bifurcation_ixs:
                    entities.append(int(u_vertex))
                    values.append(BIF_OUT)
                elif u in self.boundary_ixs:
                    entities.append(int(u_vertex))
                    values.append(BOUN_IN)

            
            if v_vertex is not None:
                if v in self.bifurcation_ixs:
                    entities.append(int(v_vertex))
                    values.append(BIF_IN)
                elif v in self.boundary_ixs:
                    entities.append(int(v_vertex))
                    values.append(BOUN_OUT)

            if len(entities) == 0:
                
                
                
                vertex_tags = dolfinx.mesh.meshtags(
                    submesh,
                    0,
                    np.array([], dtype=np.int32),
                    np.array([], dtype=np.int32),
                )
            else:
                vertex_tags = dolfinx.mesh.meshtags(
                    submesh,
                    0,
                    np.asarray(entities, dtype=np.int32),
                    np.asarray(values, dtype=np.int32),
                )

            self.edges[e]["submesh"] = EdgeSubmesh(
                mesh=submesh,
                cell_map=cell_map,
                vertex_map=vertex_map,
                geom_node_map=geom_node_map,
                vertex_tags=vertex_tags,
            )

            
            self.edge_submeshes[e] = self.edges[e]["submesh"]

    
    
    

    def record_bifurcation_and_boundary_nodes(self) -> None:
        
        bifurcation_ixs: List[int] = []
        boundary_ixs: List[int] = []
        for v in self.nodes():
            num_conn_edges = len(self.in_edges(v)) + len(self.out_edges(v))
            if num_conn_edges == 1:
                boundary_ixs.append(v)
            elif num_conn_edges > 1:
                bifurcation_ixs.append(v)
            elif num_conn_edges == 0:
                
                if self.mesh is None or self.mesh.comm.rank == 0:
                    print(f"Node {v} in G is lonely (i.e. unconnected)")

        self.bifurcation_ixs = bifurcation_ixs
        self.num_bifurcations = len(bifurcation_ixs)
        self.boundary_ixs = boundary_ixs

    def compute_edge_lengths(self) -> None:
        
        for (u, v) in self.edges():
            p0 = _as_3d(self.nodes[u]["pos"])
            p1 = _as_3d(self.nodes[v]["pos"])
            self.edges[u, v]["length"] = float(np.linalg.norm(p1 - p0))

    def compute_vertex_degrees(self) -> None:
        
        if len(self.edges) == 0:
            return
        
        e0 = next(iter(self.edges))
        if "length" not in self.edges[e0]:
            self.compute_edge_lengths()

        for v in self.nodes():
            l_v = 0.0
            for e in self.in_edges(v):
                l_v += float(self.edges[e]["length"])
            for e in self.out_edges(v):
                l_v += float(self.edges[e]["length"])
            self.nodes[v]["degree"] = l_v / 2.0

        degrees = nx.get_node_attributes(self, "degree")
        if degrees:
            self.degree_min = min(degrees.values())
            self.degree_max = max(degrees.values())
        else:
            self.degree_min = 0.0
            self.degree_max = 0.0

    
    
    

    def assign_tangents(self) -> None:
        
        if self.mesh is None or self.edge_tags is None:
            raise RuntimeError("Call make_mesh() before assign_tangents().")

        
        for (u, v) in self.edge_list:
            t = _as_3d(self.nodes[v]["pos"]) - _as_3d(self.nodes[u]["pos"])
            norm = np.linalg.norm(t)
            if norm == 0.0:
                raise ValueError(f"Zero-length edge {u}->{v}")
            self.edges[u, v]["tangent"] = (t / norm).astype(np.float64)

        
        Vt = dolfinx.fem.functionspace(self.mesh, ("DG", 0, (3,)))
        tangent = dolfinx.fem.Function(Vt)

        
        tdim = self.mesh.topology.dim
        num_cells = self.mesh.topology.index_map(tdim).size_local
        dm = Vt.dofmap.list[:num_cells]
        
        
        
        
        
        value_shape = Vt.element.basix_element.value_shape
        value_size = int(np.prod(value_shape)) if len(value_shape) > 0 else 1
        bs = Vt.dofmap.bs
        if bs > 1:
            
            assert value_size == 1, "Unexpected combined value_shape and block size"
            value_size = bs
        assert value_size == 3

        
        
        
        cell_edge_ids = np.asarray(self.edge_tags.values[:num_cells], dtype=np.int32)
        edge_tangents = [self.edges[e]["tangent"] for e in self.edge_list]

        
        arr = tangent.x.array
        for c in range(num_cells):
            dof = int(dm[c, 0])
            tvec = edge_tangents[int(cell_edge_ids[c])]
            arr[dof * value_size : (dof + 1) * value_size] = tvec
        tangent.x.scatter_forward()
        self.tangent = tangent

    def dds(self, f: ufl.core.expr.Expr) -> ufl.core.expr.Expr:
        
        if self.tangent is None:
            raise RuntimeError("Tangents not available; call make_mesh() first")
        return ufl.dot(ufl.grad(f), self.tangent)

    def dds_i(self, f: ufl.core.expr.Expr, i: int) -> ufl.core.expr.Expr:
        
        if i < 0 or i >= len(self.edge_list):
            raise IndexError(f"Edge index {i} out of range")
        tangent = self.edges[self.edge_list[i]]["tangent"]
        return ufl.dot(ufl.grad(f), ufl.as_vector(tangent))

    def get_num_inlets_outlets(self) -> Tuple[int, int]:
        
        num_inlets, num_outlets = 0, 0
        for e in self.edge_list:
            sub = self.edges[e].get("submesh", None)
            if sub is None:
                continue
            vt: dolfinx.mesh.MeshTags = sub.vertex_tags
            vals = np.asarray(vt.values, dtype=np.int32)
            num_inlets += int(np.sum(vals == BOUN_IN))
            num_outlets += int(np.sum(vals == BOUN_OUT))
        return num_inlets, num_outlets

    def edge_attribute_array(self, attr: str, dtype=np.float64) -> np.ndarray:
        
        if self.edge_list is None:
            self.edge_list = list(self.edges())
        vals = []
        for e in self.edge_list:
            if attr not in self.edges[e]:
                raise KeyError(f"Edge {e} is missing attribute '{attr}'")
            vals.append(self.edges[e][attr])
        return np.asarray(vals, dtype=dtype)

    def edge_attribute_callable(
        self,
        attr: str,
        *,
        default: float | None = None,
        padding: float = 1e-12,
        dtype=np.float64,
    ):
        
        if self.mesh is None or self.edge_tags is None:
            raise RuntimeError("Call make_mesh() before building attribute callables")

        edge_vals = self.edge_attribute_array(attr, dtype=dtype)

        tdim = self.mesh.topology.dim
        imap = self.mesh.topology.index_map(tdim)
        num_cells = imap.size_local + imap.num_ghosts

        
        
        
        
        
        
        cell_to_edge = np.full(num_cells, -1, dtype=np.int32)
        try:
            orig = np.asarray(self.mesh.topology.original_cell_index, dtype=np.int64)
            cell_edge_ids = getattr(self, "_cell_edge_ids_global")
            if orig.shape[0] == num_cells:
                cell_to_edge[:] = np.asarray(cell_edge_ids, dtype=np.int32)[orig]
            else:
                
                orig_owned = np.asarray(orig[: imap.size_local], dtype=np.int64)
                cell_to_edge[: imap.size_local] = np.asarray(cell_edge_ids, dtype=np.int32)[
                    orig_owned
                ]
                
                cell_to_edge[self.edge_tags.indices] = self.edge_tags.values
        except Exception:
            
            cell_to_edge[self.edge_tags.indices] = self.edge_tags.values

        bb_tree = dolfinx.geometry.bb_tree(self.mesh, tdim, padding=padding)

        def _call(points):
            pts = np.asarray(points, dtype=np.float64)
            if pts.ndim == 1:
                pts = pts.reshape(1, -1)
            if pts.shape[0] == 3 and pts.shape[1] != 3:
                pts = pts.T
            if pts.shape[1] != 3:
                raise ValueError("Points must have shape (n, 3) or (3, n)")

            
            candidates = dolfinx.geometry.compute_collisions_points(bb_tree, pts)
            colliding = dolfinx.geometry.compute_colliding_cells(self.mesh, candidates, pts)

            out = np.empty(pts.shape[0], dtype=dtype)
            for i in range(pts.shape[0]):
                cell_candidates = colliding.links(i)
                if len(cell_candidates) == 0:
                    if default is None:
                        raise RuntimeError(f"Could not locate point {pts[i]} in graph mesh")
                    out[i] = default
                    continue
                chosen: int | None = None
                for c in cell_candidates:
                    cc = int(c)
                    if 0 <= cc < num_cells and int(cell_to_edge[cc]) >= 0:
                        chosen = cc
                        break
                if chosen is None:
                    if default is None:
                        raise RuntimeError(
                            f"Could not map point {pts[i]} to an edge id (candidates={list(cell_candidates)})"
                        )
                    out[i] = default
                    continue
                out[i] = edge_vals[int(cell_to_edge[chosen])]

            return out

        return _call


def copy_from_nx_graph(G_nx: nx.Graph) -> FenicsGraph:
    G = FenicsGraph()
    G.graph.update(G_nx.graph)

    
    G.add_nodes_from((n, d.copy()) for n, d in G_nx.nodes(data=True))

    
    for u, v, d in G_nx.edges(data=True):
        G.add_edge(u, v)
        
        for k, val in d.items():
            G.edges[u, v][k] = val

    return G


def nxgraph_attribute_to_dolfinx(G: FenicsGraph, attr: str) -> dolfinx.fem.Function:
    
    if G.mesh is None or G.edge_tags is None:
        raise RuntimeError("Call G.make_mesh() before converting attributes")

    
    edge_values = []
    for e in G.edge_list:
        if attr not in G.edges[e]:
            raise KeyError(f"Edge {e} is missing attribute '{attr}'")
        edge_values.append(float(G.edges[e][attr]))

    V = dolfinx.fem.functionspace(G.mesh, ("DG", 0))
    f = dolfinx.fem.Function(V)

    tdim = G.mesh.topology.dim
    num_cells = G.mesh.topology.index_map(tdim).size_local
    dm = V.dofmap.list[:num_cells]

    cell_edge_ids = np.asarray(G.edge_tags.values[:num_cells], dtype=np.int32)

    arr = f.x.array
    for c in range(num_cells):
        dof = int(dm[c, 0])
        arr[dof] = edge_values[int(cell_edge_ids[c])]
    f.x.scatter_forward()

    return f
