from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import networkx as nx
import numpy as np

import dolfinx
import dolfinx.fem
import dolfinx.geometry
import ufl

from .fenics_graph import FenicsGraph


def color_graph(G: nx.Graph) -> None:
    
    G_undir = nx.Graph(G)
    G_disconn = nx.Graph(G_undir)

    C = nx.adjacency_matrix(G_undir)
    num_vertex_conns = np.asarray(C.sum(axis=1)).flatten()
    bifurcation_points = np.where(num_vertex_conns > 2)[0].tolist()

    for b in bifurcation_points:
        for e in list(G_undir.edges(b)):
            
            v1, v2 = e
            other = v2 if v1 == b else v1

            new_bif_vertex = f"{b} {other}"
            G_disconn.add_node(new_bif_vertex)
            if "pos" in G_undir.nodes[b]:
                G_disconn.nodes[new_bif_vertex]["pos"] = G_undir.nodes[b]["pos"]

            G_disconn.add_edge(new_bif_vertex, other)
            
            if G_disconn.has_edge(v1, v2):
                G_disconn.remove_edge(v1, v2)

    
    subG = [G_disconn.subgraph(c).copy() for c in nx.connected_components(G_disconn)]

    n_branches = 0
    for sG in subG:
        C = nx.adjacency_matrix(sG)
        num_vertex_conns = np.asarray(C.sum(axis=1)).flatten()

        is_disconn_graph = (np.min(num_vertex_conns) > 0) and (np.max(num_vertex_conns) < 3)
        if is_disconn_graph:
            for e in sG.edges():
                v1 = str(e[0]).split(" ")[0]
                v2 = str(e[1]).split(" ")[0]
                orig_e1 = (int(v1), int(v2))
                orig_e2 = (int(v2), int(v1))
                if orig_e1 in G.edges:
                    G.edges[orig_e1]["color"] = n_branches
                elif orig_e2 in G.edges:
                    G.edges[orig_e2]["color"] = n_branches
            n_branches += 1

    
    for e in G.edges():
        if "color" not in G.edges[e]:
            G.edges[e]["color"] = n_branches
            n_branches += 1


def plot_graph_color(G: nx.Graph, *, ax=None):
    import matplotlib.pyplot as plt

    pos = nx.get_node_attributes(G, "pos")
    colors = nx.get_edge_attributes(G, "color")

    if ax is None:
        fig, ax = plt.subplots()

    nx.draw_networkx_edge_labels(G, pos, colors, ax=ax)
    nx.draw_networkx(G, pos, ax=ax)
    return ax


def assign_radius_using_Murrays_law(
    G: nx.DiGraph,
    start_node: int,
    start_radius: float,
) -> FenicsGraph:
    
    G_bfs = nx.bfs_tree(G, source=start_node)
    for n in G_bfs.nodes():
        G_bfs.nodes[n]["pos"] = G.nodes[n]["pos"]

    
    for u, v in G_bfs.edges():
        p0 = np.asarray(G_bfs.nodes[u]["pos"], dtype=float)
        p1 = np.asarray(G_bfs.nodes[v]["pos"], dtype=float)
        G_bfs.edges[u, v]["length"] = float(np.linalg.norm(p1 - p0))

    out_edges = list(G_bfs.out_edges(start_node))
    if len(out_edges) != 1:
        raise ValueError("start node must have a single edge sprouting from it")

    
    for i, e in enumerate(G_bfs.edges()):
        u, v = e
        if i == 0:
            G_bfs.edges[e]["radius"] = float(start_radius)
            continue

        
        parent_e = list(G_bfs.in_edges(u))[0]
        radius_p = float(G_bfs.edges[parent_e]["radius"])

        
        
        descendants_v = nx.descendants(G_bfs, v)
        descendants_u = nx.descendants(G_bfs, u)

        def subtree_length(root: int, descendants: Iterable[int]) -> float:
            nodes = set(descendants) | {root}
            sub = G_bfs.subgraph(nodes)
            return float(sum(G_bfs.edges[edge]["length"] for edge in sub.edges()))

        len_v = subtree_length(v, descendants_v) + float(G_bfs.edges[e]["length"])
        len_u = subtree_length(u, descendants_u)

        if len_u <= 0.0:
            
            n_children = max(1, len(list(G_bfs.out_edges(u))))
            fraction = 1.0 / n_children
        else:
            fraction = len_v / len_u

        radius_d = (fraction ** (1.0 / 3.0)) * radius_p
        G_bfs.edges[e]["radius"] = float(radius_d)

    from .fenics_graph import copy_from_nx_graph

    return copy_from_nx_graph(G_bfs)


@dataclass
class DistFromSource:
    G: FenicsGraph
    source_node: int

    def __post_init__(self):
        if self.G.mesh is None or self.G.edge_tags is None:
            raise RuntimeError("Call G.make_mesh() before constructing DistFromSource")

        
        if len(nx.get_edge_attributes(self.G, "length")) == 0:
            self.G.compute_edge_lengths()

        
        Gu = nx.Graph()
        for u, v in self.G.edges():
            Gu.add_edge(u, v, length=float(self.G.edges[u, v]["length"]))
        for n in self.G.nodes():
            Gu.nodes[n]["pos"] = self.G.nodes[n]["pos"]

        dist_node: Dict[int, float] = nx.single_source_dijkstra_path_length(
            Gu, self.source_node, weight="length"
        )

        
        mesh = self.G.mesh
        tdim = mesh.topology.dim
        mesh.topology.create_connectivity(0, tdim)
        v_to_c = mesh.topology.connectivity(0, tdim)

        num_cells_local = mesh.topology.index_map(tdim).size_local
        cell_to_edge = np.full(num_cells_local, -1, dtype=np.int32)
        cell_to_edge[self.G.edge_tags.indices] = self.G.edge_tags.values

        
        num_vertices_local = mesh.topology.index_map(0).size_local
        x_vertices = mesh.geometry.x[:num_vertices_local]

        
        values = np.zeros(num_vertices_local, dtype=dolfinx.default_scalar_type)
        for vi in range(num_vertices_local):
            incident = v_to_c.links(vi)
            if len(incident) == 0:
                values[vi] = 0.0
                continue
            cell = int(incident[0])
            edge_id = int(cell_to_edge[cell])
            if edge_id < 0:
                raise RuntimeError("Failed to map cell to edge id")

            u, v = self.G.edge_list[edge_id]
            p_u = np.asarray(self.G.nodes[u]["pos"], dtype=float)
            p_v = np.asarray(self.G.nodes[v]["pos"], dtype=float)
            if p_u.shape[0] != 3:
                p_u = np.pad(p_u, (0, 3 - p_u.shape[0]))
            if p_v.shape[0] != 3:
                p_v = np.pad(p_v, (0, 3 - p_v.shape[0]))

            
            if u not in dist_node or v not in dist_node:
                raise ValueError("Graph appears disconnected from the source node")
            if dist_node[u] <= dist_node[v]:
                p0 = p_u
                d0 = dist_node[u]
            else:
                p0 = p_v
                d0 = dist_node[v]

            values[vi] = d0 + float(np.linalg.norm(x_vertices[vi] - p0))

        
        V = dolfinx.fem.functionspace(mesh, ("CG", 1))
        f = dolfinx.fem.Function(V)

        
        
        num_dofs_local = V.dofmap.index_map.size_local
        f.x.array[:num_dofs_local] = values[:num_dofs_local]
        f.x.scatter_forward()

        self.function = f

    def eval(self, points: np.ndarray, *, padding: float = 1.0e-10) -> np.ndarray:
        mesh = self.G.mesh
        assert mesh is not None
        points = np.asarray(points, dtype=np.float64)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        if points.shape[1] != 3:
            raise ValueError("Points must have shape (N, 3)")

        
        tdim = mesh.topology.dim
        bb = dolfinx.geometry.bb_tree(mesh, tdim)
        collisions = dolfinx.geometry.compute_collisions_points(bb, points)
        cells = dolfinx.geometry.compute_colliding_cells(mesh, collisions, points)

        cell_ids = np.full(points.shape[0], -1, dtype=np.int32)
        for i in range(points.shape[0]):
            if len(cells.links(i)) > 0:
                cell_ids[i] = int(cells.links(i)[0])

        vals = self.function.eval(points, cell_ids)
        
        return vals.reshape(points.shape[0], -1)[:, 0]
