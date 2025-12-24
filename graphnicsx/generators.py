from __future__ import annotations

import networkx as nx
import numpy as np

from .fenics_graph import FenicsGraph, copy_from_nx_graph


def line_graph(n: int, dim: int = 2, dx: float = 1.0, *, refine: int = 1) -> FenicsGraph:
    if n < 2:
        raise ValueError("Need at least 2 nodes for a line graph")
    if dim not in (1, 2, 3):
        raise ValueError("dim must be 1, 2 or 3")

    G = FenicsGraph()
    G.add_nodes_from(range(0, n))
    for i in range(0, n):
        G.nodes[i]["pos"] = [i * dx] + [0.0] * (dim - 1)

    for i in range(0, n - 1):
        G.add_edge(i, i + 1)

    G.make_mesh(n=refine)
    return G


def honeycomb(n: int, m: int, *, refine: int = 1) -> FenicsGraph:
    G0 = nx.hexagonal_lattice_graph(n, m)

    if len(nx.get_node_attributes(G0, "pos")) == 0:
        pos = {node: np.array(node, dtype=float) for node in G0.nodes}
        nx.set_node_attributes(G0, pos, "pos")

    G0 = nx.convert_node_labels_to_integers(G0)

    inlet = len(G0.nodes)
    G0.add_node(inlet)
    G0.nodes[inlet]["pos"] = np.array([0.0, -1.0])
    G0.add_edge(inlet, 0)

    pos = nx.get_node_attributes(G0, "pos")
    all_coords = np.asarray(list(pos.values()), dtype=float)
    if all_coords.shape[1] == 1:
        all_coords = np.hstack([all_coords, np.zeros((all_coords.shape[0], 1))])

    furthest_node_ix = int(np.argmax(np.linalg.norm(all_coords, axis=1)))
    coord_furthest = all_coords[furthest_node_ix, :]

    outlet = len(G0.nodes)
    G0.add_node(outlet)
    G0.nodes[outlet]["pos"] = coord_furthest + np.asarray([0.7, 1.0])
    G0.add_edge(furthest_node_ix, outlet)

    G = copy_from_nx_graph(G0)

    if (0, inlet) in G.edges():
        G.remove_edge(0, inlet)
        G.add_edge(inlet, 0)

    G.make_mesh(n=refine)
    return G


def Y_bifurcation(dim: int = 2, *, refine: int = 1) -> FenicsGraph:
    if dim not in (2, 3):
        raise ValueError("dim must be 2 or 3")

    G = FenicsGraph()
    G.add_nodes_from([0, 1, 2, 3])
    G.nodes[0]["pos"] = [0.0, 0.0] + [0.0] * (dim - 2)
    G.nodes[1]["pos"] = [0.0, 0.5] + [0.0] * (dim - 2)
    G.nodes[2]["pos"] = [-0.5, 1.0] + [0.0] * (dim - 2)
    G.nodes[3]["pos"] = [0.5, 1.0] + [0.0] * (dim - 2)

    G.add_edge(0, 1)
    G.add_edge(1, 2)
    G.add_edge(1, 3)

    G.make_mesh(n=refine)
    return G


def YY_bifurcation(dim: int = 2, *, refine: int = 1) -> FenicsGraph:
    if dim not in (2, 3):
        raise ValueError("dim must be 2 or 3")

    G = FenicsGraph()

    G.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7])
    G.nodes[0]["pos"] = [0.0, 0.0] + [0.0] * (dim - 2)
    G.nodes[1]["pos"] = [0.0, 0.5] + [0.0] * (dim - 2)
    G.nodes[2]["pos"] = [-0.5, 1.0] + [0.0] * (dim - 2)
    G.nodes[3]["pos"] = [0.5, 1.0] + [0.0] * (dim - 2)

    G.add_edge(0, 1)
    G.add_edge(1, 2)
    G.add_edge(1, 3)

    G.nodes[4]["pos"] = [-0.75, 1.5] + [0.0] * (dim - 2)
    G.nodes[5]["pos"] = [-0.25, 1.5] + [0.0] * (dim - 2)
    G.nodes[6]["pos"] = [0.25, 1.5] + [0.0] * (dim - 2)
    G.nodes[7]["pos"] = [0.75, 1.5] + [0.0] * (dim - 2)

    G.add_edge(2, 4)
    G.add_edge(2, 5)
    G.add_edge(3, 6)
    G.add_edge(3, 7)

    G.make_mesh(n=refine)
    return G
