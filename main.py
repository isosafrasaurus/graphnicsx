import numpy as np
from mpi4py import MPI
from dolfinx import fem

from graphnicsx import Y_bifurcation, TubeFile


def y_bifurcation_tube():
    
    G = Y_bifurcation(dim=3)

    
    for e in G.edges():
        G.edges[e]["radius"] = 0.02

    
    G.make_mesh(n=1)
    G.make_submeshes()

    
    assert G.mesh.topology.dim == 1
    assert G.mesh.geometry.dim == 3
    assert hasattr(G, "edge_submeshes")
    assert len(G.edge_submeshes) == len(G.edges())

    
    
    V = fem.functionspace(G.mesh, ("CG", 1))
    f = fem.Function(V, name="xcoord")
    f.interpolate(lambda x: x[0])

    if MPI.COMM_WORLD.rank == 0:
        print("Writing TubeFile output: out.pvd")

    tube = TubeFile(G, "out.pvd")
    tube << (f, 0)


if __name__ == "__main__":
    y_bifurcation_tube()
