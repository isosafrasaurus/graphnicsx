from __future__ import annotations

import fenicsx_ii
import networkx as nx
import numpy as np
import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
from dolfinx import fem, mesh, io
from petsc4py import PETSc

from graphnicsx import Y_bifurcation, TubeFile


def _cell_name(msh: mesh.Mesh) -> str:
    c = msh.ufl_cell()
    if hasattr(c, "cellname"):
        return c.cellname()
    if hasattr(c, "_cellname"):
        return c._cellname
    if hasattr(c, "name"):
        return c.name
    s = str(c)
    return s.split("(")[-1].split(")")[0].strip("'\"")


def locate_boundary_dofs(V: fem.FunctionSpace) -> np.ndarray:
    
    msh = V.mesh
    tdim = msh.topology.dim
    if tdim == 1:
        entities = mesh.locate_entities_boundary(
            msh, 0, lambda x: np.full(x.shape[1], True, dtype=bool)
        )
        dofs = fem.locate_dofs_topological(V, 0, entities)
    else:
        entities = mesh.locate_entities_boundary(
            msh, tdim - 1, lambda x: np.full(x.shape[1], True, dtype=bool)
        )
        dofs = fem.locate_dofs_topological(V, tdim - 1, entities)
    return dofs


def locate_boundary_dofs_mixed_subspace(
        W_sub: fem.FunctionSpace, V_collapsed: fem.FunctionSpace
) -> np.ndarray:
    
    msh = W_sub.mesh
    tdim = msh.topology.dim
    if tdim == 1:
        entities = mesh.locate_entities_boundary(
            msh, 0, lambda x: np.full(x.shape[1], True, dtype=bool)
        )
        dofs = fem.locate_dofs_topological((W_sub, V_collapsed), 0, entities)
    else:
        entities = mesh.locate_entities_boundary(
            msh, tdim - 1, lambda x: np.full(x.shape[1], True, dtype=bool)
        )
        dofs = fem.locate_dofs_topological((W_sub, V_collapsed), tdim - 1, entities)
    return dofs


class HydraulicNetwork:

    def __init__(
            self,
            G,
            f: fem.Constant | float = 0.0,
            g: fem.Constant | float = 0.0,
            p_bc: fem.Function | None = None,
            Res: fem.Constant | float = 1.0,
            degree: int = 1,
    ):
        self.G = G

        self.f = f if isinstance(f, fem.Constant) else fem.Constant(G.mesh, PETSc.ScalarType(f))
        self.g = g if isinstance(g, fem.Constant) else fem.Constant(G.mesh, PETSc.ScalarType(g))
        self.Res = Res if isinstance(Res, fem.Constant) else fem.Constant(G.mesh, PETSc.ScalarType(Res))

        
        
        q_deg = max(degree - 1, 0)
        Q_el = basix_element("DG", _cell_name(G.mesh), q_deg)
        P_el = basix_element("Lagrange", _cell_name(G.mesh), degree)
        W_el = basix_mixed_element([Q_el, P_el])
        self.W = fem.functionspace(G.mesh, W_el)

        
        q, p = ufl.TrialFunctions(self.W)
        v, phi = ufl.TestFunctions(self.W)
        self.q, self.p, self.v, self.phi = q, p, v, phi

        
        if p_bc is None:
            Vp, _ = self.W.sub(1).collapse()
            p_bc = fem.Function(Vp)
            p_bc.interpolate(lambda x: np.zeros(x.shape[1], dtype=PETSc.ScalarType))
        self.p_bc = p_bc

    def forms(self):
        G = self.G
        q, p, v, phi = self.q, self.p, self.v, self.phi

        dx = ufl.Measure("dx", domain=G.mesh)

        a = (
                self.Res * ufl.inner(q, v) * dx
                + ufl.inner(G.dds(p), v) * dx
                - ufl.inner(G.dds(phi), q) * dx
        )
        L = self.g * v * dx + self.f * phi * dx
        return a, L

    def bcs(self):
        
        Vp, _ = self.W.sub(1).collapse()
        dofs = locate_boundary_dofs_mixed_subspace(self.W.sub(1), Vp)
        bc = fem.dirichletbc(self.p_bc, dofs, self.W.sub(1))
        return [bc]

    def solve(self):
        a, L = self.forms()
        bcs = self.bcs()

        uh = fem.Function(self.W, name="qp")

        opts = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }
        problem = fem.petsc.LinearProblem(
            a,
            L,
            bcs=bcs,
            u=uh,
            petsc_options=opts,
            petsc_options_prefix="hydraulic_",
        )
        uh = problem.solve()

        
        qh_view = uh.sub(0)
        ph_view = uh.sub(1)
        Q, _ = self.W.sub(0).collapse()
        P, _ = self.W.sub(1).collapse()
        qh = fem.Function(Q, name="q")
        ph = fem.Function(P, name="p")
        qh.x.array[:] = qh_view.collapse().x.array
        ph.x.array[:] = ph_view.collapse().x.array
        return qh, ph


G = Y_bifurcation(dim=3)
G.make_mesh(n=4)



for e in G.edges():
    G.edges[e]["radius"] = 0.05


Vp = fem.FunctionSpace(G.mesh, ("CG", 1))
p_bc = fem.Function(Vp)
p_bc.interpolate(lambda x: x[1])

model = HydraulicNetwork(G, p_bc=p_bc, degree=1)
q1d, p1d = model.solve()

if G.mesh.comm.rank == 0:
    print("Solved 1D hydraulic network.")


import pathlib as _pathlib

hydro_out = _pathlib.Path("plots/hydraulic1d")
if G.mesh.comm.rank == 0:
    hydro_out.mkdir(parents=True, exist_ok=True)
G.mesh.comm.barrier()
if G.mesh.comm.rank == 0:
    tube_h = TubeFile(G, str(hydro_out / "pressure1d_tube.pvd"))
    tube_h << (p1d, 0)

G = Y_bifurcation(dim=3)
G.make_mesh(n=4)
mesh1d = G.mesh

for e in G.edges():
    G.edges[e]["radius"] = 0.05


pos = nx.get_node_attributes(G, "pos")
node_coords = np.asarray(list(pos.values()), dtype=float)




R = 0.1
margin = R + 0.05
lo = np.min(node_coords, axis=0) - margin
hi = np.max(node_coords, axis=0) + margin

mesh3d = mesh.create_box(
    mesh1d.comm,
    np.array([lo, hi], dtype=np.float64),
    [40, 40, 40],
    cell_type=mesh.CellType.tetrahedron,
)


V3 = fem.FunctionSpace(mesh3d, ("CG", 1))
V1 = fem.FunctionSpace(mesh1d, ("CG", 1))

u3 = ufl.TrialFunction(V3)
v3 = ufl.TestFunction(V3)
u1 = ufl.TrialFunction(V1)
v1 = ufl.TestFunction(V1)



dofs3 = locate_boundary_dofs(V3)
bc3 = fem.dirichletbc(fem.Constant(mesh3d, PETSc.ScalarType(0.0)), dofs3, V3)


u_bc_1 = fem.Function(V1)
u_bc_1.interpolate(lambda x: x[1])
dofs1 = locate_boundary_dofs(V1)
bc1 = fem.dirichletbc(u_bc_1, dofs1)


beta = 1.0
circum = 2.0 * np.pi


circle_op = fenicsx_ii.Circle(mesh1d, radius=R, degree=10)


Pi_u = fenicsx_ii.Average(u3, circle_op, V1)
Pi_v = fenicsx_ii.Average(v3, circle_op, V1)

dx3 = ufl.Measure("dx", domain=mesh3d)
dx1 = ufl.Measure("dx", domain=mesh1d)


a00 = ufl.inner(ufl.grad(u3), ufl.grad(v3)) * dx3 + circum * beta * ufl.inner(Pi_u, Pi_v) * dx1
a01 = -beta * circum * ufl.inner(u1, Pi_v) * dx1
a10 = -beta * ufl.inner(Pi_u, v1) * dx1
a11 = ufl.inner(ufl.grad(u1), ufl.grad(v1)) * dx1 + beta * ufl.inner(u1, v1) * dx1


zero1 = fem.Constant(mesh1d, PETSc.ScalarType(0.0))
zero3 = fem.Constant(mesh3d, PETSc.ScalarType(0.0))
L0 = ufl.inner(zero1, Pi_v) * dx1
L1 = ufl.inner(zero1, v1) * dx1


A00 = fenicsx_ii.assemble_matrix(a00, bcs=[bc3])
A01 = fenicsx_ii.assemble_matrix(a01, bcs=[bc3])  
A10 = fenicsx_ii.assemble_matrix(a10, bcs=[bc1])  
A11 = fenicsx_ii.assemble_matrix(a11, bcs=[bc1])

b0 = fenicsx_ii.assemble_vector(L0, bcs=[bc3])
b1 = fenicsx_ii.assemble_vector(L1, bcs=[bc1])


A = PETSc.Mat().createNest([[A00, A01], [A10, A11]], comm=mesh1d.comm)
A.assemble()

b = PETSc.Vec().createNest([b0, b1], comm=mesh1d.comm)

u3_h = fem.Function(V3, name="pressure3d")
u1_h = fem.Function(V1, name="pressure1d")
x_sol = PETSc.Vec().createNest([u3_h.x.petsc_vec, u1_h.x.petsc_vec], comm=mesh1d.comm)


ksp = PETSc.KSP().create(mesh1d.comm)
ksp.setOperators(A)
ksp.setType("preonly")
pc = ksp.getPC()
pc.setType("lu")
pc.setFactorSolverType("mumps")
ksp.setFromOptions()

ksp.solve(b, x_sol)


u3_h.x.scatter_forward()
u1_h.x.scatter_forward()

if mesh1d.comm.rank == 0:
    print("Solved coupled 3D–1D problem.")


import pathlib

outdir = pathlib.Path("plots/coupled1d3d-2")
if mesh1d.comm.rank == 0:
    outdir.mkdir(parents=True, exist_ok=True)
mesh1d.comm.barrier()



if mesh1d.comm.rank == 0:
    tube = TubeFile(G, str(outdir / "pressure1d_tube.pvd"))
    tube << (u1_h, 0)


with io.VTKFile(mesh3d.comm, str(outdir / "pressure3d.pvd"), "w") as vtk:
    vtk.write_mesh(mesh3d)
    vtk.write_function(u3_h, t=0.0)
