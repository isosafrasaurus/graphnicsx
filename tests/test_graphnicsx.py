from __future__ import annotations

import ast
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pytest


def _find_project_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / "graphnicsx" / "__init__.py").exists():
            return p
    return start.parent


PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


dolfinx = pytest.importorskip("dolfinx")
ufl = pytest.importorskip("ufl")
basix = pytest.importorskip("basix")
mpi4py = pytest.importorskip("mpi4py")

from mpi4py import MPI  
from dolfinx import fem, mesh  


import graphnicsx  
from graphnicsx import (  
    BIF_IN,
    BIF_OUT,
    BOUN_IN,
    BOUN_OUT,
    Y_bifurcation,
    line_graph,
)
from graphnicsx.plot import (  
    TubeFile,
    _point_values_on_vertices,
    _radius_values_on_vertices,
    _vertex_incident_owned_cell,
)


def rprint(msg: str, comm: MPI.Intracomm | None = None) -> None:
    
    if comm is None:
        comm = MPI.COMM_WORLD
    prefix = f"[rank {comm.rank}/{comm.size}]"
    print(f"{prefix} {msg}", flush=True)


def assert_close(name: str, got: float, expected: float, tol: float) -> None:
    rprint(f"{name}: got={got:.16e}, expected={expected:.16e}, |diff|={abs(got-expected):.3e}, tol={tol:.3e}")
    assert abs(got - expected) <= tol


def find_vertex_by_coord(msh: mesh.Mesh, pt: np.ndarray, tol: float = 1e-12) -> int:
    X = np.asarray(msh.geometry.x, dtype=np.float64)
    pt = np.asarray(pt, dtype=np.float64).reshape(1, 3)
    d = np.linalg.norm(X - pt, axis=1)
    hits = np.where(d < tol)[0]
    if hits.size == 0:
        raise AssertionError(f"Could not find a vertex at {pt.flatten().tolist()} (min dist {d.min():.3e})")
    return int(hits[0])


def dg0_cell_values(f_dg0: fem.Function) -> np.ndarray:
    V = f_dg0.function_space
    msh = V.mesh
    tdim = msh.topology.dim
    num_cells = msh.topology.index_map(tdim).size_local
    dm = V.dofmap.list[:num_cells]
    vals = np.empty(num_cells, dtype=np.float64)
    for c in range(num_cells):
        dof = int(dm[c, 0])
        vals[c] = float(f_dg0.x.array[dof])
    return vals


def test_make_mesh_edge_tags_y_bifurcation():
    rprint("=== test_make_mesh_edge_tags_y_bifurcation ===", MPI.COMM_SELF)

    G = Y_bifurcation(dim=3, refine=0)  
    n = 2
    G.make_mesh(n=n, comm=MPI.COMM_SELF)

    assert G.mesh is not None
    assert G.edge_tags is not None

    rprint(f"Edges: {list(G.edges())}", MPI.COMM_SELF)
    rprint(f"edge_list (tag order): {G.edge_list}", MPI.COMM_SELF)

    assert G.mesh.topology.dim == 1
    assert G.mesh.geometry.dim == 3

    num_edges = len(G.edge_list)
    nseg = 2**n

    tdim = G.mesh.topology.dim
    num_cells = G.mesh.topology.index_map(tdim).size_local
    expected_cells = num_edges * nseg

    rprint(f"num_edges={num_edges}, n={n}, nseg={nseg}", MPI.COMM_SELF)
    rprint(f"num_cells={num_cells}, expected={expected_cells}", MPI.COMM_SELF)

    assert num_cells == expected_cells

    tags = np.asarray(G.edge_tags.values[:num_cells], dtype=np.int32)
    rprint(f"edge_tags.values range: [{tags.min()}, {tags.max()}]", MPI.COMM_SELF)
    assert tags.min() >= 0
    assert tags.max() < num_edges

    for edge_id in range(num_edges):
        cells = np.asarray(G.edge_tags.find(edge_id), dtype=np.int32)
        rprint(f"edge_id={edge_id}, num_cells_on_edge={cells.size} (expected {nseg})", MPI.COMM_SELF)
        assert cells.size == nseg


def test_dds_patch_on_straight_line():
    rprint("=== test_dds_patch_on_straight_line ===", MPI.COMM_SELF)

    
    L = 2.0
    G = line_graph(n=2, dim=3, dx=L, refine=0)  
    G.make_mesh(n=3, comm=MPI.COMM_SELF)  

    V = fem.functionspace(G.mesh, ("CG", 1))
    f = fem.Function(V, name="xcoord")
    f.interpolate(lambda x: x[0])  

    dxm = ufl.Measure("dx", domain=G.mesh)
    err2 = fem.assemble_scalar(fem.form((G.dds(f) - 1.0) ** 2 * dxm))

    rprint(f"L2 error^2 of dds(x) - 1: {err2:.3e}", MPI.COMM_SELF)
    assert err2 < 1e-24


def test_make_submeshes_and_endpoint_tags_y():
    rprint("=== test_make_submeshes_and_endpoint_tags_y ===", MPI.COMM_SELF)

    G = Y_bifurcation(dim=3, refine=0)
    G.make_mesh(n=2, comm=MPI.COMM_SELF)
    G.make_submeshes()

    assert hasattr(G, "edge_submeshes")
    assert len(G.edge_submeshes) == len(G.edge_list)

    n_in, n_out = G.get_num_inlets_outlets()
    rprint(f"Computed inlets/outlets: inlets={n_in}, outlets={n_out}", MPI.COMM_SELF)
    assert (n_in, n_out) == (1, 2)

    
    expected = {
        (0, 1): {BOUN_IN, BIF_IN},
        (1, 2): {BIF_OUT, BOUN_OUT},
        (1, 3): {BIF_OUT, BOUN_OUT},
    }

    for e in G.edge_list:
        sub = G.edge_submeshes[e]
        rprint(f"Edge {e}: submesh exists? {sub is not None}", MPI.COMM_SELF)
        assert sub is not None

        vt = sub.vertex_tags
        vals = set(np.asarray(vt.values, dtype=np.int32).tolist())
        rprint(f"  vertex_tags.values={list(np.asarray(vt.values, dtype=np.int32))}", MPI.COMM_SELF)

        exp = expected.get(e, None)
        if exp is not None:
            rprint(f"  expected to contain tags: {sorted(exp)}", MPI.COMM_SELF)
            
            assert exp.issubset(vals)


def test_edge_attribute_callable_radius_lookup():
    rprint("=== test_edge_attribute_callable_radius_lookup ===", MPI.COMM_SELF)

    G = Y_bifurcation(dim=3, refine=0)

    
    radii = {(0, 1): 0.01, (1, 2): 0.02, (1, 3): 0.03}
    for e in G.edges():
        G.edges[e]["radius"] = float(radii[e])

    G.make_mesh(n=2, comm=MPI.COMM_SELF)

    rcall = G.edge_attribute_callable("radius", default=None)

    
    for (u, v) in G.edge_list:
        p0 = np.asarray(G.nodes[u]["pos"], dtype=float)
        p1 = np.asarray(G.nodes[v]["pos"], dtype=float)
        if p0.shape[0] == 2:
            p0 = np.array([p0[0], p0[1], 0.0])
        if p1.shape[0] == 2:
            p1 = np.array([p1[0], p1[1], 0.0])

        t = 0.37
        p = (1 - t) * p0 + t * p1

        got_row = float(rcall(p.reshape(1, 3))[0])
        got_col = float(rcall(p.reshape(3, 1))[0])
        exp = float(radii[(u, v)])

        rprint(f"Edge {(u,v)} point {p.tolist()} -> radius(row)={got_row}, radius(col)={got_col}, expected={exp}", MPI.COMM_SELF)
        assert abs(got_row - exp) < 1e-12
        assert abs(got_col - exp) < 1e-12


def test_plot_helpers_point_values_and_radius_averaging():
    rprint("=== test_plot_helpers_point_values_and_radius_averaging ===", MPI.COMM_SELF)

    G = line_graph(n=2, dim=3, dx=1.0, refine=0)
    G.make_mesh(n=3, comm=MPI.COMM_SELF)

    V = fem.functionspace(G.mesh, ("CG", 1))
    f = fem.Function(V, name="xcoord")
    f.interpolate(lambda x: x[0])

    msh = G.mesh
    tdim = msh.topology.dim
    num_cells = msh.topology.index_map(tdim).size_local
    owned_cell_mask = np.ones(num_cells, dtype=bool)
    cell_of_vertex = _vertex_incident_owned_cell(msh, owned_cell_mask=owned_cell_mask)

    num_vertices = msh.topology.index_map(0).size_local + msh.topology.index_map(0).num_ghosts
    used_vertices = np.arange(num_vertices, dtype=np.int32)
    X = msh.geometry.x[used_vertices]

    vals = _point_values_on_vertices(f, X, cell_of_vertex[used_vertices])
    max_err = float(np.nanmax(np.abs(vals - X[:, 0])))

    rprint(f"Max |f(vertex)-x| on line graph: {max_err:.3e}", MPI.COMM_SELF)
    assert max_err < 1e-12

    Gy = Y_bifurcation(dim=3, refine=0)
    
    r01, r12, r13 = 0.01, 0.02, 0.03
    Gy.edges[(0, 1)]["radius"] = r01
    Gy.edges[(1, 2)]["radius"] = r12
    Gy.edges[(1, 3)]["radius"] = r13
    Gy.make_mesh(n=2, comm=MPI.COMM_SELF)

    mshy = Gy.mesh
    assert mshy is not None
    tdim = mshy.topology.dim
    num_cells = mshy.topology.index_map(tdim).size_local
    owned_cell_mask = np.ones(num_cells, dtype=bool)
    cell_of_vertex = _vertex_incident_owned_cell(mshy, owned_cell_mask=owned_cell_mask)

    
    cell_to_edge = np.full(num_cells, -1, dtype=np.int32)
    assert Gy.edge_tags is not None
    cell_to_edge[Gy.edge_tags.indices] = Gy.edge_tags.values

    num_vertices = mshy.topology.index_map(0).size_local + mshy.topology.index_map(0).num_ghosts
    used_vertices = np.arange(num_vertices, dtype=np.int32)
    rvals = _radius_values_on_vertices(Gy, used_vertices, cell_of_vertex, cell_to_edge)

    
    junction_pt = np.array([0.0, 0.5, 0.0], dtype=np.float64)
    vj = find_vertex_by_coord(mshy, junction_pt, tol=1e-12)
    got = float(rvals[vj])
    expected = float(np.mean([r01, r12, r13]))

    rprint(f"Junction vertex index={vj}, coord={mshy.geometry.x[vj].tolist()}", MPI.COMM_SELF)
    assert_close("radius@junction", got, expected, tol=1e-12)

    
    node0_pt = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    v0 = find_vertex_by_coord(mshy, node0_pt, tol=1e-12)
    got0 = float(rvals[v0])
    rprint(f"Boundary vertex node0 index={v0}, coord={mshy.geometry.x[v0].tolist()}", MPI.COMM_SELF)
    assert_close("radius@node0", got0, r01, tol=1e-12)


def test_tubefile_vtp_roundtrip(tmp_path: Path):
    vtk = pytest.importorskip("vtk")

    rprint("=== test_tubefile_vtp_roundtrip ===", MPI.COMM_SELF)
    rprint("VTK is available; running TubeFile VTP roundtrip test.", MPI.COMM_SELF)

    G = line_graph(n=2, dim=3, dx=1.0, refine=0)
    for e in G.edges():
        G.edges[e]["radius"] = 0.02
    G.make_mesh(n=2, comm=MPI.COMM_SELF)

    V = fem.functionspace(G.mesh, ("CG", 1))
    f = fem.Function(V, name="xcoord")
    f.interpolate(lambda x: x[0])

    out_pvd = tmp_path / "out.pvd"
    tube = TubeFile(G, str(out_pvd))
    tube << (f, 0)

    vtp = tmp_path / "out000000.vtp"
    rprint(f"Expecting VTP file at: {vtp}", MPI.COMM_SELF)
    assert vtp.exists()

    from vtk import vtkXMLPolyDataReader

    reader = vtkXMLPolyDataReader()
    reader.SetFileName(str(vtp))
    reader.Update()
    poly = reader.GetOutput()

    pd = poly.GetPointData()
    arr_f = pd.GetArray("xcoord")
    arr_r = pd.GetArray("radius")
    rprint(f"Point-data arrays: {[pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]}", MPI.COMM_SELF)

    assert arr_f is not None
    assert arr_r is not None

    npts = poly.GetNumberOfPoints()
    rprint(f"Number of points in VTP: {npts}", MPI.COMM_SELF)
    assert npts > 0

    
    for i in range(min(5, npts)):
        x, y, z = poly.GetPoint(i)
        fx = float(arr_f.GetTuple1(i))
        rr = float(arr_r.GetTuple1(i))
        rprint(f"  pt[{i}]=({x:.3f},{y:.3f},{z:.3f})  xcoord={fx:.6f} radius={rr:.6f}", MPI.COMM_SELF)
        assert abs(fx - x) < 1e-10
        assert abs(rr - 0.02) < 1e-12


@dataclass
class LoadedMain2:
    HydraulicNetwork: Any
    locate_boundary_dofs: Any
    locate_boundary_dofs_mixed_subspace: Any


def _load_main2_until_hydraulicnetwork(main2_path: Path) -> LoadedMain2:
    
    src = main2_path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(main2_path))

    kept = []
    for node in tree.body:
        
        
        if isinstance(node, ast.Import):
            if any(alias.name == "fenicsx_ii" for alias in node.names):
                continue
        if isinstance(node, ast.ImportFrom):
            if node.module == "fenicsx_ii":
                continue

        kept.append(node)
        if isinstance(node, ast.ClassDef) and node.name == "HydraulicNetwork":
            break

    mod = ast.Module(body=kept, type_ignores=[])
    code = compile(mod, filename=str(main2_path), mode="exec")
    ns: Dict[str, Any] = {}
    exec(code, ns)

    return LoadedMain2(
        HydraulicNetwork=ns["HydraulicNetwork"],
        locate_boundary_dofs=ns.get("locate_boundary_dofs"),
        locate_boundary_dofs_mixed_subspace=ns.get("locate_boundary_dofs_mixed_subspace"),
    )


@pytest.fixture(scope="session")
def main2_defs() -> LoadedMain2:
    main2_path = PROJECT_ROOT / "main2.py"
    if not main2_path.exists():
        pytest.skip(f"Could not find main2.py at {main2_path}")
    rprint(f"Loading HydraulicNetwork from {main2_path}", MPI.COMM_SELF)
    return _load_main2_until_hydraulicnetwork(main2_path)


def _solve_hydraulic(model) -> Tuple[fem.Function, fem.Function]:
    
    try:
        rprint("Attempting HydraulicNetwork.solve() (uses MUMPS in the example).", MPI.COMM_SELF)
        return model.solve()
    except Exception as e:
        rprint(f"HydraulicNetwork.solve() failed: {type(e).__name__}: {e}", MPI.COMM_SELF)
        rprint("Falling back to a PETSc LU solve without forcing MUMPS.", MPI.COMM_SELF)

        a, L = model.forms()
        bcs = model.bcs()
        uh = fem.Function(model.W, name="qp")

        opts = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            
        }
        problem = fem.petsc.LinearProblem(
            a,
            L,
            bcs=bcs,
            u=uh,
            petsc_options=opts,
            petsc_options_prefix="hydraulic_fallback_",
        )
        uh = problem.solve()

        qh_view = uh.sub(0)
        ph_view = uh.sub(1)
        Q, _ = model.W.sub(0).collapse()
        P, _ = model.W.sub(1).collapse()
        qh = fem.Function(Q, name="q")
        ph = fem.Function(P, name="p")
        qh.x.array[:] = qh_view.collapse().x.array
        ph.x.array[:] = ph_view.collapse().x.array
        return qh, ph


def test_hydraulic_patch_line_exact(main2_defs: LoadedMain2):
    rprint("=== test_hydraulic_patch_line_exact ===", MPI.COMM_SELF)

    HydraulicNetwork = main2_defs.HydraulicNetwork

    
    L = 2.0
    Res = 1.0

    G = line_graph(n=2, dim=3, dx=L, refine=0)
    G.make_mesh(n=3, comm=MPI.COMM_SELF)

    model = HydraulicNetwork(G, f=0.0, g=0.0, Res=Res, degree=1)
    
    model.p_bc.interpolate(lambda x: x[0] / L)

    qh, ph = _solve_hydraulic(model)

    
    q_cell = dg0_cell_values(qh)
    q_expected = -1.0 / (Res * L)
    q_mean = float(np.mean(q_cell))
    q_maxerr = float(np.max(np.abs(q_cell - q_expected)))

    rprint(f"q_expected={q_expected:.16e}, q_mean={q_mean:.16e}, max|q-q_expected|={q_maxerr:.3e}", MPI.COMM_SELF)
    assert abs(q_mean - q_expected) < 1e-10
    assert q_maxerr < 1e-10

    
    msh = G.mesh
    tdim = msh.topology.dim
    num_cells = msh.topology.index_map(tdim).size_local
    owned_cell_mask = np.ones(num_cells, dtype=bool)
    cell_of_vertex = _vertex_incident_owned_cell(msh, owned_cell_mask=owned_cell_mask)

    num_vertices = msh.topology.index_map(0).size_local
    used_vertices = np.arange(num_vertices, dtype=np.int32)
    X = msh.geometry.x[used_vertices]
    pvals = _point_values_on_vertices(ph, X, cell_of_vertex[used_vertices])

    p_expected = X[:, 0] / L
    p_maxerr = float(np.nanmax(np.abs(pvals - p_expected)))

    rprint(f"Max |p(vertex) - x/L|: {p_maxerr:.3e}", MPI.COMM_SELF)
    assert p_maxerr < 1e-10


def test_hydraulic_yjunction_kirchhoff(main2_defs: LoadedMain2):
    rprint("=== test_hydraulic_yjunction_kirchhoff ===", MPI.COMM_SELF)

    HydraulicNetwork = main2_defs.HydraulicNetwork

    
    G = Y_bifurcation(dim=3, refine=0)
    G.make_mesh(n=3, comm=MPI.COMM_SELF)

    model = HydraulicNetwork(G, f=0.0, g=0.0, Res=1.0, degree=1)
    model.p_bc.interpolate(lambda x: x[1])

    qh, ph = _solve_hydraulic(model)

    
    p0 = np.asarray(G.nodes[0]["pos"], dtype=float)
    p1 = np.asarray(G.nodes[1]["pos"], dtype=float)
    p2 = np.asarray(G.nodes[2]["pos"], dtype=float)
    p3 = np.asarray(G.nodes[3]["pos"], dtype=float)
    
    def _to3(p):
        p = np.asarray(p, dtype=float)
        if p.shape[0] == 2:
            return np.array([p[0], p[1], 0.0])
        return p

    p0, p1, p2, p3 = map(_to3, [p0, p1, p2, p3])

    L01 = float(np.linalg.norm(p1 - p0))
    L12 = float(np.linalg.norm(p2 - p1))
    L13 = float(np.linalg.norm(p3 - p1))
    p0v, p2v, p3v = 0.0, 1.0, 1.0

    P1 = (p0v / L01 + p2v / L12 + p3v / L13) / (1.0 / L01 + 1.0 / L12 + 1.0 / L13)
    rprint(f"Edge lengths: L01={L01:.6f}, L12={L12:.6f}, L13={L13:.6f}", MPI.COMM_SELF)
    rprint(f"Analytic junction pressure P1={P1:.12f}", MPI.COMM_SELF)

    
    junction_pt = _to3([0.0, 0.5, 0.0])
    vj = find_vertex_by_coord(G.mesh, junction_pt, tol=1e-12)
    
    msh = G.mesh
    tdim = msh.topology.dim
    num_cells = msh.topology.index_map(tdim).size_local
    owned_cell_mask = np.ones(num_cells, dtype=bool)
    cell_of_vertex = _vertex_incident_owned_cell(msh, owned_cell_mask=owned_cell_mask)
    p_at_junction = float(_point_values_on_vertices(ph, msh.geometry.x[[vj]], cell_of_vertex[[vj]])[0])

    assert_close("p_at_junction", p_at_junction, P1, tol=5e-10)

    
    q_cell = dg0_cell_values(qh)
    assert G.edge_tags is not None
    edge_flux: Dict[Tuple[int, int], float] = {}
    for edge_id, e in enumerate(G.edge_list):
        cells = np.asarray(G.edge_tags.find(edge_id), dtype=np.int32)
        vals = q_cell[cells]
        edge_flux[e] = float(np.mean(vals))
        rprint(f"Edge {e} (edge_id={edge_id}) q_mean={edge_flux[e]: .12f} over {len(vals)} cells", MPI.COMM_SELF)

    
    q01 = edge_flux[(0, 1)]
    q12 = edge_flux[(1, 2)]
    q13 = edge_flux[(1, 3)]
    kir = (-q01) + q12 + q13

    rprint(f"Kirchhoff check: (-q01)+q12+q13 = {kir:.3e}", MPI.COMM_SELF)
    assert abs(kir) < 1e-10


fenicsx_ii = None
try:
    import fenicsx_ii as _fenicsx_ii  
    fenicsx_ii = _fenicsx_ii
except Exception:
    fenicsx_ii = None


def locate_boundary_dofs(V) -> np.ndarray:
    
    msh = V.mesh
    tdim = msh.topology.dim
    if tdim == 1:
        entities = mesh.locate_entities_boundary(
            msh, 0, lambda x: np.full(x.shape[1], True, dtype=bool)
        )
        return fem.locate_dofs_topological(V, 0, entities)
    else:
        entities = mesh.locate_entities_boundary(
            msh, tdim - 1, lambda x: np.full(x.shape[1], True, dtype=bool)
        )
        return fem.locate_dofs_topological(V, tdim - 1, entities)


@pytest.mark.coupled
@pytest.mark.skipif(fenicsx_ii is None, reason="fenicsx_ii is not installed")
def test_coupled_3d1d_zero_solution_and_residual():
    rprint("=== test_coupled_3d1d_zero_solution_and_residual ===", MPI.COMM_SELF)
    from petsc4py import PETSc

    
    G = line_graph(n=2, dim=3, dx=1.0, refine=0)
    G.make_mesh(n=2, comm=MPI.COMM_SELF)
    mesh1d = G.mesh

    lo = np.array([-0.25, -0.25, -0.25], dtype=np.float64)
    hi = np.array([1.25, 0.25, 0.25], dtype=np.float64)
    mesh3d = mesh.create_box(mesh1d.comm, np.array([lo, hi], dtype=np.float64), [6, 6, 6], cell_type=mesh.CellType.tetrahedron)

    V3 = fem.functionspace(mesh3d, ("CG", 1))
    V1 = fem.functionspace(mesh1d, ("CG", 1))

    u3 = ufl.TrialFunction(V3)
    v3 = ufl.TestFunction(V3)
    u1 = ufl.TrialFunction(V1)
    v1 = ufl.TestFunction(V1)

    
    dofs3 = locate_boundary_dofs(V3)
    bc3 = fem.dirichletbc(PETSc.ScalarType(0.0), dofs3, V3)

    u_bc_1 = fem.Function(V1)
    u_bc_1.interpolate(lambda x: np.zeros(x.shape[1], dtype=PETSc.ScalarType))
    dofs1 = locate_boundary_dofs(V1)
    bc1 = fem.dirichletbc(u_bc_1, dofs1)

    beta = 1.0
    R = 0.1
    circle_degree = 4
    circum = 2.0 * np.pi

    circle_op = fenicsx_ii.Circle(mesh1d, radius=R, degree=circle_degree)
    Pi_u = fenicsx_ii.Average(u3, circle_op, V1)
    Pi_v = fenicsx_ii.Average(v3, circle_op, V1)

    dx3 = ufl.Measure("dx", domain=mesh3d)
    dx1 = ufl.Measure("dx", domain=mesh1d)

    a00 = ufl.inner(ufl.grad(u3), ufl.grad(v3)) * dx3 + circum * beta * ufl.inner(Pi_u, Pi_v) * dx1
    a01 = -beta * circum * ufl.inner(u1, Pi_v) * dx1
    a10 = -beta * ufl.inner(Pi_u, v1) * dx1
    a11 = ufl.inner(ufl.grad(u1), ufl.grad(v1)) * dx1 + beta * ufl.inner(u1, v1) * dx1

    zero1 = fem.Constant(mesh1d, PETSc.ScalarType(0.0))
    L0 = ufl.inner(zero1, Pi_v) * dx1
    L1 = ufl.inner(zero1, v1) * dx1

    rprint("Assembling coupled matrices...", MPI.COMM_SELF)
    A00 = fenicsx_ii.assemble_matrix(a00, bcs=[bc3])
    A01 = fenicsx_ii.assemble_matrix(a01, bcs=[bc3])
    A10 = fenicsx_ii.assemble_matrix(a10, bcs=[bc1])
    A11 = fenicsx_ii.assemble_matrix(a11, bcs=[bc1])

    b0 = fenicsx_ii.assemble_vector(L0, bcs=[bc3])
    b1 = fenicsx_ii.assemble_vector(L1, bcs=[bc1])

    A = PETSc.Mat().createNest([[A00, A01], [A10, A11]], comm=mesh1d.comm)
    A.assemble()
    b = PETSc.Vec().createNest([b0, b1], comm=mesh1d.comm)

    u3_h = fem.Function(V3, name="u3")
    u1_h = fem.Function(V1, name="u1")
    x_sol = PETSc.Vec().createNest([u3_h.x.petsc_vec, u1_h.x.petsc_vec], comm=mesh1d.comm)

    ksp = PETSc.KSP().create(mesh1d.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    ksp.setFromOptions()

    rprint("Solving coupled system...", MPI.COMM_SELF)
    ksp.solve(b, x_sol)

    u3_h.x.scatter_forward()
    u1_h.x.scatter_forward()

    max_u3 = float(np.max(np.abs(u3_h.x.array)))
    max_u1 = float(np.max(np.abs(u1_h.x.array)))
    rprint(f"Solution max norms: max|u3|={max_u3:.3e}, max|u1|={max_u1:.3e}", MPI.COMM_SELF)
    assert max_u3 < 1e-10
    assert max_u1 < 1e-10

    
    r = b.duplicate()
    A.mult(x_sol, r)
    r.axpy(-1.0, b)
    rnorm = float(r.norm())
    bnorm = float(b.norm())
    rel = rnorm / (bnorm + 1e-30)
    rprint(f"Residual norm: ||Ax-b||={rnorm:.3e}, ||b||={bnorm:.3e}, rel={rel:.3e}", MPI.COMM_SELF)
    assert rel < 1e-10


@pytest.mark.coupled
@pytest.mark.skipif(fenicsx_ii is None, reason="fenicsx_ii is not installed")
def test_coupled_beta_zero_decouples_and_1d_matches_linear():
    rprint("=== test_coupled_beta_zero_decouples_and_1d_matches_linear ===", MPI.COMM_SELF)
    from petsc4py import PETSc

    
    L = 1.0
    G = line_graph(n=2, dim=3, dx=L, refine=0)
    G.make_mesh(n=2, comm=MPI.COMM_SELF)
    mesh1d = G.mesh

    lo = np.array([-0.25, -0.25, -0.25], dtype=np.float64)
    hi = np.array([1.25, 0.25, 0.25], dtype=np.float64)
    mesh3d = mesh.create_box(mesh1d.comm, np.array([lo, hi], dtype=np.float64), [6, 6, 6], cell_type=mesh.CellType.tetrahedron)

    V3 = fem.functionspace(mesh3d, ("CG", 1))
    V1 = fem.functionspace(mesh1d, ("CG", 1))

    u3 = ufl.TrialFunction(V3)
    v3 = ufl.TestFunction(V3)
    u1 = ufl.TrialFunction(V1)
    v1 = ufl.TestFunction(V1)

    
    dofs3 = locate_boundary_dofs(V3)
    bc3 = fem.dirichletbc(PETSc.ScalarType(0.0), dofs3, V3)

    
    u_bc_1 = fem.Function(V1)
    u_bc_1.interpolate(lambda x: x[0] / L)
    dofs1 = locate_boundary_dofs(V1)
    bc1 = fem.dirichletbc(u_bc_1, dofs1)

    beta = fem.Constant(mesh1d, PETSc.ScalarType(0.0))
    R = 0.1
    circle_degree = 4
    circum = 2.0 * np.pi

    circle_op = fenicsx_ii.Circle(mesh1d, radius=R, degree=circle_degree)
    Pi_u = fenicsx_ii.Average(u3, circle_op, V1)
    Pi_v = fenicsx_ii.Average(v3, circle_op, V1)

    dx3 = ufl.Measure("dx", domain=mesh3d)
    dx1 = ufl.Measure("dx", domain=mesh1d)

    a00 = ufl.inner(ufl.grad(u3), ufl.grad(v3)) * dx3 + circum * beta * ufl.inner(Pi_u, Pi_v) * dx1
    a01 = -beta * circum * ufl.inner(u1, Pi_v) * dx1
    a10 = -beta * ufl.inner(Pi_u, v1) * dx1
    a11 = ufl.inner(ufl.grad(u1), ufl.grad(v1)) * dx1 + beta * ufl.inner(u1, v1) * dx1

    zero1 = fem.Constant(mesh1d, PETSc.ScalarType(0.0))
    L0 = ufl.inner(zero1, Pi_v) * dx1
    L1 = ufl.inner(zero1, v1) * dx1

    rprint("Assembling coupled matrices (beta=0)...", MPI.COMM_SELF)
    A00 = fenicsx_ii.assemble_matrix(a00, bcs=[bc3])
    A01 = fenicsx_ii.assemble_matrix(a01, bcs=[bc3])
    A10 = fenicsx_ii.assemble_matrix(a10, bcs=[bc1])
    A11 = fenicsx_ii.assemble_matrix(a11, bcs=[bc1])

    
    n01 = float(A01.norm())
    n10 = float(A10.norm())
    rprint(f"Off-diagonal norms: ||A01||={n01:.3e}, ||A10||={n10:.3e}", MPI.COMM_SELF)
    assert n01 < 1e-12
    assert n10 < 1e-12

    b0 = fenicsx_ii.assemble_vector(L0, bcs=[bc3])
    b1 = fenicsx_ii.assemble_vector(L1, bcs=[bc1])

    from petsc4py import PETSc as _PETSc

    A = _PETSc.Mat().createNest([[A00, A01], [A10, A11]], comm=mesh1d.comm)
    A.assemble()
    b = _PETSc.Vec().createNest([b0, b1], comm=mesh1d.comm)

    u3_h = fem.Function(V3, name="u3")
    u1_h = fem.Function(V1, name="u1")
    x_sol = _PETSc.Vec().createNest([u3_h.x.petsc_vec, u1_h.x.petsc_vec], comm=mesh1d.comm)

    ksp = _PETSc.KSP().create(mesh1d.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    ksp.setFromOptions()

    rprint("Solving coupled system (beta=0)...", MPI.COMM_SELF)
    ksp.solve(b, x_sol)

    u3_h.x.scatter_forward()
    u1_h.x.scatter_forward()

    
    max_u3 = float(np.max(np.abs(u3_h.x.array)))
    rprint(f"max|u3|={max_u3:.3e}", MPI.COMM_SELF)
    assert max_u3 < 1e-10

    
    msh1 = mesh1d
    tdim = msh1.topology.dim
    num_cells = msh1.topology.index_map(tdim).size_local
    owned_cell_mask = np.ones(num_cells, dtype=bool)
    cell_of_vertex = _vertex_incident_owned_cell(msh1, owned_cell_mask=owned_cell_mask)

    num_vertices = msh1.topology.index_map(0).size_local
    used_vertices = np.arange(num_vertices, dtype=np.int32)
    X = msh1.geometry.x[used_vertices]
    uvals = _point_values_on_vertices(u1_h, X, cell_of_vertex[used_vertices])
    uexp = X[:, 0] / L

    max_err = float(np.nanmax(np.abs(uvals - uexp)))
    rprint(f"Max |u1(vertex) - x/L| = {max_err:.3e}", MPI.COMM_SELF)
    assert max_err < 1e-10


@pytest.mark.coupled
@pytest.mark.skipif(fenicsx_ii is None, reason="fenicsx_ii is not installed")
@pytest.mark.skipif(os.getenv("GRAPHNICSX_TEST_SYMMETRY", "0") != "1", reason="Set GRAPHNICSX_TEST_SYMMETRY=1 to enable")
def test_coupled_symmetry_diagnostic_no_bcs():
    rprint("=== test_coupled_symmetry_diagnostic_no_bcs ===", MPI.COMM_SELF)
    rprint("This is a diagnostic: it may fail if the weak form is not intended to be symmetric.", MPI.COMM_SELF)

    from petsc4py import PETSc

    G = line_graph(n=2, dim=3, dx=1.0, refine=0)
    G.make_mesh(n=1, comm=MPI.COMM_SELF)
    mesh1d = G.mesh

    lo = np.array([-0.25, -0.25, -0.25], dtype=np.float64)
    hi = np.array([1.25, 0.25, 0.25], dtype=np.float64)
    mesh3d = mesh.create_box(mesh1d.comm, np.array([lo, hi], dtype=np.float64), [4, 4, 4], cell_type=mesh.CellType.tetrahedron)

    V3 = fem.functionspace(mesh3d, ("CG", 1))
    V1 = fem.functionspace(mesh1d, ("CG", 1))

    u3 = ufl.TrialFunction(V3)
    v3 = ufl.TestFunction(V3)
    u1 = ufl.TrialFunction(V1)
    v1 = ufl.TestFunction(V1)

    beta = 1.0
    R = 0.1
    circle_degree = 3
    circum = 2.0 * np.pi

    circle_op = fenicsx_ii.Circle(mesh1d, radius=R, degree=circle_degree)
    Pi_u = fenicsx_ii.Average(u3, circle_op, V1)
    Pi_v = fenicsx_ii.Average(v3, circle_op, V1)

    dx3 = ufl.Measure("dx", domain=mesh3d)
    dx1 = ufl.Measure("dx", domain=mesh1d)

    a00 = ufl.inner(ufl.grad(u3), ufl.grad(v3)) * dx3 + circum * beta * ufl.inner(Pi_u, Pi_v) * dx1
    a01 = -beta * circum * ufl.inner(u1, Pi_v) * dx1
    a10 = -beta * ufl.inner(Pi_u, v1) * dx1
    a11 = ufl.inner(ufl.grad(u1), ufl.grad(v1)) * dx1 + beta * ufl.inner(u1, v1) * dx1

    A00 = fenicsx_ii.assemble_matrix(a00, bcs=[])
    A01 = fenicsx_ii.assemble_matrix(a01, bcs=[])
    A10 = fenicsx_ii.assemble_matrix(a10, bcs=[])
    A11 = fenicsx_ii.assemble_matrix(a11, bcs=[])

    A = PETSc.Mat().createNest([[A00, A01], [A10, A11]], comm=mesh1d.comm)
    A.assemble()

    
    
    x3 = PETSc.Vec().createSeq(V3.dofmap.index_map.size_local)
    y3 = PETSc.Vec().createSeq(V3.dofmap.index_map.size_local)
    x1 = PETSc.Vec().createSeq(V1.dofmap.index_map.size_local)
    y1 = PETSc.Vec().createSeq(V1.dofmap.index_map.size_local)

    rng = np.random.default_rng(1234)
    x3.setArray(rng.standard_normal(x3.getSize()))
    y3.setArray(rng.standard_normal(y3.getSize()))
    x1.setArray(rng.standard_normal(x1.getSize()))
    y1.setArray(rng.standard_normal(y1.getSize()))

    x = PETSc.Vec().createNest([x3, x1], comm=mesh1d.comm)
    y = PETSc.Vec().createNest([y3, y1], comm=mesh1d.comm)

    Ax = y.duplicate()
    Ay = x.duplicate()
    A.mult(x, Ax)
    A.mult(y, Ay)

    xtAy = float(x.dot(Ay))
    ytAx = float(y.dot(Ax))
    diff = abs(xtAy - ytAx)
    denom = abs(xtAy) + abs(ytAx) + 1e-30
    rel = diff / denom

    rprint(f"Symmetry diagnostic: x^T A y = {xtAy:.6e}, y^T A x = {ytAx:.6e}, rel diff = {rel:.3e}", MPI.COMM_SELF)

    assert rel < 1e-10
