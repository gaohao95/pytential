import numpy as np
import numpy.linalg as la
import pyopencl as cl
import pyopencl.clmath  # noqa

from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import \
        InterpolatoryQuadratureSimplexGroupFactory

from pytential import bind, sym  # noqa

from mpi4py import MPI

# {{{ set some constants for use below

nelements = 20
bdry_quad_order = 4
mesh_order = bdry_quad_order
qbx_order = bdry_quad_order
bdry_ovsmp_quad_order = 4*bdry_quad_order
fmm_order = 10
k = 3

# }}}

# {{{ setup MPI

comm = MPI.COMM_WORLD
current_rank = comm.Get_rank()
total_rank = comm.Get_size()

# }}}


if current_rank == 0:
    import logging
    logging.basicConfig(level=logging.WARNING)  # INFO for more progress info

    cl_ctx = cl.create_some_context()
    queue = cl.CommandQueue(cl_ctx)

    from meshmode.mesh.generation import ellipse, make_curve_mesh
    from functools import partial

    if 0:
        mesh = make_curve_mesh(
                partial(ellipse, 1),
                np.linspace(0, 1, nelements+1),
                mesh_order)
    else:
        base_mesh = make_curve_mesh(
                partial(ellipse, 1),
                np.linspace(0, 1, nelements+1),
                mesh_order)

        from meshmode.mesh.processing import affine_map, merge_disjoint_meshes
        nx = 2
        ny = 2
        dx = 2 / nx
        meshes = [
                affine_map(
                    base_mesh,
                    A=np.diag([dx*0.25, dx*0.25]),
                    b=np.array([dx*(ix-nx/2), dx*(iy-ny/2)]))
                for ix in range(nx)
                for iy in range(ny)]

        mesh = merge_disjoint_meshes(meshes, single_group=True)

        if 0:
            from meshmode.mesh.visualization import draw_curve
            draw_curve(mesh)
            import matplotlib.pyplot as plt
            plt.show()

    pre_density_discr = Discretization(
            cl_ctx, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(bdry_quad_order))

    from pytential.qbx import QBXLayerPotentialSource
    qbx, _ = QBXLayerPotentialSource(
            pre_density_discr, fine_order=bdry_ovsmp_quad_order, qbx_order=qbx_order,
            fmm_order=fmm_order,
            fmm_backend="distributed"
            ).with_refinement()
    density_discr = qbx.density_discr

    # {{{ describe bvp

    from sumpy.kernel import HelmholtzKernel
    kernel = HelmholtzKernel(2)

    cse = sym.cse

    sigma_sym = sym.var("sigma")
    sqrt_w = sym.sqrt_jac_q_weight(2)
    inv_sqrt_w_sigma = cse(sigma_sym/sqrt_w)

    # Brakhage-Werner parameter
    alpha = 1j

    # -1 for interior Dirichlet
    # +1 for exterior Dirichlet
    loc_sign = +1

    bdry_op_sym = (-loc_sign*0.5*sigma_sym
            + sqrt_w*(
                alpha*sym.S(kernel, inv_sqrt_w_sigma, k=sym.var("k"))
                - sym.D(kernel, inv_sqrt_w_sigma, k=sym.var("k"))
                ))

    # }}}

    bound_op = bind(qbx, bdry_op_sym)

    # {{{ fix rhs and solve

    nodes = density_discr.nodes().with_queue(queue)
    k_vec = np.array([2, 1])
    k_vec = k * k_vec / la.norm(k_vec, 2)

    def u_incoming_func(x):
        return cl.clmath.exp(
                1j * (x[0] * k_vec[0] + x[1] * k_vec[1]))

    bc = -u_incoming_func(nodes)

    bvp_rhs = bind(qbx, sqrt_w*sym.var("bc"))(queue, bc=bc)

    from pytential.solve import gmres
    gmres_result = gmres(
            bound_op.scipy_op(queue, "sigma", dtype=np.complex128, k=k),
            bvp_rhs, tol=1e-8, progress=True,
            stall_iterations=0,
            hard_failure=True)

    # }}}

    # {{{ Terminate helper drivers

    flag = False
    flag = comm.bcast(flag, root=0)

    # }}}

else:
    from pytential.qbx.distributed import drive_dfmm
    while True:
        wrangler = None
        weights = None
        flag = drive_dfmm(wrangler, weights)
        if not flag:
            break
