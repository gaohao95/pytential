import pyopencl as cl
from pytential import bind, sym, norm
import numpy as np
from sympy.core.cache import clear_cache
from pytools.convergence import EOCRecorder
from mpi4py import MPI
from pytential.qbx.distributed import DistributedQBXLayerPotentialSource
from sumpy.kernel import LaplaceKernel
import matplotlib.pyplot as pt

comm = MPI.COMM_WORLD
current_rank = comm.Get_rank()
total_rank = comm.Get_size()

# prevent cache 'splosion
clear_cache()

# Setup PyOpenCL
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

if current_rank == 0:

    class GreenExpr(object):
        zero_op_name = "green"

        def get_zero_op(self, kernel, **knl_kwargs):

            u_sym = sym.var("u")
            dn_u_sym = sym.var("dn_u")

            return (
                sym.S(kernel, dn_u_sym, qbx_forced_limit=-1, **knl_kwargs)
                - sym.D(kernel, u_sym, qbx_forced_limit="avg", **knl_kwargs)
                - 0.5*u_sym)

        order_drop = 0

    def get_sphere_mesh(refinement_increment, target_order):
        from meshmode.mesh.generation import generate_icosphere
        mesh = generate_icosphere(1, target_order)
        from meshmode.mesh.refinement import Refiner

        refiner = Refiner(mesh)
        for i in range(refinement_increment):
            flags = np.ones(mesh.nelements, dtype=bool)
            refiner.refine(flags)
            mesh = refiner.get_current_mesh()

        return mesh

    class SphereGeometry(object):
        mesh_name = "sphere"
        dim = 3

        resolutions = [0, 1]

        def get_mesh(self, resolution, tgt_order):
            return get_sphere_mesh(resolution, tgt_order)

    expr = GreenExpr()
    geometry = SphereGeometry()

    target_order = 8
    k = 0
    qbx_order = 3
    fmm_order = 10
    resolutions = [0, 1]
    _expansion_stick_out_factor = 0.5
    visualize = False

    eoc_rec = EOCRecorder()

    for resolution in resolutions:
        mesh = geometry.get_mesh(resolution, target_order)
        if mesh is None:
            break

        d = mesh.ambient_dim

        lap_k_sym = LaplaceKernel(d)
        k_sym = lap_k_sym
        knl_kwargs = {}

        from meshmode.discretization import Discretization
        from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory

        pre_density_discr = Discretization(
            ctx, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(target_order))

        refiner_extra_kwargs = {}

        qbx, _ = DistributedQBXLayerPotentialSource(
            comm,
            pre_density_discr, 4 * target_order,
            qbx_order,
            fmm_order=fmm_order,
            _expansions_in_tree_have_extent=True,
            _expansion_stick_out_factor=_expansion_stick_out_factor
        ).with_refinement(**refiner_extra_kwargs)

        density_discr = qbx.density_discr

        # {{{ compute values of a solution to the PDE

        nodes_host = density_discr.nodes().get(queue)
        normal = bind(density_discr, sym.normal(d))(queue).as_vector(np.object)
        normal_host = [normal[j].get() for j in range(d)]

        center = np.array([3, 1, 2])[:d]
        diff = nodes_host - center[:, np.newaxis]
        dist_squared = np.sum(diff ** 2, axis=0)
        dist = np.sqrt(dist_squared)
        if d == 2:
            u = np.log(dist)
            grad_u = diff / dist_squared
        elif d == 3:
            u = 1 / dist
            grad_u = -diff / dist ** 3
        else:
            assert False

        dn_u = 0
        for i in range(d):
            dn_u = dn_u + normal_host[i] * grad_u[i]

        # }}}

        u_dev = cl.array.to_device(queue, u)
        dn_u_dev = cl.array.to_device(queue, dn_u)
        grad_u_dev = cl.array.to_device(queue, grad_u)

        key = (qbx_order, geometry.mesh_name, resolution,
               expr.zero_op_name)

        bound_op = bind(qbx, expr.get_zero_op(k_sym, **knl_kwargs))
        error = bound_op(
            queue, u=u_dev, dn_u=dn_u_dev, grad_u=grad_u_dev, k=k)
        if 0:
            pt.plot(error)
            pt.show()

        linf_error_norm = norm(density_discr, queue, error, p=np.inf)
        print("--->", key, linf_error_norm)

        eoc_rec.add_data_point(qbx.h_max, linf_error_norm)

        if visualize:
            from meshmode.discretization.visualization import make_visualizer

            bdry_vis = make_visualizer(queue, density_discr, target_order)

            bdry_normals = bind(density_discr, sym.normal(mesh.ambient_dim))(queue) \
                .as_vector(dtype=object)

            bdry_vis.write_vtk_file("source-%s.vtu" % resolution, [
                ("u", u_dev),
                ("bdry_normals", bdry_normals),
                ("error", error),
            ])

    print(eoc_rec)
    tgt_order = qbx_order - expr.order_drop
    assert eoc_rec.order_estimate() > tgt_order - 1.6

else:
    while True:
        lp_source = DistributedQBXLayerPotentialSource(comm, None, None)
        distribute_geo_data = lp_source.distibuted_geo_data(None, queue, None)

        from pytential.qbx.distributed import drive_dfmm
        wrangler = None
        weights = None
        drive_dfmm(queue, weights, distribute_geo_data, comm=comm)
