import numpy as np
import numpy.linalg as la  # noqa
import pyopencl as cl
import pyopencl.clmath  # noqa
from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)

from meshmode.mesh.generation import make_curve_mesh, ellipse
from sumpy.visualization import FieldPlotter
from pytential import bind, sym
from boxtree.tools import run_mpi

import pytest
from functools import partial
import sys
import os

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import logging
logger = logging.getLogger(__name__)


# {{{ test off-surface eval

def _test_off_surface_eval(ctx_factory, use_fmm, do_plot=False):
    logging.basicConfig(level=logging.INFO)

    cl_ctx = ctx_factory()
    queue = cl.CommandQueue(cl_ctx)

    # prevent cache 'splosion
    from sympy.core.cache import clear_cache
    clear_cache()

    nelements = 30
    target_order = 8
    qbx_order = 3
    if use_fmm:
        fmm_order = qbx_order
    else:
        fmm_order = False

    if rank == 0:

        mesh = make_curve_mesh(partial(ellipse, 3),
                np.linspace(0, 1, nelements+1),
                target_order)

        from pytential.qbx.distributed import DistributedQBXLayerPotentialSource
        from meshmode.discretization import Discretization
        from meshmode.discretization.poly_element import \
                InterpolatoryQuadratureSimplexGroupFactory

        pre_density_discr = Discretization(
            cl_ctx, mesh, InterpolatoryQuadratureSimplexGroupFactory(target_order))
        qbx, _ = DistributedQBXLayerPotentialSource(
            pre_density_discr,
            fine_order=4*target_order,
            qbx_order=qbx_order,
            fmm_order=fmm_order,
            comm=comm,
            knl_specific_calibration_params="constant_one"
        ).with_refinement()

        density_discr = qbx.density_discr

        from sumpy.kernel import LaplaceKernel
        op = sym.D(LaplaceKernel(2), sym.var("sigma"), qbx_forced_limit=-2)

        sigma = density_discr.zeros(queue) + 1
        qbx_ctx = {"sigma": sigma}

        fplot = FieldPlotter(np.zeros(2), extent=0.54, npoints=30)

        from pytential.target import PointsTarget
        targets = PointsTarget(fplot.points)

    else:
        qbx = None
        targets = None
        op = None
        qbx_ctx = {}

    from pytential.symbolic.execution import bind_distributed
    fld_in_vol = bind_distributed(comm, (qbx, targets), op)(queue, **qbx_ctx)

    if rank == 0:
        # test against shared memory result

        from pytential.qbx import QBXLayerPotentialSource
        qbx, _ = QBXLayerPotentialSource(
            pre_density_discr,
            4 * target_order,
            qbx_order,
            fmm_order=fmm_order,
            _from_sep_smaller_min_nsources_cumul=0
        ).with_refinement()

        fld_in_vol_single_node = bind((qbx, targets), op)(queue, **qbx_ctx)

        linf_err = cl.array.max(
            cl.clmath.fabs(fld_in_vol - fld_in_vol_single_node)
        )

        assert linf_err < 1e-13

        # test against analytical solution

        err = cl.clmath.fabs(fld_in_vol - (-1))

        linf_err = cl.array.max(err).get()
        print("l_inf error:", linf_err)

        if do_plot:
            fplot.show_scalar_in_matplotlib(fld_in_vol.get())
            import matplotlib.pyplot as pt
            pt.colorbar()
            pt.show()

        assert linf_err < 1e-3


@pytest.mark.mpi
@pytest.mark.parametrize("num_processes, use_fmm", [
    # (4, False),
    (4, True)
])
@pytest.mark.skipif(sys.version_info < (3, 5),
                    reason="distributed implementation requires 3.5 or higher")
def test_off_surface_eval(num_processes, use_fmm, do_plot=False):
    pytest.importorskip("mpi4py")

    newenv = os.environ.copy()
    newenv["PYTEST"] = "1"
    newenv["OMP_NUM_THREADS"] = "1"
    newenv["use_fmm"] = str(use_fmm)
    newenv["do_plot"] = str(do_plot)

    run_mpi(__file__, num_processes, newenv)

# }}}


# {{{ compare on-surface urchin geometry against single-rank result

def single_layer_wrapper(kernel):
    u_sym = sym.var("u")
    return sym.S(kernel, u_sym, qbx_forced_limit=-1)


def double_layer_wrapper(kernel):
    u_sym = sym.var("u")
    return sym.D(kernel, u_sym, qbx_forced_limit="avg")


def _test_urchin_against_single_rank(ctx_factory, m, n, op_wrapper):
    logging.basicConfig(level=logging.INFO)

    qbx_order = 3
    fmm_order = 10
    target_order = 8
    est_rel_interp_tolerance = 1e-10
    _expansion_stick_out_factor = 0.5

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    if rank == 0:
        from meshmode.mesh.generation import generate_urchin
        mesh = generate_urchin(target_order, m, n, est_rel_interp_tolerance)
        d = mesh.ambient_dim

        from sumpy.kernel import LaplaceKernel
        k_sym = LaplaceKernel(d)
        op = op_wrapper(k_sym)

        from meshmode.discretization import Discretization
        from meshmode.discretization.poly_element import \
            InterpolatoryQuadratureSimplexGroupFactory

        pre_density_discr = Discretization(
            ctx, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(target_order)
        )

        refiner_extra_kwargs = {}

        params = {
            "qbx_order": qbx_order,
            "fmm_order": fmm_order,
            "_from_sep_smaller_min_nsources_cumul": 0,
            "_expansions_in_tree_have_extent": True,
            "_expansion_stick_out_factor": _expansion_stick_out_factor,
            "_use_target_specific_qbx": True,
            "fmm_backend": 'fmmlib'
        }

        from pytential.qbx.distributed import DistributedQBXLayerPotentialSource
        qbx, _ = DistributedQBXLayerPotentialSource(
            density_discr=pre_density_discr,
            fine_order=4 * target_order,
            comm=comm,
            knl_specific_calibration_params="constant_one",
            **params
        ).with_refinement(**refiner_extra_kwargs)

        density_discr = qbx.density_discr

        # {{{ compute values of a solution to the PDE

        nodes_host = density_discr.nodes().get(queue)

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

        # }}}

        u_dev = cl.array.to_device(queue, u)
        grad_u_dev = cl.array.to_device(queue, grad_u)
        context = {'u': u_dev, 'grad_u': grad_u_dev}
    else:
        qbx = None
        op = None
        context = {}

    from pytential.symbolic.execution import bind_distributed
    bound_op = bind_distributed(comm, qbx, op)
    distributed_result = bound_op(queue, **context)

    if rank == 0:
        from pytential.qbx import QBXLayerPotentialSource
        qbx, _ = QBXLayerPotentialSource(
            density_discr=pre_density_discr,
            fine_order=4 * target_order,
            **params
        ).with_refinement()

        single_node_result = bind(qbx, op)(queue, **context)

        distributed_result = distributed_result.get()
        single_node_result = single_node_result.get()

        linf_err = la.norm(distributed_result - single_node_result, ord=np.inf)

        print("l_inf error:", linf_err)

        assert linf_err < 1e-13


@pytest.mark.mpi
@pytest.mark.parametrize("num_processes, m, n, op_wrapper", [
    (4, 1, 3, "single_layer_wrapper"),
    (4, 1, 3, "double_layer_wrapper")
])
@pytest.mark.skipif(sys.version_info < (3, 5),
                    reason="distributed implementation requires 3.5 or higher")
def test_urchin_against_single_rank(num_processes, m, n, op_wrapper):
    pytest.importorskip("mpi4py")

    newenv = os.environ.copy()
    newenv["PYTEST"] = "2"
    newenv["OMP_NUM_THREADS"] = "1"
    newenv["m"] = str(m)
    newenv["n"] = str(n)
    newenv["op_wrapper"] = op_wrapper

    run_mpi(__file__, num_processes, newenv)

# }}}


if __name__ == "__main__":
    if "PYTEST" in os.environ:
        if os.environ["PYTEST"] == "1":
            # Run "test_off_surface_eval" test case
            use_fmm = (os.environ["use_fmm"] == 'True')
            do_plot = (os.environ["do_plot"] == 'True')

            _test_off_surface_eval(cl.create_some_context, use_fmm, do_plot=do_plot)
        elif os.environ["PYTEST"] == "2":
            # Run "test_urchin_against_single_rank" test case
            m = int(os.environ["m"])
            n = int(os.environ["n"])
            op_wrapper_str = os.environ["op_wrapper"]

            if op_wrapper_str == "single_layer_wrapper":
                op_wrapper = single_layer_wrapper
            elif op_wrapper_str == "double_layer_wrapper":
                op_wrapper = double_layer_wrapper
            else:
                raise ValueError("unknown op wrapper")

            _test_urchin_against_single_rank(
                cl.create_some_context, m, n, op_wrapper
            )
    else:
        if len(sys.argv) > 1:

            # You can test individual routines by typing
            # $ python test_distributed.py 'test_off_surface_eval(4, True, True)'
            exec(sys.argv[1])
