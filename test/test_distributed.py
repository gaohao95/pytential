import numpy as np
import numpy.linalg as la  # noqa
import pyopencl as cl
import pyopencl.clmath  # noqa
from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)

from meshmode.mesh.generation import make_curve_mesh, ellipse
from sumpy.visualization import FieldPlotter
from pytential import sym
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


if __name__ == "__main__":
    if "PYTEST" in os.environ:
        if os.environ["PYTEST"] == "1":
            # Run "test_off_surface_eval" test case
            use_fmm = (os.environ["use_fmm"] == 'True')
            do_plot = (os.environ["do_plot"] == 'True')

            _test_off_surface_eval(cl.create_some_context, use_fmm, do_plot=do_plot)
    else:
        if len(sys.argv) > 1:

            # You can test individual routines by typing
            # $ python test_distributed.py 'test_off_surface_eval(4, True, True)'
            exec(sys.argv[1])
