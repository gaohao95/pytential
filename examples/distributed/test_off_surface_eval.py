import pyopencl as cl
from meshmode.mesh.generation import (
    make_curve_mesh, ellipse)
import functools
from sympy.core.cache import clear_cache
import numpy as np
from pytential.qbx.distributed import DistributedQBXLayerPotentialSource
from pytential.symbolic.execution import bind_distributed
from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import (
    InterpolatoryQuadratureSimplexGroupFactory)
from sumpy.kernel import LaplaceKernel
import pytential
from sumpy.visualization import FieldPlotter
from pytential.target import PointsTarget
import matplotlib.pyplot as pt
from mpi4py import MPI
import logging
import os

# Set up logging infrastructure
logging.basicConfig(level=os.environ.get("LOGLEVEL", "WARNING"))
logging.getLogger("boxtree.distributed").setLevel(logging.INFO)
logging.getLogger("pytential.qbx.distributed").setLevel(logging.INFO)

# Get MPI information
comm = MPI.COMM_WORLD
current_rank = comm.Get_rank()
total_rank = comm.Get_size()

# Disable sympy cache
clear_cache()

# Setup PyOpenCL
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

# Parameters
nelements = 30
target_order = 8
qbx_order = 3
fmm_order = qbx_order

if current_rank == 0:  # master rank
    mesh = make_curve_mesh(functools.partial(ellipse, 3),
                           np.linspace(0, 1, nelements + 1),
                           target_order)

    pre_density_discr = Discretization(
        ctx, mesh, InterpolatoryQuadratureSimplexGroupFactory(target_order))

    qbx = DistributedQBXLayerPotentialSource(
        comm,
        pre_density_discr,
        fine_order=4 * target_order,
        qbx_order=qbx_order,
        fmm_order=fmm_order,
        knl_specific_calibration_params="constant_one"
    )

    op = pytential.sym.D(
        LaplaceKernel(2), pytential.sym.var("sigma"), qbx_forced_limit=-2)

    qbx, _ = qbx.with_refinement()
    density_discr = qbx.density_discr
    sigma = density_discr.zeros(queue) + 1
    qbx_ctx = {"sigma": sigma}

    fplot = FieldPlotter(np.zeros(2), extent=0.54, npoints=30)
    targets = PointsTarget(fplot.points)
else:
    qbx = None
    targets = None
    op = None
    qbx_ctx = {}

fld_in_vol = bind_distributed(comm, (qbx, targets), op)(queue, **qbx_ctx)

if current_rank == 0:
    err = cl.clmath.fabs(fld_in_vol - (-1))

    linf_err = cl.array.max(err).get()
    print("l_inf error:", linf_err)

    fplot.show_scalar_in_matplotlib(fld_in_vol.get())

    pt.colorbar()
    pt.show()

    # FIXME: Why does the FMM only meet this sloppy tolerance?
    assert linf_err < 1e-2
