import pyopencl as cl
from meshmode.mesh.generation import (
    make_curve_mesh, ellipse)
import functools
from sympy.core.cache import clear_cache
import numpy as np
from pytential.qbx.distributed import DistributedQBXLayerPotentialSource
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

    qbx, _ = DistributedQBXLayerPotentialSource(
        comm,
        pre_density_discr,
        fine_order=4 * target_order,
        qbx_order=qbx_order,
        fmm_order=fmm_order
    ).with_refinement()

    density_discr = qbx.density_discr

    op = pytential.sym.D(
        LaplaceKernel(2), pytential.sym.var("sigma"), qbx_forced_limit=-2)

    sigma = density_discr.zeros(queue) + 1

    fplot = FieldPlotter(np.zeros(2), extent=0.54, npoints=30)

    fld_in_vol = pytential.bind(
        (qbx, PointsTarget(fplot.points)),
        op)(queue, sigma=sigma)

    err = cl.clmath.fabs(fld_in_vol - (-1))

    linf_err = cl.array.max(err).get()
    print("l_inf error:", linf_err)

    fplot.show_scalar_in_matplotlib(fld_in_vol.get())

    pt.colorbar()
    pt.show()

    # FIXME: Why does the FMM only meet this sloppy tolerance?
    assert linf_err < 1e-2

else:  # helper rank
    lp_source = DistributedQBXLayerPotentialSource(comm, None, None)
    distribute_geo_data = lp_source.distibuted_geo_data(None, queue, None)

    from pytential.qbx.distributed import drive_dfmm
    wrangler = None
    weights = None
    drive_dfmm(queue, weights, distribute_geo_data, comm=comm)
