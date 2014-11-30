from __future__ import division, absolute_import, print_function

__copyright__ = "Copyright (C) 2014 Shidong Jiang, Andreas Kloeckner"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


import numpy as np
import numpy.linalg as la
import pyopencl as cl
import pyopencl.array  # noqa
import pyopencl.clmath  # noqa

import pytest

from pytools.obj_array import make_obj_array

from meshmode.discretization import Discretization
from meshmode.discretization.poly_element import \
        InterpolatoryQuadratureSimplexGroupFactory

from six.moves import range

from pytential import bind, sym, norm  # noqa

from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)

import logging
logger = logging.getLogger(__name__)


def run_dielectric_test(cl_ctx, queue, nelements, qbx_order,
        k0=3, k1=2.9, mesh_order=10,
        bdry_quad_order=None, bdry_ovsmp_quad_order=None,
        fmm_order=False, visualize=False):

    if bdry_quad_order is None:
        bdry_quad_order = mesh_order
    if bdry_ovsmp_quad_order is None:
        bdry_ovsmp_quad_order = 4*bdry_quad_order

    from meshmode.mesh.generation import ellipse, make_curve_mesh
    from functools import partial
    mesh = make_curve_mesh(
            partial(ellipse, 3),
            np.linspace(0, 1, nelements+1),
            mesh_order)

    density_discr = Discretization(
            cl_ctx, mesh,
            InterpolatoryQuadratureSimplexGroupFactory(bdry_quad_order))

    logger.info("%d elements" % mesh.nelements)

    # from meshmode.discretization.visualization import make_visualizer
    # bdry_vis = make_visualizer(queue, density_discr, 20)

    # {{{ solve bvp

    from sumpy.kernel import HelmholtzKernel, AxisTargetDerivative
    kernel_K0 = HelmholtzKernel(2, helmholtz_k_name="K0")
    kernel_K1 = HelmholtzKernel(2, helmholtz_k_name="K1")

    cse = sym.cse

    sqrt_w_sym = sym.sqrt_jac_q_weight()

    unkwown = sym.make_sym_vector("unknown", 2)
    sigma_sym = unkwown[0]
    mu_sym = unkwown[1]
    inv_sqrt_w_sigma = cse(sigma_sym/sqrt_w_sym)
    inv_sqrt_w_mu = cse(mu_sym/sqrt_w_sym)

    beta = 2.5
    K0 = np.sqrt(k0**2-beta**2)
    K1 = np.sqrt(k1**2-beta**2)

    print("K0", K0)
    print("K1", K1)

    alpha0 = k0**2/(k0**2-beta**2)
    alpha1 = k1**2/(k1**2-beta**2)

    coeff_0_S = 1
    coeff_1_S = 1

    coeff_0_D = alpha1
    coeff_1_D = alpha0

    bdry_op_sym = make_obj_array([
        (-0.5*(alpha0*coeff_0_S+alpha1*coeff_1_S)*sigma_sym
            + sqrt_w_sym*(
                alpha0*coeff_0_S*sym.Sp(kernel_K0, inv_sqrt_w_sigma)
                - alpha1*coeff_1_S*sym.Sp(kernel_K1, inv_sqrt_w_sigma)
                + alpha0*coeff_0_D*sym.Dp(kernel_K0, inv_sqrt_w_mu)
                - alpha1*coeff_1_D*sym.Dp(kernel_K1, inv_sqrt_w_mu)
                )),
        (0.5*(coeff_0_D+coeff_1_D)*mu_sym
            + sqrt_w_sym*(
                coeff_0_S*sym.S(kernel_K0, inv_sqrt_w_sigma)
                - coeff_1_S*sym.S(kernel_K1, inv_sqrt_w_sigma)
                + coeff_0_D*sym.D(kernel_K0, inv_sqrt_w_mu)
                - coeff_1_D*sym.D(kernel_K1, inv_sqrt_w_mu)
                ))
        ])
    #print sym.pretty(bdry_op_sym)
    #1/0

    from pytential.qbx import QBXLayerPotentialSource
    qbx = QBXLayerPotentialSource(
            density_discr, fine_order=bdry_ovsmp_quad_order, qbx_order=qbx_order,
            fmm_order=fmm_order
            )

    bound_op = bind(qbx, bdry_op_sym)

    sources_0 = make_obj_array(list(np.array([
        [0.1, 0.2]
        ]).T.copy()))
    strengths_0 = np.array([1])
    sources_1 = make_obj_array(list(np.array([
        [4, 4]
        ]).T.copy()))
    strengths_1 = np.array([1])

    kernel_K0_grad = [
        AxisTargetDerivative(i, kernel_K0) for i in range(density_discr.ambient_dim)]
    kernel_K1_grad = [
        AxisTargetDerivative(i, kernel_K1) for i in range(density_discr.ambient_dim)]

    from sumpy.p2p import P2P
    pot_p2p_K0 = P2P(cl_ctx, [kernel_K0], exclude_self=False)
    pot_p2p_K1 = P2P(cl_ctx, [kernel_K1], exclude_self=False)
    pot_p2p_grad_K0 = P2P(cl_ctx, kernel_K0_grad, exclude_self=False)
    pot_p2p_grad_K1 = P2P(cl_ctx, kernel_K1_grad, exclude_self=False)

    normal = bind(density_discr, sym.normal())(queue).as_vector(np.object)

    _, (E0,) = pot_p2p_K0(queue, density_discr.nodes(), sources_0, [strengths_0],
                    out_host=False, K0=K0)
    _, (E1,) = pot_p2p_K1(queue, density_discr.nodes(), sources_1, [strengths_1],
                    out_host=False, K1=K1)
    _, (grad0_E0, grad1_E0) = pot_p2p_grad_K0(
        queue, density_discr.nodes(), sources_0, [strengths_0],
        out_host=False, K0=K0)
    _, (grad0_E1, grad1_E1) = pot_p2p_grad_K1(
        queue, density_discr.nodes(), sources_1, [strengths_1],
        out_host=False, K1=K1)

    E0_dntarget = (grad0_E0*normal[0] + grad1_E0*normal[1])
    E1_dntarget = (grad0_E1*normal[0] + grad1_E1*normal[1])

    sqrt_w = bind(density_discr, sqrt_w_sym)(queue)

    bvp_rhs = make_obj_array([
        sqrt_w*(alpha0 * E0_dntarget - alpha1 * E1_dntarget),
        sqrt_w*(E0 - E1)
        ])

    from pytential.gmres import gmres
    gmres_result = gmres(
            bound_op.scipy_op(queue, "unknown",
                domains=[sym.DEFAULT_TARGET]*2, K0=K0, K1=K1),
            bvp_rhs, tol=1e-14, progress=True,
            hard_failure=True)

    # }}}

    unknown = gmres_result.solution

    representation0_sym = (
            coeff_0_S*sym.S(kernel_K0, inv_sqrt_w_sigma)
            + coeff_0_D*sym.D(kernel_K0, inv_sqrt_w_mu))

    representation1_sym = (
            coeff_1_S*sym.S(kernel_K1, inv_sqrt_w_sigma)
            + coeff_1_D*sym.D(kernel_K1, inv_sqrt_w_mu))

    targets_0 = make_obj_array(list(np.array([
        [3.2 + t, -4]
        for t in [0, 0.5, 1]
        ]).T.copy()))
    targets_1 = make_obj_array(list(np.array([
        [t*-0.3, t*-0.2]
        for t in [0, 0.5, 1]
        ]).T.copy()))

    from pytential.target import PointsTarget
    E0_tgt = bind(
            (qbx, PointsTarget(targets_0)),
            representation0_sym)(queue, unknown=unknown, K0=K0, K1=K1).get()
    E1_tgt = bind(
            (qbx, PointsTarget(targets_1)),
            representation1_sym)(queue, unknown=unknown, K0=K0, K1=K1).get()

    _, (E0_tgt_true,) = pot_p2p_K0(queue, targets_0, sources_0, [strengths_0],
                    out_host=True, K0=K0)
    _, (E1_tgt_true,) = pot_p2p_K1(queue, targets_1, sources_1, [strengths_1],
                    out_host=True, K1=K1)

    err_E0 = la.norm(E0_tgt - E0_tgt_true)/la.norm(E0_tgt_true)
    err_E1 = la.norm(E1_tgt - E1_tgt_true)/la.norm(E1_tgt_true)

    print("Err E0", err_E0)
    print("Err E1", err_E1)

    if visualize:
        from sumpy.visualization import FieldPlotter
        fplot = FieldPlotter(np.zeros(2), extent=5, npoints=300)
        from pytential.target import PointsTarget
        fld0 = bind(
                (qbx, PointsTarget(fplot.points)),
                representation0_sym)(queue, unknown=unknown, K0=K0).get()
        fld1 = bind(
                (qbx, PointsTarget(fplot.points)),
                representation1_sym)(queue, unknown=unknown, K1=K1).get()
        _, (fld0_true,) = pot_p2p_K0(queue, fplot.points, sources_0, [strengths_0],
                        out_host=True, K0=K0)
        _, (fld1_true,) = pot_p2p_K1(queue, fplot.points, sources_1, [strengths_1],
                        out_host=True, K1=K1)

        #fplot.show_scalar_in_mayavi(fld_in_vol.real, max_val=5)
        fplot.write_vtk_file(
                "potential.vts",
                [
                    ("fld0", fld0),
                    ("fld1", fld1),
                    ("fld0_true", fld0_true),
                    ("fld1_true", fld1_true),
                    ]
                )

    return err_E0, err_E1


@pytest.mark.parametrize("qbx_order", [4])
def test_dielectric(ctx_getter, qbx_order, visualize=False):
    cl_ctx = ctx_getter()
    queue = cl.CommandQueue(cl_ctx)

    import logging
    logging.basicConfig(level=logging.INFO)

    from pytools.convergence import EOCRecorder
    eoc_rec = EOCRecorder()

    for nelements in [30, 50, 70]:
        # prevent cache 'splosion
        from sympy.core.cache import clear_cache
        clear_cache()

        errs = run_dielectric_test(
                cl_ctx, queue,
                nelements=nelements, qbx_order=qbx_order)

        eoc_rec.add_data_point(1/nelements, la.norm(list(errs)))

    print(eoc_rec)
    assert eoc_rec.order_estimate() > qbx_order - 0.5


# You can test individual routines by typing
# $ python test_layer_pot.py 'test_routine()'

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

# vim: fdm=marker