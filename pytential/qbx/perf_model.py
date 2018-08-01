from boxtree.distributed.perf_model import PerformanceCounter
from collections import namedtuple

QBXParameters = namedtuple(
    "QBXParameters",
    ['ncoeffs_fmm_by_level',
     'ncoeffs_qbx',
     'translation_source_power',
     'translation_target_power',
     'translation_max_power']
)


class QBXPerformanceCounter(PerformanceCounter):

    def __init__(self, traversal, wrangler, uses_pde_expansions):
        self.traversal = traversal
        self.wrangler = wrangler
        self.uses_pde_expansions = uses_pde_expansions

        self.parameters = self.get_qbx_parameters(
            traversal.tree.dimensions,
            uses_pde_expansions,
            wrangler.level_nterms,
            wrangler.qbx_order
        )

    @staticmethod
    def get_qbx_parameters(dimensions, use_pde_expansions, level_nterms, qbx_order):
        fmm_parameters = PerformanceCounter.get_fmm_parameters(
            dimensions, use_pde_expansions, level_nterms
        )

        if use_pde_expansions:
            ncoeffs_qbx = qbx_order ** (dimensions - 1)
        else:
            ncoeffs_qbx = qbx_order ** dimensions

        return QBXParameters(
            ncoeffs_fmm_by_level=fmm_parameters.ncoeffs_fmm_by_level,
            ncoeffs_qbx=ncoeffs_qbx,
            translation_source_power=fmm_parameters.translation_source_power,
            translation_target_power=fmm_parameters.translation_target_power,
            translation_max_power=fmm_parameters.translation_max_power,
        )
