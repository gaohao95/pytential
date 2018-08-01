import numpy as np
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

    def __init__(self, geo_data, wrangler, uses_pde_expansions):
        self.geo_data = geo_data
        self.traversal = geo_data.traversal()
        self.wrangler = wrangler
        self.uses_pde_expansions = uses_pde_expansions

        self.parameters = self.get_qbx_parameters(
            self.traversal.tree.dimensions,
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

    def count_direct(self, use_global_idx=False):
        """
        This method overwrites the one in parent class because the only non-qbx
        targets should be counted.

        :return: If *use_global_idx* is True, return a numpy array of shape
            (tree.nboxes,) such that the ith entry represents the workload from
            direct evaluation on box i. If *use_global_idx* is False, return a numpy
            array of shape (ntarget_boxes,) such that the ith entry represents the
            workload on *target_boxes* i.
        """
        box_target_counts_nonchild = self.geo_data.non_qbx_box_target_lists()\
                                         .box_target_counts_nonchild
        traversal = self.traversal
        tree = traversal.tree

        if use_global_idx:
            direct_workload = np.zeros((tree.nboxes,), dtype=np.intp)
        else:
            ntarget_boxes = len(traversal.target_boxes)
            direct_workload = np.zeros((ntarget_boxes,), dtype=np.intp)

        for itgt_box, tgt_ibox in enumerate(traversal.target_boxes):
            ntargets = box_target_counts_nonchild[tgt_ibox]
            nsources = 0

            start, end = traversal.neighbor_source_boxes_starts[itgt_box:itgt_box+2]

            for src_ibox in traversal.neighbor_source_boxes_lists[start:end]:
                nsources += tree.box_source_counts_nonchild[src_ibox]

            if traversal.from_sep_close_smaller_starts is not None:
                start, end = (
                    traversal.from_sep_close_smaller_starts[itgt_box:itgt_box+2])

                for src_ibox in traversal.from_sep_close_smaller_lists[start:end]:
                    nsources += tree.box_source_counts_nonchild[src_ibox]

            if traversal.from_sep_close_bigger_starts is not None:
                start, end = (
                    traversal.from_sep_close_bigger_starts[itgt_box:itgt_box+2])

                for src_ibox in traversal.from_sep_close_bigger_lists[start:end]:
                    nsources += tree.box_source_counts_nonchild[src_ibox]

            count = nsources * ntargets

            if use_global_idx:
                direct_workload[tgt_ibox] = count
            else:
                direct_workload[itgt_box] = count

        return direct_workload
