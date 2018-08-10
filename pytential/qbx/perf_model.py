import numpy as np
from boxtree.distributed.perf_model import PerformanceCounter, PerformanceModel
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

    def count_p2qbxl(self, use_global_idx=False):
        geo_data = self.geo_data
        traversal = self.traversal
        tree = traversal.tree
        qbx_center_to_target_box = geo_data.qbx_center_to_target_box()

        if use_global_idx:
            np2qbxl = np.zeros((tree.nboxes,), dtype=np.intp)
        else:
            ntarget_boxes = len(traversal.target_boxes)
            np2qbxl = np.zeros((ntarget_boxes,), dtype=np.intp)

        for tgt_icenter in geo_data.global_qbx_centers():
            itgt_box = qbx_center_to_target_box[tgt_icenter]

            np2qbxl_srcs = 0

            # list 1
            start, end = traversal.neighbor_source_boxes_starts[
                            itgt_box:itgt_box + 2]
            for src_ibox in traversal.neighbor_source_boxes_lists[start:end]:
                np2qbxl_srcs += tree.box_source_counts_nonchild[src_ibox]

            # list 3 close
            if traversal.from_sep_close_smaller_starts is not None:
                start, end = traversal.from_sep_close_smaller_starts[
                                itgt_box:itgt_box + 2]
                for src_ibox in traversal.from_sep_close_smaller_lists[start:end]:
                    np2qbxl_srcs += tree.box_source_counts_nonchild[src_ibox]

            # list 4 close
            if traversal.from_sep_close_bigger_starts is not None:
                # POSSIBLY USE INTERFACE WRONGLY
                start, end = traversal.from_sep_close_bigger_starts[
                                itgt_box:itgt_box + 2]
                for src_ibox in traversal.from_sep_close_bigger_lists[start:end]:
                    np2qbxl_srcs += tree.box_source_counts_nonchild[src_ibox]

            workload = np2qbxl_srcs * self.parameters.ncoeffs_qbx

            if use_global_idx:
                np2qbxl[traversal.target_boxes[itgt_box]] += workload
            else:
                np2qbxl[itgt_box] += workload

        return np2qbxl

    def count_m2qbxl(self, use_global_idx=False):
        geo_data = self.geo_data
        traversal = self.traversal
        tree = traversal.tree
        global_qbx_centers = geo_data.global_qbx_centers()

        if use_global_idx:
            nm2qbxl = np.zeros((tree.nboxes,), dtype=np.intp)
        else:
            ntarget_boxes = len(traversal.target_boxes)
            nm2qbxl = np.zeros((ntarget_boxes,), dtype=np.intp)

        for isrc_level, ssn in enumerate(traversal.from_sep_smaller_by_level):

            target_boxes_sep_smaller_current_level = \
                traversal.target_boxes_sep_smaller_by_source_level[isrc_level]

            cost_coefficient = self.xlat_cost(
                self.wrangler.level_nterms[isrc_level],
                self.wrangler.qbx_order,
                self.parameters
            )

            qbx_center_to_target_box_current_level = \
                geo_data.qbx_center_to_target_box_source_level(isrc_level)

            for itgt_center, tgt_icenter in enumerate(global_qbx_centers):
                icontaining_tgt_box = qbx_center_to_target_box_current_level[
                    tgt_icenter
                ]

                if icontaining_tgt_box == -1:
                    continue

                start, stop = (
                        ssn.starts[icontaining_tgt_box],
                        ssn.starts[icontaining_tgt_box+1])

                cost = (stop - start) * cost_coefficient

                if use_global_idx:
                    global_boxes_idx = \
                        target_boxes_sep_smaller_current_level[icontaining_tgt_box]
                    nm2qbxl[global_boxes_idx] += cost
                else:
                    target_boxes_idx = ssn.nonempty_indices[icontaining_tgt_box]
                    nm2qbxl[target_boxes_idx] += cost

        return nm2qbxl

    def count_l2qbxl(self, use_global_idx=False):
        geo_data = self.geo_data
        traversal = self.traversal
        tree = traversal.tree
        qbx_center_to_target_box = geo_data.qbx_center_to_target_box()
        global_qbx_centers = geo_data.global_qbx_centers()

        if use_global_idx:
            nl2qbxl = np.zeros(tree.nboxes, dtype=np.intp)
        else:
            ntarget_boxes = len(traversal.target_boxes)
            nl2qbxl = np.zeros(ntarget_boxes, dtype=np.intp)

        for src_icenter in global_qbx_centers:
            target_box_idx = qbx_center_to_target_box[src_icenter]
            global_box_idx = traversal.target_boxes[target_box_idx]

            box_level = tree.box_levels[global_box_idx]

            cost = self.xlat_cost(
                self.wrangler.level_nterms[box_level],
                self.wrangler.qbx_order,
                self.parameters
            )

            if use_global_idx:
                nl2qbxl[global_box_idx] += cost
            else:
                nl2qbxl[target_box_idx] += cost

        return nl2qbxl

    def count_eval_qbxl(self, use_global_idx=False):
        geo_data = self.geo_data
        traversal = self.traversal
        tree = traversal.tree
        qbx_center_to_target_box = geo_data.qbx_center_to_target_box()
        global_qbx_centers = geo_data.global_qbx_centers()
        center_to_targets_starts = geo_data.center_to_tree_targets().starts

        if use_global_idx:
            neval_qbxl = np.zeros((tree.nboxes,), dtype=np.intp)
        else:
            ntarget_boxes = len(traversal.target_boxes)
            neval_qbxl = np.zeros((ntarget_boxes,), dtype=np.intp)

        for src_icenter in global_qbx_centers:
            start, end = center_to_targets_starts[src_icenter:src_icenter+2]
            cost = (end - start) * self.parameters.ncoeffs_qbx

            target_box_idx = qbx_center_to_target_box[src_icenter]

            if use_global_idx:
                global_box_idx = traversal.target_boxes[target_box_idx]
                neval_qbxl[global_box_idx] += cost
            else:
                neval_qbxl[target_box_idx] += cost

        return neval_qbxl


class QBXPerformanceModel(PerformanceModel):

    def __init__(self, cl_context, uses_pde_expansions):
        super(QBXPerformanceModel, self).__init__(
            cl_context, uses_pde_expansions
        )

    def time_performance(self, traversal, wrangler):
        raise NotImplementedError("Please use time_qbx_performance instead.")

    def time_qbx_performance(self, queue, bound_op, context):
        timing_data = {}

        def expansion_wrangler_inspector(wrangler):
            counter = QBXPerformanceCounter(
                wrangler.geo_data, wrangler, self.uses_pde_expansions
            )
            traversal = wrangler.geo_data.traversal()

            nm2p, nm2p_boxes = counter.count_m2p()

            from pytential.qbx.fmm import add_dicts
            timing_data.update(add_dicts(timing_data, {
                "nterms_fmm_total": counter.count_nters_fmm_total(),
                "direct_workload": np.sum(counter.count_direct()),
                "direct_nsource_boxes": traversal.neighbor_source_boxes_starts[-1],
                "m2l_workload": np.sum(counter.count_m2l()),
                "m2p_workload": np.sum(nm2p),
                "m2p_nboxes": np.sum(nm2p_boxes),
                "p2l_workload": np.sum(counter.count_p2l()),
                "p2l_nboxes": np.sum(counter.count_p2l_source_boxes()),
                "eval_part_workload": np.sum(counter.count_eval_part()),
                "p2qbxl_workload": np.sum(counter.count_p2qbxl()),
                "m2qbxl_workload": np.sum(counter.count_m2qbxl()),
                "l2qbxl_workload": np.sum(counter.count_l2qbxl()),
                "eval_qbxl_workload": np.sum(counter.count_eval_qbxl())
            }))

        from pytential.symbolic.primitives import DEFAULT_SOURCE
        bound_op.places[DEFAULT_SOURCE].bind_expansion_wrangler_inspector(
            expansion_wrangler_inspector
        )

        bound_op.eval(queue, context=context, timing_data=timing_data)

        self.time_result.append(timing_data)

    def form_global_qbx_locals_model(self, wall_time=True):
        return self.linear_regression(
            "form_global_qbx_locals", ["p2qbxl_workload"],
            wall_time=wall_time
        )

    def translate_box_multipoles_to_qbx_local_model(self, wall_time=True):
        return self.linear_regression(
            "translate_box_multipoles_to_qbx_local", ["m2qbxl_workload"],
            wall_time=wall_time
        )

    def translate_box_local_to_qbx_local_model(self, wall_time=True):
        return self.linear_regression(
            "translate_box_local_to_qbx_local", ["l2qbxl_workload"],
            wall_time=wall_time
        )

    def eval_qbx_expansions_model(self, wall_time=True):
        return self.linear_regression(
            "eval_qbx_expansions", ["eval_qbxl_workload"],
            wall_time=wall_time
        )

    def predict_boxes_time(self, geo_data, wrangler):
        boxes_time = super(QBXPerformanceModel, self).predict_boxes_time(
            geo_data.traversal(), wrangler
        )

        counter = QBXPerformanceCounter(geo_data, wrangler, self.uses_pde_expansions)

        # {{{ form_global_qbx_locals time

        param = self.form_global_qbx_locals_model()

        p2qbxl_workload = counter.count_p2qbxl(use_global_idx=True)

        boxes_time += (p2qbxl_workload * param[0] + param[1])

        # }}}

        # {{{ translate_box_multipoles_to_qbx_local time

        param = self.translate_box_multipoles_to_qbx_local_model()

        m2qbxl_workload = counter.count_m2qbxl(use_global_idx=True)

        boxes_time += (m2qbxl_workload * param[0] + param[1])

        # }}}

        # {{{ translate_box_local_to_qbx_local time

        param = self.translate_box_local_to_qbx_local_model()

        l2qbxl_workload = counter.count_l2qbxl(use_global_idx=True)

        boxes_time += (l2qbxl_workload * param[0] + param[1])

        # }}}

        # {{{ eval_qbx_expansions time

        param = self.eval_qbx_expansions_model()

        eval_qbxl_workload = counter.count_eval_qbxl(use_global_idx=True)

        boxes_time += (eval_qbxl_workload * param[0] + param[1])

        # }}}

        return boxes_time

    def predict_step_time(self, eval_counter, wall_time=True):
        predict_timing = super(QBXPerformanceModel, self).predict_step_time(
            eval_counter, wall_time=wall_time
        )

        # {{{ Predict form_global_qbx_locals time

        param = self.form_global_qbx_locals_model(wall_time=wall_time)

        p2qbxl_workload = np.sum(eval_counter.count_p2qbxl())

        predict_timing["form_global_qbx_locals"] = (
            p2qbxl_workload * param[0] + param[1]
        )

        # }}}

        # {{{ Predict translate_box_multipoles_to_qbx_local time

        param = self.translate_box_multipoles_to_qbx_local_model(wall_time=wall_time)

        m2qbxl_workload = np.sum(eval_counter.count_m2qbxl())

        predict_timing["translate_box_multipoles_to_qbx_local"] = (
            m2qbxl_workload * param[0] + param[1]
        )

        # }}}

        # {{{ Predict translate_box_local_to_qbx_local time

        param = self.translate_box_local_to_qbx_local_model(wall_time=wall_time)

        l2qbxl_workload = np.sum(eval_counter.count_l2qbxl())

        predict_timing["translate_box_local_to_qbx_local"] = (
            l2qbxl_workload * param[0] + param[1]
        )

        # }}}

        # {{{ Predict eval_qbx_expansions time

        param = self.eval_qbx_expansions_model(wall_time=wall_time)

        eval_qbxl_workload = np.sum(eval_counter.count_eval_qbxl())

        predict_timing["eval_qbx_expansions"] = (
            eval_qbxl_workload * param[0] + param[1]
        )

        # }}}

        return predict_timing

    def evaluate_model(self, queue, bound_op, context, wall_time=True):
        predict_timing = {}

        def expansion_wrangler_inspector(wrangler):
            eval_counter = QBXPerformanceCounter(
                wrangler.geo_data, wrangler, self.uses_pde_expansions
            )

            from pytential.qbx.fmm import add_dicts
            predict_timing.update(add_dicts(
                predict_timing,
                self.predict_step_time(eval_counter, wall_time=wall_time)
            ))

        from pytential.symbolic.primitives import DEFAULT_SOURCE
        bound_op.places[DEFAULT_SOURCE].bind_expansion_wrangler_inspector(
            expansion_wrangler_inspector
        )

        actual_timing = {}
        bound_op.eval(queue, context=context, timing_data=actual_timing)

        for field in ["eval_direct", "multipole_to_local", "eval_multipoles",
                      "form_locals", "eval_locals", "form_global_qbx_locals",
                      "translate_box_multipoles_to_qbx_local",
                      "translate_box_local_to_qbx_local", "eval_qbx_expansions"]:
            predict_time_field = predict_timing[field]

            if wall_time:
                true_time_field = actual_timing[field].wall_elapsed
            else:
                true_time_field = actual_timing[field].process_elapsed

            diff = abs(predict_time_field - true_time_field)

            print(field + ": predict " + str(predict_time_field) + " actual "
                  + str(true_time_field) + " error " + str(diff / true_time_field))

    def load_default_model(self):
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        default_perf_file_path = os.path.join(current_dir, 'default_perf_model.json')
        self.loadjson(default_perf_file_path)
