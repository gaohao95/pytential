from pytential.qbx.fmmlib import (
    QBXFMMLibExpansionWranglerCodeContainer)
from mpi4py import MPI


# {{{ Expansion Wrangler

class QBXDistributedFMMLibExpansionWranglerCodeContainer(
        QBXFMMLibExpansionWranglerCodeContainer):
    pass

# }}}


# {{{ FMM Driver

def drive_dfmm(expansion_wrangler, src_weights, comm=MPI.COMM_WORLD):
    current_rank = comm.Get_rank()
    # total_rank = comm.Get_size()

    if current_rank == 0:
        wrangler = expansion_wrangler

        geo_data = wrangler.geo_data
        traversal = geo_data.traversal()
        tree = traversal.tree

        # Interface guidelines: Attributes of the tree are assumed to be known
        # to the expansion wrangler and should not be passed.

        src_weights = wrangler.reorder_sources(src_weights)

        # {{{ construct local multipoles

        mpole_exps = wrangler.form_multipoles(
            traversal.level_start_source_box_nrs,
            traversal.source_boxes,
            src_weights)

        # }}}

        # {{{ propagate multipoles upward

        wrangler.coarsen_multipoles(
            traversal.level_start_source_parent_box_nrs,
            traversal.source_parent_boxes,
            mpole_exps)

        # }}}

        # {{{ direct evaluation from neighbor source boxes ("list 1")

        non_qbx_potentials = wrangler.eval_direct(
            traversal.target_boxes,
            traversal.neighbor_source_boxes_starts,
            traversal.neighbor_source_boxes_lists,
            src_weights)

        # }}}

        # {{{ translate separated siblings' ("list 2") mpoles to local

        local_exps = wrangler.multipole_to_local(
            traversal.level_start_target_or_target_parent_box_nrs,
            traversal.target_or_target_parent_boxes,
            traversal.from_sep_siblings_starts,
            traversal.from_sep_siblings_lists,
            mpole_exps)

        # }}}

        # {{{ evaluate sep. smaller mpoles ("list 3") at particles

        # (the point of aiming this stage at particles is specifically to keep its
        # contribution *out* of the downward-propagating local expansions)

        non_qbx_potentials = non_qbx_potentials + wrangler.eval_multipoles(
            traversal.target_boxes_sep_smaller_by_source_level,
            traversal.from_sep_smaller_by_level,
            mpole_exps)

        # assert that list 3 close has been merged into list 1
        assert traversal.from_sep_close_smaller_starts is None

        # }}}

        # {{{ form locals for separated bigger source boxes ("list 4")

        local_exps = local_exps + wrangler.form_locals(
            traversal.level_start_target_or_target_parent_box_nrs,
            traversal.target_or_target_parent_boxes,
            traversal.from_sep_bigger_starts,
            traversal.from_sep_bigger_lists,
            src_weights)

        # assert that list 4 close has been merged into list 1
        assert traversal.from_sep_close_bigger_starts is None

        # }}}

        # {{{ propagate local_exps downward

        wrangler.refine_locals(
            traversal.level_start_target_or_target_parent_box_nrs,
            traversal.target_or_target_parent_boxes,
            local_exps)

        # }}}

        # {{{ evaluate locals

        non_qbx_potentials = non_qbx_potentials + wrangler.eval_locals(
            traversal.level_start_target_box_nrs,
            traversal.target_boxes,
            local_exps)

        # }}}

        # {{{ wrangle qbx expansions

        qbx_expansions = wrangler.form_global_qbx_locals(src_weights)

        qbx_expansions = qbx_expansions + \
                         wrangler.translate_box_multipoles_to_qbx_local(mpole_exps)

        qbx_expansions = qbx_expansions + \
                         wrangler.translate_box_local_to_qbx_local(local_exps)

        qbx_potentials = wrangler.eval_qbx_expansions(
            qbx_expansions)

        # }}}

        # {{{ reorder potentials

        nqbtl = geo_data.non_qbx_box_target_lists()

        all_potentials_in_tree_order = wrangler.full_output_zeros()

        for ap_i, nqp_i in zip(all_potentials_in_tree_order, non_qbx_potentials):
            ap_i[nqbtl.unfiltered_from_filtered_target_indices] = nqp_i

        all_potentials_in_tree_order += qbx_potentials

        def reorder_and_finalize_potentials(x):
            # "finalize" gives host FMMs (like FMMlib) a chance to turn the
            # potential back into a CL array.
            return wrangler.finalize_potentials(x[tree.sorted_target_ids])

        from pytools.obj_array import with_object_array_or_scalar
        result = with_object_array_or_scalar(
            reorder_and_finalize_potentials, all_potentials_in_tree_order)

        # }}}

        return result
    else:
        pass

# }}}
