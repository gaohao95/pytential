from pytential.qbx.fmmlib import QBXFMMLibExpansionWrangler
from boxtree.distributed import DistributedFMMLibExpansionWrangler, queue
from boxtree.tree import FilteredTargetListsInTreeOrder
from mpi4py import MPI
import numpy as np

import pyopencl as cl


# {{{ Expansion Wrangler

class QBXDistributedFMMLibExpansionWrangler(
        QBXFMMLibExpansionWrangler, DistributedFMMLibExpansionWrangler):

    @classmethod
    def distribute(cls, wrangler, distributed_geo_data, comm=MPI.COMM_WORLD):
        if wrangler is not None:  # master process
            import copy
            distributed_wrangler = copy.copy(wrangler)
            distributed_wrangler.queue = None
            distributed_wrangler.geo_data = None
            distributed_wrangler.code = None
            distributed_wrangler.tree = None
            distributed_wrangler.__class__ = cls
        else:  # worker process
            distributed_wrangler = None

        distributed_wrangler = comm.bcast(distributed_wrangler, root=0)
        distributed_wrangler.tree = distributed_geo_data.local_tree
        distributed_wrangler.geo_data = distributed_geo_data

        return distributed_wrangler

# }}}


# {{{

class DistributedGeoData(object):
    def __init__(self, geo_data, comm=MPI.COMM_WORLD):
        self.comm = comm
        current_rank = comm.Get_rank()
        total_rank = comm.Get_size()

        if geo_data is not None:  # master process
            traversal = geo_data.traversal()
            tree = traversal.tree
            # ncenters = geo_data.ncenters
            # centers = geo_data.centers()
            # expansion_radii = geo_data.expansion_radii()
            # global_qbx_centers = geo_data.global_qbx_centers()
            # qbx_center_to_target_box = geo_data.qbx_center_to_target_box()
            non_qbx_box_target_lists = geo_data.non_qbx_box_target_lists()
            # center_to_tree_targets = geo_data.center_to_tree_targets()

            nlevels = traversal.tree.nlevels
            self.qbx_center_to_target_box_source_level = np.empty(
                (nlevels,), dtype=object)
            for level in range(nlevels):
                self.qbx_center_to_target_box_source_level[level] = (
                    geo_data.qbx_center_to_target_box_source_level(level))
        else:  # worker process
            traversal = None

        from boxtree.distributed import generate_local_tree
        self.local_tree, self.local_data, self.box_bounding_box, knls = \
            generate_local_tree(traversal)

        from boxtree.distributed import generate_local_travs
        self.trav_local, self.trav_global = generate_local_travs(
            self.local_tree, self.box_bounding_box, comm=comm)

        # {{{ Distribute non_qbx_box_target_lists

        if current_rank == 0:  # master process
            box_target_starts = cl.array.to_device(
                queue, non_qbx_box_target_lists.box_target_starts)
            box_target_counts_nonchild = cl.array.to_device(
                queue, non_qbx_box_target_lists.box_target_counts_nonchild)
            nfiltered_targets = non_qbx_box_target_lists.nfiltered_targets
            targets = non_qbx_box_target_lists.targets

            reqs = np.empty((total_rank,), dtype=object)
            local_non_qbx_box_target_lists = np.empty((total_rank,), dtype=object)

            for irank in range(total_rank):
                particle_mask = cl.array.zeros(queue, (nfiltered_targets,),
                                               dtype=tree.particle_id_dtype)
                knls["particle_mask_knl"](
                    self.local_data[irank]["tgt_box_mask"],
                    box_target_starts,
                    box_target_counts_nonchild,
                    particle_mask
                )

                particle_scan = cl.array.empty(queue, (nfiltered_targets + 1,),
                                               dtype=tree.particle_id_dtype)
                particle_scan[0] = 0
                knls["mask_scan_knl"](particle_mask, particle_scan)

                local_box_target_starts = cl.array.empty(
                    queue, (tree.nboxes,), dtype=tree.particle_id_dtype)
                knls["generate_box_particle_starts"](
                    box_target_starts, particle_scan,
                    local_box_target_starts
                )

                local_box_target_counts_nonchild = cl.array.zeros(
                    queue, (tree.nboxes,), dtype=tree.particle_id_dtype)
                knls["generate_box_particle_counts_nonchild"](
                    self.local_data[irank]["tgt_box_mask"],
                    box_target_counts_nonchild,
                    local_box_target_counts_nonchild
                )

                local_nfiltered_targets = particle_scan[-1].get(queue)

                particle_mask = particle_mask.get().astype(bool)
                local_targets = np.empty((tree.dimensions,), dtype=object)
                for idimension in range(tree.dimensions):
                    local_targets[idimension] = targets[idimension][particle_mask]

                local_non_qbx_box_target_lists[irank] = {
                    "nfiltered_targets": local_nfiltered_targets,
                    "box_target_starts": local_box_target_starts.get(),
                    "box_target_counts_nonchild":
                        local_box_target_counts_nonchild.get(),
                    "targets": local_targets
                }

                reqs[irank] = comm.isend(local_non_qbx_box_target_lists[irank],
                                         dest=irank, tag=0)

            for irank in range(1, total_rank):
                reqs[irank].wait()
        if current_rank == 0:
            local_non_qbx_box_target_lists = local_non_qbx_box_target_lists[0]
        else:
            local_non_qbx_box_target_lists = comm.recv(source=0, tag=0)

        self._non_qbx_box_target_lists = FilteredTargetListsInTreeOrder(
            nfiltered_targets=local_non_qbx_box_target_lists["nfiltered_targets"],
            box_target_starts=local_non_qbx_box_target_lists["box_target_starts"],
            box_target_counts_nonchild=local_non_qbx_box_target_lists[
                "box_target_counts_nonchild"],
            targets=local_non_qbx_box_target_lists["targets"],
            unfiltered_from_filtered_target_indices=None
        )

        # }}}

    def non_qbx_box_target_lists(self):
        return self._non_qbx_box_target_lists

# }}}


# {{{ FMM Driver

def drive_dfmm(root_wrangler, src_weights, comm=MPI.COMM_WORLD,
               _communicate_mpoles_via_allreduce=False):
    current_rank = comm.Get_rank()
    total_rank = comm.Get_size()

    if current_rank == 0:
        distributed_geo_data = DistributedGeoData(root_wrangler.geo_data)
    else:
        distributed_geo_data = DistributedGeoData(None)

    distributed_wrangler = QBXDistributedFMMLibExpansionWrangler.distribute(
        root_wrangler, distributed_geo_data)
    wrangler = distributed_wrangler

    local_traversal = distributed_geo_data.trav_local
    global_traversal = distributed_geo_data.trav_global

    # {{{ Distribute source weights

    if current_rank == 0:
        global_tree = root_wrangler.geo_data.tree()
        src_weights = root_wrangler.reorder_sources(src_weights)
    else:
        global_tree = None

    from boxtree.distributed import distribute_source_weights
    local_source_weights = distribute_source_weights(
        src_weights, global_tree, distributed_geo_data.local_data, comm=comm)

    # }}}

    # {{{ construct local multipoles

    mpole_exps = wrangler.form_multipoles(
            local_traversal.level_start_source_box_nrs,
            local_traversal.source_boxes,
            local_source_weights)

    # }}}

    # {{{ propagate multipoles upward

    wrangler.coarsen_multipoles(
            local_traversal.level_start_source_parent_box_nrs,
            local_traversal.source_parent_boxes,
            mpole_exps)

    # }}}

    # {{{ direct evaluation from neighbor source boxes ("list 1")

    non_qbx_potentials = wrangler.eval_direct(
            global_traversal.target_boxes,
            global_traversal.neighbor_source_boxes_starts,
            global_traversal.neighbor_source_boxes_lists,
            local_source_weights)

    # }}}

    return None

# }}}
