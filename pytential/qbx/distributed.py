from pytential.qbx.fmmlib import QBXFMMLibExpansionWrangler
from boxtree.distributed import DistributedFMMLibExpansionWrangler, queue
from boxtree.tree import FilteredTargetListsInTreeOrder
from mpi4py import MPI
import numpy as np

import pyopencl as cl

# {{{ MPITags used in this module

MPITags = {
    "non_qbx_box_target_lists": 0,
    "global_qbx_centers": 1,
    "centers": 2,
    "dipole_vec": 3,
    "expansion_radii": 4,
    "qbx_center_to_target_box": 5
}

# }}}


# {{{ Expansion Wrangler

class QBXDistributedFMMLibExpansionWrangler(
        QBXFMMLibExpansionWrangler, DistributedFMMLibExpansionWrangler):

    @classmethod
    def distribute(cls, wrangler, distributed_geo_data, comm=MPI.COMM_WORLD):
        current_rank = comm.Get_rank()
        total_rank = comm.Get_size()

        if wrangler is not None:  # master process
            import copy
            distributed_wrangler = copy.copy(wrangler)
            distributed_wrangler.queue = None
            distributed_wrangler.geo_data = None
            distributed_wrangler.code = None
            distributed_wrangler.tree = None
            distributed_wrangler.dipole_vec = None
            distributed_wrangler.__class__ = cls
        else:  # worker process
            distributed_wrangler = None

        distributed_wrangler = comm.bcast(distributed_wrangler, root=0)
        distributed_wrangler.tree = distributed_geo_data.local_tree
        distributed_wrangler.geo_data = distributed_geo_data

        # {{{ Distribute dipole_vec

        if current_rank == 0:
            reqs_dipole_vec = np.empty((total_rank,), dtype=object)
            local_dipole_vec = np.empty((total_rank,), dtype=object)
            for irank in range(total_rank):
                src_mask = distributed_geo_data.local_data[irank]["src_mask"].get()
                local_dipole_vec[irank] = \
                    wrangler.dipole_vec[:, src_mask.astype(bool)]
                reqs_dipole_vec[irank] = comm.isend(
                    local_dipole_vec[irank],
                    dest=irank,
                    tag=MPITags["dipole_vec"]
                )

            for irank in range(1, total_rank):
                reqs_dipole_vec[irank].wait()
            distributed_wrangler.dipole_vec = local_dipole_vec[0]
        else:
            distributed_wrangler.dipole_vec = comm.recv(
                source=0, tag=MPITags["dipole_vec"])

        # }}}

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
            nlevels = tree.nlevels

            ncenters = geo_data.ncenters
            centers = geo_data.centers()
            expansion_radii = geo_data.expansion_radii()
            global_qbx_centers = geo_data.global_qbx_centers()
            qbx_center_to_target_box = geo_data.qbx_center_to_target_box()
            non_qbx_box_target_lists = geo_data.non_qbx_box_target_lists()
            # center_to_tree_targets = geo_data.center_to_tree_targets()

            qbx_center_to_target_box_source_level = np.empty(
                (nlevels,), dtype=object)
            for level in range(nlevels):
                qbx_center_to_target_box_source_level[level] = (
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

                if irank != 0:
                    reqs[irank] = comm.isend(
                        local_non_qbx_box_target_lists[irank],
                        dest=irank,
                        tag=MPITags["non_qbx_box_target_lists"]
                    )

            for irank in range(1, total_rank):
                reqs[irank].wait()
        if current_rank == 0:
            local_non_qbx_box_target_lists = local_non_qbx_box_target_lists[0]
        else:
            local_non_qbx_box_target_lists = comm.recv(
                source=0, tag=MPITags["non_qbx_box_target_lists"])

        self._non_qbx_box_target_lists = FilteredTargetListsInTreeOrder(
            nfiltered_targets=local_non_qbx_box_target_lists["nfiltered_targets"],
            box_target_starts=local_non_qbx_box_target_lists["box_target_starts"],
            box_target_counts_nonchild=local_non_qbx_box_target_lists[
                "box_target_counts_nonchild"],
            targets=local_non_qbx_box_target_lists["targets"],
            unfiltered_from_filtered_target_indices=None
        )

        # }}}

        # {{{ Distribute global_qbx_centers, centers and expansion_radii

        if current_rank == 0:
            local_global_qbx_centers = np.empty((total_rank,), dtype=object)
            local_centers = np.empty((total_rank,), dtype=object)
            local_expansion_radii = np.empty((total_rank,), dtype=object)
            local_qbx_center_to_target_box = np.empty((total_rank,), dtype=object)

            reqs_centers = np.empty((total_rank,), dtype=object)
            reqs_global_qbx_centers = np.empty((total_rank,), dtype=object)
            reqs_expansion_radii = np.empty((total_rank,), dtype=object)
            reqs_qbx_center_to_target_box = np.empty((total_rank,), dtype=object)

            for irank in range(total_rank):
                tgt_mask = self.local_data[irank]["tgt_mask"].get().astype(bool)
                tgt_mask_user_order = tgt_mask[tree.sorted_target_ids]
                centers_mask = tgt_mask_user_order[:ncenters]
                centers_scan = np.empty(
                    (ncenters + 1,), dtype=tree.particle_id_dtype)
                centers_scan[1:] = np.cumsum(
                    centers_mask.astype(tree.particle_id_dtype))
                centers_scan[0] = 0

                # {{{ Distribute centers

                nlocal_centers = np.sum(centers_mask.astype(np.int32))
                centers_dims = centers.shape[0]
                local_centers[irank] = np.empty((centers_dims, nlocal_centers),
                                                dtype=centers[0].dtype)
                for idims in range(centers_dims):
                    local_centers[irank][idims][:] = centers[idims][centers_mask]

                if irank != 0:
                    reqs_centers[irank] = comm.isend(
                        local_centers[irank],
                        dest=irank,
                        tag=MPITags["centers"]
                    )

                # }}}

                # {{{ Distribute global_qbx_centers

                local_global_qbx_centers[irank] = centers_scan[
                    global_qbx_centers[centers_mask[global_qbx_centers]]]

                if irank != 0:
                    reqs_global_qbx_centers[irank] = comm.isend(
                        local_global_qbx_centers[irank],
                        dest=irank,
                        tag=MPITags["global_qbx_centers"]
                    )

                # }}}

                # {{{ Distribute expansion_radii

                local_expansion_radii[irank] = expansion_radii[centers_mask]
                if irank != 0:
                    reqs_expansion_radii[irank] = comm.isend(
                        local_expansion_radii[irank],
                        dest=irank,
                        tag=MPITags["expansion_radii"]
                    )

                # }}}

                # {{{ Distribute qbx_center_to_target_box

                # Note: The code transforms qbx_center_to_target_box to global box
                # indexing from target_boxes before transmission. Each process is
                # expected to transform back to target_boxes indexing based its own
                # traversal object.

                local_qbx_center_to_target_box[irank] = \
                    traversal.target_boxes[qbx_center_to_target_box[centers_mask]]
                if irank != 0:
                    reqs_qbx_center_to_target_box[irank] = comm.isend(
                        local_qbx_center_to_target_box[irank],
                        dest=irank,
                        tag=MPITags["qbx_center_to_target_box"]
                    )

                # }}}

            for irank in range(1, total_rank):
                reqs_centers[irank].wait()
            local_centers = local_centers[0]

            for irank in range(1, total_rank):
                reqs_global_qbx_centers[irank].wait()
            local_global_qbx_centers = local_global_qbx_centers[0]

            for irank in range(1, total_rank):
                reqs_expansion_radii[irank].wait()
            local_expansion_radii = local_expansion_radii[0]

            for irank in range(1, total_rank):
                reqs_qbx_center_to_target_box[irank].wait()
            local_qbx_center_to_target_box = local_qbx_center_to_target_box[0]

        else:
            local_centers = comm.recv(
                source=0, tag=MPITags["centers"])
            local_global_qbx_centers = comm.recv(
                source=0, tag=MPITags["global_qbx_centers"])
            local_expansion_radii = comm.recv(
                source=0, tag=MPITags["expansion_radii"])
            local_qbx_center_to_target_box = comm.recv(
                source=0, tag=MPITags["qbx_center_to_target_box"]
            )

        self._local_centers = local_centers
        self._global_qbx_centers = local_global_qbx_centers
        self._expansion_radii = local_expansion_radii

        # Transform local_qbx_center_to_target_box to target_boxes indexing
        global_boxes_to_target_boxes = np.ones(
            (self.local_tree.nboxes,), dtype=self.local_tree.particle_id_dtype)
        # make sure accessing invalid position raises an error
        global_boxes_to_target_boxes *= -1
        global_boxes_to_target_boxes[self.trav_global.target_boxes] = \
            np.arange(self.trav_global.target_boxes.shape[0])
        self._local_qbx_center_to_target_box = \
            global_boxes_to_target_boxes[local_qbx_center_to_target_box]

        # }}}

    def non_qbx_box_target_lists(self):
        return self._non_qbx_box_target_lists

    def traversal(self):
        return self.trav_global

    def centers(self):
        return self._local_centers

    @property
    def ncenters(self):
        return self._local_centers.shape[1]

    def global_qbx_centers(self):
        return self._global_qbx_centers

    def expansion_radii(self):
        return self._expansion_radii

    def qbx_center_to_target_box(self):
        return self._local_qbx_center_to_target_box

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

    # {{{ translate separated siblings' ("list 2") mpoles to local

    local_exps = wrangler.multipole_to_local(
        global_traversal.level_start_target_or_target_parent_box_nrs,
        global_traversal.target_or_target_parent_boxes,
        global_traversal.from_sep_siblings_starts,
        global_traversal.from_sep_siblings_lists,
        mpole_exps)

    # }}}

    # {{{ evaluate sep. smaller mpoles ("list 3") at particles

    # (the point of aiming this stage at particles is specifically to keep its
    # contribution *out* of the downward-propagating local expansions)

    non_qbx_potentials = non_qbx_potentials + wrangler.eval_multipoles(
        global_traversal.target_boxes_sep_smaller_by_source_level,
        global_traversal.from_sep_smaller_by_level,
        mpole_exps)

    # assert that list 3 close has been merged into list 1
    # assert global_traversal.from_sep_close_smaller_starts is None
    if global_traversal.from_sep_close_smaller_starts is not None:
        non_qbx_potentials = non_qbx_potentials + wrangler.eval_direct(
            global_traversal.target_boxes,
            global_traversal.from_sep_close_smaller_starts,
            global_traversal.from_sep_close_smaller_lists,
            local_source_weights)

    # }}}

    # {{{ form locals for separated bigger source boxes ("list 4")

    local_exps = local_exps + wrangler.form_locals(
        global_traversal.level_start_target_or_target_parent_box_nrs,
        global_traversal.target_or_target_parent_boxes,
        global_traversal.from_sep_bigger_starts,
        global_traversal.from_sep_bigger_lists,
        local_source_weights)

    if global_traversal.from_sep_close_bigger_starts is not None:
        non_qbx_potentials = non_qbx_potentials + wrangler.eval_direct(
            global_traversal.target_or_target_parent_boxes,
            global_traversal.from_sep_close_bigger_starts,
            global_traversal.from_sep_close_bigger_lists,
            local_source_weights)

    # }}}

    # {{{ propagate local_exps downward

    wrangler.refine_locals(
        global_traversal.level_start_target_or_target_parent_box_nrs,
        global_traversal.target_or_target_parent_boxes,
        local_exps)

    # }}}

    # {{{ evaluate locals

    non_qbx_potentials = non_qbx_potentials + wrangler.eval_locals(
        global_traversal.level_start_target_box_nrs,
        global_traversal.target_boxes,
        local_exps)

    # }}}

    # {{{ wrangle qbx expansions

    qbx_expansions = wrangler.form_global_qbx_locals(local_source_weights)

    # }}}

    return None

# }}}
