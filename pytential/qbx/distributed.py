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
    "qbx_center_to_target_box": 5,
    "center_to_tree_targets": 6,
    "qbx_targets": 7,
    "non_qbx_potentials": 8,
    "qbx_potentials": 9
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

    def eval_qbx_expansions(self, qbx_expansions):
        geo_data = self.geo_data
        ctt = geo_data.center_to_tree_targets()
        global_qbx_centers = geo_data.global_qbx_centers()
        qbx_centers = geo_data.centers()
        qbx_radii = geo_data.expansion_radii()

        from pytools.obj_array import make_obj_array
        output = make_obj_array([np.zeros(len(ctt.lists), self.dtype)
                                 for k in self.outputs])

        all_targets = geo_data.qbx_targets()

        taeval = self.get_expn_eval_routine("ta")

        for isrc_center, src_icenter in enumerate(global_qbx_centers):
            for icenter_tgt in range(
                    ctt.starts[src_icenter],
                    ctt.starts[src_icenter+1]):

                center_itgt = ctt.lists[icenter_tgt]

                center = qbx_centers[:, src_icenter]

                pot, grad = taeval(
                        rscale=qbx_radii[src_icenter],
                        center=center,
                        expn=qbx_expansions[src_icenter].T,
                        ztarg=all_targets[:, center_itgt],
                        **self.kernel_kwargs)

                self.add_potgrad_onto_output(output, center_itgt, pot, grad)

        return output

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
            center_to_tree_targets = geo_data.center_to_tree_targets()

            qbx_center_to_target_box_source_level = np.empty(
                (nlevels,), dtype=object)
            for level in range(nlevels):
                qbx_center_to_target_box_source_level[level] = (
                    geo_data.qbx_center_to_target_box_source_level(level))

        else:  # worker process
            traversal = None

        # {{{ Distribute _from_sep_smaller_min_nsources_cumul

        if current_rank == 0:
            trav_param = {
                "_from_sep_smaller_min_nsources_cumul":
                    geo_data.geo_data.lpot_source.
                    _from_sep_smaller_min_nsources_cumul,
                "well_sep_is_n_away":
                    geo_data.geo_data.code_getter.build_traversal.well_sep_is_n_away,
                "from_sep_smaller_crit":
                    geo_data.geo_data.code_getter.build_traversal.
                    from_sep_smaller_crit
            }
        else:
            trav_param = None

        trav_param = comm.bcast(trav_param, root=0)

        # }}}

        from boxtree.distributed import generate_local_tree
        self.local_tree, self.local_data, self.box_bounding_box, knls = \
            generate_local_tree(traversal)

        from boxtree.distributed import generate_local_travs
        self.trav_local, self.trav_global = generate_local_travs(
            self.local_tree, self.box_bounding_box, comm=comm,
            well_sep_is_n_away=trav_param["well_sep_is_n_away"],
            from_sep_smaller_crit=trav_param["from_sep_smaller_crit"],
            _from_sep_smaller_min_nsources_cumul=trav_param[
                "_from_sep_smaller_min_nsources_cumul"],
            merge_close_lists=True
        )

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
            self.particle_mask = np.empty((total_rank,), dtype=object)

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
                self.particle_mask[irank] = particle_mask
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

        # {{{ Distribute other useful fields of geo_data

        if current_rank == 0:
            local_global_qbx_centers = np.empty((total_rank,), dtype=object)
            local_centers = np.empty((total_rank,), dtype=object)
            local_expansion_radii = np.empty((total_rank,), dtype=object)
            local_qbx_center_to_target_box = np.empty((total_rank,), dtype=object)
            local_center_to_tree_targets = np.empty((total_rank,), dtype=object)
            local_qbx_targets = np.empty((total_rank,), dtype=object)

            reqs_centers = np.empty((total_rank,), dtype=object)
            reqs_global_qbx_centers = np.empty((total_rank,), dtype=object)
            reqs_expansion_radii = np.empty((total_rank,), dtype=object)
            reqs_qbx_center_to_target_box = np.empty((total_rank,), dtype=object)
            reqs_center_to_tree_targets = np.empty((total_rank,), dtype=object)
            reqs_qbx_targets = np.empty((total_rank,), dtype=object)
            self.qbx_target_mask = np.empty((total_rank,), dtype=object)

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

                # {{{ Distribute local_qbx_targets and center_to_tree_targets

                starts = center_to_tree_targets.starts
                lists = center_to_tree_targets.lists
                local_starts = np.empty((nlocal_centers + 1,), dtype=starts.dtype)
                local_lists = np.empty(lists.shape, dtype=lists.dtype)

                qbx_target_mask = np.zeros((tree.ntargets,), dtype=bool)
                current_start = 0  # index into local_lists
                ilocal_center = 0
                local_starts[0] = 0

                for icenter in range(ncenters):
                    if not centers_mask[icenter]:
                        continue

                    current_center_targets = lists[
                        starts[icenter]:starts[icenter + 1]]
                    qbx_target_mask[current_center_targets] = True
                    current_stop = \
                        current_start + starts[icenter + 1] - starts[icenter]
                    local_starts[ilocal_center + 1] = current_stop
                    local_lists[current_start:current_stop] = \
                        lists[starts[icenter]:starts[icenter + 1]]

                    current_start = current_stop
                    ilocal_center += 1

                self.qbx_target_mask[irank] = qbx_target_mask

                local_lists = local_lists[:current_start]

                qbx_target_scan = np.empty((tree.ntargets + 1,), dtype=lists.dtype)
                qbx_target_scan[0] = 0
                qbx_target_scan[1:] = np.cumsum(qbx_target_mask.astype(lists.dtype))
                nlocal_qbx_target = qbx_target_scan[-1]

                local_qbx_targets[irank] = np.empty(
                    (tree.dimensions, nlocal_qbx_target),
                    dtype=tree.targets[0].dtype
                )
                for idim in range(tree.dimensions):
                    local_qbx_targets[irank][idim, :] = \
                        tree.targets[idim][qbx_target_mask]
                reqs_qbx_targets[irank] = comm.isend(
                    local_qbx_targets[irank],
                    dest=irank,
                    tag=MPITags["qbx_targets"]
                )

                local_lists = qbx_target_scan[local_lists]
                local_center_to_tree_targets[irank] = {
                    "starts": local_starts,
                    "lists": local_lists
                }
                reqs_center_to_tree_targets[irank] = comm.isend(
                    local_center_to_tree_targets[irank],
                    dest=irank,
                    tag=MPITags["center_to_tree_targets"])

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

            for irank in range(1, total_rank):
                reqs_center_to_tree_targets[irank].wait()
            local_center_to_tree_targets = local_center_to_tree_targets[0]

            for irank in range(1, total_rank):
                reqs_qbx_targets[irank].wait()
            local_qbx_targets = local_qbx_targets[0]

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
            local_center_to_tree_targets = comm.recv(
                source=0, tag=MPITags["center_to_tree_targets"]
            )
            local_qbx_targets = comm.recv(
                source=0, tag=MPITags["qbx_targets"]
            )

        self._local_centers = local_centers
        self._global_qbx_centers = local_global_qbx_centers
        self._expansion_radii = local_expansion_radii
        self._qbx_targets = local_qbx_targets

        # Transform local_qbx_center_to_target_box to target_boxes indexing
        global_boxes_to_target_boxes = np.ones(
            (self.local_tree.nboxes,), dtype=self.local_tree.particle_id_dtype)
        # make sure accessing invalid position raises an error
        global_boxes_to_target_boxes *= -1
        global_boxes_to_target_boxes[self.trav_global.target_boxes] = \
            np.arange(self.trav_global.target_boxes.shape[0])
        self._local_qbx_center_to_target_box = \
            global_boxes_to_target_boxes[local_qbx_center_to_target_box]

        from pytential.qbx.geometry import CenterToTargetList
        self._local_center_to_tree_targets = CenterToTargetList(
            starts=local_center_to_tree_targets["starts"],
            lists=local_center_to_tree_targets["lists"]
        )

        # }}}

        # {{{ Construct qbx_center_to_target_box_source_level

        # This is modified from pytential.geometry.QBXFMMGeometryData.
        # qbx_center_to_target_box_source_level but on host using Numpy instead of
        # PyOpenCL.

        traversal = self.traversal()
        qbx_center_to_target_box = self.qbx_center_to_target_box()
        tree = traversal.tree

        self._qbx_center_to_target_box_source_level = np.empty(
            (tree.nlevels,), dtype=object)

        for source_level in range(tree.nlevels):
            sep_smaller = traversal.from_sep_smaller_by_level[source_level]

            target_box_to_target_box_source_level = np.empty(
                len(traversal.target_boxes),
                dtype=tree.box_id_dtype
            )
            target_box_to_target_box_source_level.fill(-1)
            target_box_to_target_box_source_level[sep_smaller.nonempty_indices] = (
                np.arange(sep_smaller.num_nonempty_lists,
                          dtype=tree.box_id_dtype)
            )

            self._qbx_center_to_target_box_source_level[source_level] = (
                target_box_to_target_box_source_level[
                    qbx_center_to_target_box
                ]
            )

        # }}}

    def non_qbx_box_target_lists(self):
        return self._non_qbx_box_target_lists

    def traversal(self):
        return self.trav_global

    def tree(self):
        return self.traversal().tree

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

    def center_to_tree_targets(self):
        return self._local_center_to_tree_targets

    def qbx_targets(self):
        return self._qbx_targets

    def qbx_center_to_target_box_source_level(self, source_level):
        return self._qbx_center_to_target_box_source_level[source_level]

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

    # {{{ Communicate mpoles

    from boxtree.distributed import communicate_mpoles

    if _communicate_mpoles_via_allreduce:
        mpole_exps_all = np.zeros_like(mpole_exps)
        comm.Allreduce(mpole_exps, mpole_exps_all)
        mpole_exps = mpole_exps_all
    else:
        communicate_mpoles(wrangler, comm, local_traversal, mpole_exps)

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

    qbx_expansions = qbx_expansions + \
        wrangler.translate_box_multipoles_to_qbx_local(mpole_exps)

    qbx_expansions = qbx_expansions + \
        wrangler.translate_box_local_to_qbx_local(local_exps)

    qbx_potentials = wrangler.eval_qbx_expansions(qbx_expansions)

    # }}}

    if current_rank != 0:  # worker process
        comm.send(non_qbx_potentials, dest=0, tag=MPITags["non_qbx_potentials"])
        comm.send(qbx_potentials, dest=0, tag=MPITags["qbx_potentials"])
        return None

    else:  # master process

        all_potentials_in_tree_order = root_wrangler.full_output_zeros()

        nqbtl = root_wrangler.geo_data.non_qbx_box_target_lists()

        from pytools.obj_array import make_obj_array
        non_qbx_potentials_all_rank = make_obj_array([
            np.zeros(nqbtl.nfiltered_targets, root_wrangler.dtype)
            for k in root_wrangler.outputs]
        )

        for irank in range(total_rank):

            if irank == 0:
                non_qbx_potentials_cur_rank = non_qbx_potentials
            else:
                non_qbx_potentials_cur_rank = comm.recv(
                    source=irank, tag=MPITags["non_qbx_potentials"])

            for idim in range(len(root_wrangler.outputs)):
                non_qbx_potentials_all_rank[idim][
                    distributed_geo_data.particle_mask[irank]
                ] = non_qbx_potentials_cur_rank[idim]

        for ap_i, nqp_i in zip(
                all_potentials_in_tree_order, non_qbx_potentials_all_rank):
            ap_i[nqbtl.unfiltered_from_filtered_target_indices] = nqp_i

        for irank in range(total_rank):

            if irank == 0:
                qbx_potentials_cur_rank = qbx_potentials
            else:
                qbx_potentials_cur_rank = comm.recv(
                    source=irank, tag=MPITags["qbx_potentials"]
                )

            for idim in range(len(root_wrangler.outputs)):
                all_potentials_in_tree_order[idim][
                    distributed_geo_data.qbx_target_mask[irank]
                ] = qbx_potentials_cur_rank[idim]

        def reorder_and_finalize_potentials(x):
            # "finalize" gives host FMMs (like FMMlib) a chance to turn the
            # potential back into a CL array.
            return root_wrangler.finalize_potentials(
                x[root_wrangler.tree.sorted_target_ids])

        from pytools.obj_array import with_object_array_or_scalar
        result = with_object_array_or_scalar(
            reorder_and_finalize_potentials, all_potentials_in_tree_order)

        return result

# }}}
