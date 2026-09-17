"""Conservative RGB-only ray search inside an existing visual hull.

This is deliberately a one-way experiment: each initial surface voxel may be
removed only when a deeper occupied voxel on its reference-camera ray has
substantially stronger two-source warped-NCC support.  Ambiguous voxels stay.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "srp" / "stage1_hull"))

import photo_carve_warp as warp  # noqa: E402


def surface(occupancy):
    return occupancy & ~ndimage.binary_erosion(
        occupancy, ndimage.generate_binary_structure(3, 1)
    )


def _score_points(points, normals, views, patch_radius=2, tilt_deg=20.0,
                  k_src=4, min_src=2, batch_size=768):
    """Best second-source NCC for fixed 3D candidates.

    Unlike carve_warp(), candidates are not required to be the current hull
    z-buffer front layer.  That is necessary to compare an outer hull layer
    with a possible true layer behind it.  Full foreground patches are still
    mandatory in reference and source images.
    """
    count = len(points)
    scores = np.full(count, np.nan, np.float32)
    if count == 0:
        return scores

    offsets = np.arange(-patch_radius, patch_radius + 1)
    dv, du = np.meshgrid(offsets, offsets, indexing="ij")
    du = du.ravel().astype(np.float64)
    dv = dv.ravel().astype(np.float64)
    pixel_count = len(du)
    view_count = len(views)

    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        X0 = points[start:stop]
        n0 = normals[start:stop]
        size = len(X0)

        pu = np.empty((view_count, size))
        pv = np.empty((view_count, size))
        visible = np.zeros((view_count, size), bool)
        facing = np.zeros((view_count, size))
        directions = np.empty((view_count, size, 3))
        for vi, view in enumerate(views):
            X = X0 @ view["Rwc"].T + view["t"]
            z = X[:, 2]
            front = z > 1e-6
            zz = np.where(front, z, 1.0)
            u = view["K"][0, 0] * X[:, 0] / zz + view["K"][0, 2]
            v = view["K"][1, 1] * X[:, 1] / zz + view["K"][1, 2]
            in_bounds = front & (u >= 0) & (u < view["W"]) & (v >= 0) & (v < view["H"])
            to_camera = view["Cw"][None, :] - X0
            to_camera /= np.linalg.norm(to_camera, axis=1, keepdims=True) + 1e-9
            pu[vi] = u
            pv[vi] = v
            directions[vi] = to_camera
            facing[vi] = (n0 * to_camera).sum(1)
            visible[vi] = in_bounds & (facing[vi] > 0.1)

        face_masked = np.where(visible, facing, -9.0)
        ref = face_masked.argmax(0)
        has_ref = face_masked.max(0) > -1
        normal_candidates = warp.candidate_normals(n0, tilt_deg)
        normal_count = len(normal_candidates)
        ncc_count = np.zeros((normal_count, size), np.int16)
        ncc_max = np.full((normal_count, size), -np.inf)
        ncc_second = np.full((normal_count, size), -np.inf)

        for r in range(view_count):
            selected = np.flatnonzero(has_ref & (ref == r))
            if len(selected) == 0:
                continue
            dref = directions[r, selected]
            alignment = np.full((view_count, len(selected)), -9.0)
            for s in range(view_count):
                if s == r:
                    continue
                ok = visible[s, selected]
                alignment[s, ok] = (directions[s, selected][ok] * dref[ok]).sum(1)
            order = np.argsort(-alignment, axis=0)[:k_src]

            ur = pu[r, selected]
            vr = pv[r, selected]
            ref_u = np.rint(ur[:, None] + du[None, :]).astype(int)
            ref_v = np.rint(vr[:, None] + dv[None, :]).astype(int)
            ref_view = views[r]
            ref_ok = (ref_u >= 0) & (ref_u < ref_view["W"]) & (ref_v >= 0) & (ref_v < ref_view["H"])
            ref_u = np.clip(ref_u, 0, ref_view["W"] - 1)
            ref_v = np.clip(ref_v, 0, ref_view["H"] - 1)
            ref_fg = ref_ok & ref_view["fg"][ref_v, ref_u]
            ref_patch = ref_view["gray"][ref_v, ref_u]

            Rr = ref_view["Rwc"]
            tr = ref_view["t"]
            Xr = X0[selected] @ Rr.T + tr
            normals_r = np.einsum("cmj,jk->cmk", normal_candidates[:, selected], Rr.T)
            plane_d = (normals_r * Xr[None]).sum(-1)
            ref_hom = np.stack([
                ur[:, None] + du[None, :],
                vr[:, None] + dv[None, :],
                np.ones((len(selected), pixel_count)),
            ], axis=-1)

            for rank in range(k_src):
                source_choices = order[rank]
                for s in np.unique(source_choices):
                    if s == r:
                        continue
                    local = np.flatnonzero(
                        (source_choices == s) & (alignment[s] > -1)
                    )
                    if len(local) == 0:
                        continue
                    source = views[s]
                    Rsr = source["Rwc"] @ Rr.T
                    tsr = source["t"] - source["Rwc"] @ Rr.T @ tr
                    ref_points = ref_hom[local]
                    global_indices = selected[local]
                    for ci in range(normal_count):
                        nr = normals_r[ci, local]
                        dr = plane_d[ci, local]
                        nondegenerate = np.abs(dr) > 1e-6
                        if not nondegenerate.any():
                            continue
                        matrix = Rsr[None] + (tsr[None, :, None] * nr[:, None, :]) / dr[:, None, None]
                        H = np.einsum("ij,njk,kl->nil", source["K"], matrix, ref_view["Kinv"])
                        warped = np.einsum("nij,npj->npi", H, ref_points)
                        su = warped[:, :, 0] / warped[:, :, 2]
                        sv = warped[:, :, 1] / warped[:, :, 2]
                        source_patch, source_ok = warp.bilinear(source["gray"], su, sv)
                        good = source_ok & ref_ok[local] & ref_fg[local]
                        good &= warp.mask_samples(source["fg"], su, sv)
                        patch_ok = good.all(1) & nondegenerate
                        if not patch_ok.any():
                            continue
                        ncc, textured = warp.ncc_rows(
                            ref_patch[local][patch_ok], source_patch[patch_ok]
                        )
                        valid = textured & ~np.isnan(ncc)
                        if not valid.any():
                            continue
                        target = global_indices[patch_ok][valid]
                        value = ncc[valid]
                        ncc_count[ci, target] += 1
                        previous = ncc_max[ci, target].copy()
                        ncc_second[ci, target] = np.where(
                            value > previous,
                            previous,
                            np.maximum(ncc_second[ci, target], value),
                        )
                        ncc_max[ci, target] = np.maximum(previous, value)

        usable = ncc_count >= min_src
        second = np.where(usable, ncc_second, np.nan)
        best = np.full(size, np.nan)
        has_score = np.isfinite(second).any(0)
        best[has_score] = np.max(second[:, has_score], axis=0)
        scores[start:stop] = best.astype(np.float32)
    return scores


def _reference_directions(points, normals, views):
    """Return the most front-facing camera-to-point direction for each point."""
    out = np.full((len(points), 3), np.nan)
    best = np.full(len(points), -np.inf)
    for view in views:
        direction = points - view["Cw"][None, :]
        direction /= np.linalg.norm(direction, axis=1, keepdims=True) + 1e-9
        toward_camera = -direction
        value = (normals * toward_camera).sum(1)
        take = value > best
        out[take] = direction[take]
        best[take] = value[take]
    return out


def _ray_layers(occupancy, grid_min, voxel_size, indices, directions, max_steps):
    """Occupied voxel indices on each outward surface point's inward ray."""
    shape = np.asarray(occupancy.shape)
    count = len(indices)
    layers = np.full((count, max_steps + 1, 3), -1, np.int32)
    layers[:, 0] = indices
    base = grid_min + (indices + 0.5) * voxel_size
    for step in range(1, max_steps + 1):
        point = base + directions * (step * voxel_size)
        candidate = np.floor((point - grid_min) / voxel_size).astype(np.int32)
        inside = ((candidate >= 0) & (candidate < shape)).all(1)
        valid = inside.copy()
        valid[inside] &= occupancy[tuple(candidate[inside].T)]
        layers[valid, step] = candidate[valid]
    return layers


def _connected_keep(mask, minimum):
    labels, number = ndimage.label(mask, ndimage.generate_binary_structure(3, 1))
    if number == 0:
        return mask
    sizes = ndimage.sum(np.ones_like(labels), labels, index=np.arange(1, number + 1))
    keep = np.flatnonzero(sizes >= minimum) + 1
    return np.isin(labels, keep)


def ray_search_carve(occupancy, grid_min, voxel_size, views, max_steps=3,
                     ncc_reject=0.10, ncc_accept=0.25, ncc_margin=0.10, min_cluster=3):
    """Select a stronger deeper ray candidate, otherwise retain the outer layer."""
    occupancy = occupancy.copy()
    outer = surface(occupancy)
    indices = np.argwhere(outer)
    if len(indices) == 0:
        return occupancy, {"surface_voxels": 0, "moved": 0, "removed": 0}

    normals = warp.voxel_normals(occupancy)[tuple(indices.T)]
    points = grid_min + (indices + 0.5) * voxel_size
    directions = _reference_directions(points, normals, views)
    layers = _ray_layers(occupancy, grid_min, voxel_size, indices, directions, max_steps)
    valid = (layers >= 0).all(2)
    candidate_indices = layers[valid]
    candidate_points = grid_min + (candidate_indices + 0.5) * voxel_size
    candidate_normals = np.repeat(normals, valid.sum(1), axis=0)
    candidate_scores = _score_points(candidate_points, candidate_normals, views)
    scores = np.full(valid.shape, np.nan, np.float32)
    scores[valid] = candidate_scores

    outer_score = scores[:, 0]
    best_layer = np.nanargmax(np.where(np.isfinite(scores), scores, -np.inf), axis=1)
    best_score = scores[np.arange(len(indices)), best_layer]
    move = (
        (best_layer > 0)
        & np.isfinite(outer_score)
        & np.isfinite(best_score)
        & (best_score >= ncc_accept)
        & (best_score >= outer_score + ncc_margin)
        & (outer_score < ncc_reject)
    )

    remove = np.zeros_like(occupancy, bool)
    for row in np.flatnonzero(move):
        for step in range(best_layer[row]):
            cell = layers[row, step]
            if (cell >= 0).all():
                remove[tuple(cell)] = True
    remove = _connected_keep(remove, min_cluster)
    occupancy[remove] = False
    stats = {
        "surface_voxels": int(len(indices)),
        "candidates": int(valid.sum()),
        "moved": int(move.sum()),
        "removed": int(remove.sum()),
        "median_outer_ncc": float(np.nanmedian(outer_score)) if np.isfinite(outer_score).any() else None,
        "median_best_ncc": float(np.nanmedian(best_score)) if np.isfinite(best_score).any() else None,
        "ncc_reject": float(ncc_reject),
    }
    return occupancy, stats
