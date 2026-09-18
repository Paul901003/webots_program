#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""Sparse hull-bounded MVS with independent reference-depth consistency.

Each usable outer-hull pixel has a finite set of occupied candidates on its
own camera ray.  NCC is evaluated with that camera fixed as reference.  The
best candidate becomes a sparse depth-map entry.  A proposed deeper surface is
accepted only when two *other* reference depth maps independently select a
nearby 3-D point after reprojection.  This is deliberately a diagnostic
experiment and does not feed Stage 2.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[4]
os.environ.setdefault("SAM_ROOT", str(REPO / "data" / "eval" / "mobilesamv2_fast"))
os.environ.setdefault("CAPTURES_ROOT", str(REPO / "data" / "captures_fast"))
sys.path.insert(0, str(REPO / "srp" / "io"))
sys.path.insert(0, str(REPO / "srp" / "stage1_hull"))
sys.path.insert(0, str(REPO / "srp" / "stage1_hull" / "experiments" / "photo_ray_search_20260917"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import photo_carve_warp as warp  # noqa: E402
import interval_candidates as IC  # noqa: E402
from run_interval_ray_search import connected_keep, load_views, removal_rays  # noqa: E402


EVAL = REPO / "data" / "eval"


def _forced_reference_scores_cpu(points, normals, refs, views, patch_radius=2, tilt_deg=20.0,
                                 k_src=4, min_src=2, batch_size=768):
    """Best second-source NCC while keeping `refs[i]` fixed for every point."""
    result = np.full(len(points), np.nan, np.float32)
    offsets = np.arange(-patch_radius, patch_radius + 1)
    dv, du = np.meshgrid(offsets, offsets, indexing="ij")
    du = du.ravel().astype(np.float64)
    dv = dv.ravel().astype(np.float64)
    pixels = len(du)
    view_count = len(views)

    for start in range(0, len(points), batch_size):
        stop = min(start + batch_size, len(points))
        X0 = points[start:stop]
        n0 = normals[start:stop]
        ref_ids = refs[start:stop]
        size = len(X0)
        pu = np.empty((view_count, size))
        pv = np.empty((view_count, size))
        valid = np.zeros((view_count, size), bool)
        facing = np.zeros((view_count, size))
        directions = np.empty((view_count, size, 3))
        for vi, view in enumerate(views):
            X = X0 @ view["Rwc"].T + view["t"]
            z = X[:, 2]
            front = z > 1e-6
            zz = np.where(front, z, 1.0)
            u = view["K"][0, 0] * X[:, 0] / zz + view["K"][0, 2]
            v = view["K"][1, 1] * X[:, 1] / zz + view["K"][1, 2]
            to_camera = view["Cw"][None] - X0
            to_camera /= np.linalg.norm(to_camera, axis=1, keepdims=True) + 1e-9
            pu[vi], pv[vi] = u, v
            directions[vi] = to_camera
            facing[vi] = (n0 * to_camera).sum(1)
            valid[vi] = front & (u >= 0) & (u < view["W"]) & (v >= 0) & (v < view["H"]) & (facing[vi] > 0.1)

        normal_candidates = warp.candidate_normals(n0, tilt_deg)
        normal_count = len(normal_candidates)
        count = np.zeros((normal_count, size), np.int16)
        maximum = np.full((normal_count, size), -np.inf)
        second = np.full((normal_count, size), -np.inf)
        for ref in range(view_count):
            selected = np.flatnonzero((ref_ids == ref) & valid[ref])
            if len(selected) == 0:
                continue
            ref_view = views[ref]
            dref = directions[ref, selected]
            alignment = np.full((view_count, len(selected)), -9.0)
            for source in range(view_count):
                if source == ref:
                    continue
                ok = valid[source, selected]
                alignment[source, ok] = (directions[source, selected][ok] * dref[ok]).sum(1)
            order = np.argsort(-alignment, axis=0)[:k_src]
            ur, vr = pu[ref, selected], pv[ref, selected]
            ref_u = np.rint(ur[:, None] + du[None]).astype(int)
            ref_v = np.rint(vr[:, None] + dv[None]).astype(int)
            in_patch = (ref_u >= 0) & (ref_u < ref_view["W"]) & (ref_v >= 0) & (ref_v < ref_view["H"])
            ref_u = np.clip(ref_u, 0, ref_view["W"] - 1)
            ref_v = np.clip(ref_v, 0, ref_view["H"] - 1)
            ref_fg = in_patch & ref_view["fg"][ref_v, ref_u]
            ref_patch = ref_view["gray"][ref_v, ref_u]
            Xref = X0[selected] @ ref_view["Rwc"].T + ref_view["t"]
            normals_ref = np.einsum("cmj,jk->cmk", normal_candidates[:, selected], ref_view["Rwc"].T)
            plane_d = (normals_ref * Xref[None]).sum(-1)
            ref_hom = np.stack([ur[:, None] + du[None], vr[:, None] + dv[None], np.ones((len(selected), pixels))], axis=-1)

            for rank in range(k_src):
                choices = order[rank]
                for source in np.unique(choices):
                    if source == ref:
                        continue
                    local = np.flatnonzero((choices == source) & (alignment[source] > -1))
                    if len(local) == 0:
                        continue
                    source_view = views[source]
                    Rsr = source_view["Rwc"] @ ref_view["Rwc"].T
                    tsr = source_view["t"] - source_view["Rwc"] @ ref_view["Rwc"].T @ ref_view["t"]
                    hom = ref_hom[local]
                    global_indices = selected[local]
                    for ci in range(normal_count):
                        nr = normals_ref[ci, local]
                        depth = plane_d[ci, local]
                        good_plane = np.abs(depth) > 1e-6
                        if not good_plane.any():
                            continue
                        matrix = Rsr[None] + (tsr[None, :, None] * nr[:, None, :]) / depth[:, None, None]
                        H = np.einsum("ij,njk,kl->nil", source_view["K"], matrix, ref_view["Kinv"])
                        warped = np.einsum("nij,npj->npi", H, hom)
                        su = warped[:, :, 0] / warped[:, :, 2]
                        sv = warped[:, :, 1] / warped[:, :, 2]
                        source_patch, source_ok = warp.bilinear(source_view["gray"], su, sv)
                        patch_ok = source_ok & in_patch[local] & ref_fg[local]
                        patch_ok &= warp.mask_samples(source_view["fg"], su, sv)
                        patch_ok = patch_ok.all(1) & good_plane
                        if not patch_ok.any():
                            continue
                        ncc, textured = warp.ncc_rows(ref_patch[local][patch_ok], source_patch[patch_ok])
                        accepted = textured & ~np.isnan(ncc)
                        if not accepted.any():
                            continue
                        target = global_indices[patch_ok][accepted]
                        value = ncc[accepted]
                        count[ci, target] += 1
                        previous = maximum[ci, target].copy()
                        second[ci, target] = np.where(value > previous, previous, np.maximum(second[ci, target], value))
                        maximum[ci, target] = np.maximum(previous, value)

        usable = count >= min_src
        best = np.full(size, np.nan, np.float32)
        values = np.where(usable, second, np.nan)
        has_value = np.isfinite(values).any(0)
        best[has_value] = np.nanmax(values[:, has_value], axis=0)
        result[start:stop] = best
    return result



def _torch_bilinear(gray, u, v):
    height, width = gray.shape
    u0 = torch.floor(u).to(torch.long)
    v0 = torch.floor(v).to(torch.long)
    du = u - u0
    dv = v - v0
    valid = (u0 >= 0) & (v0 >= 0) & (u0 + 1 < width) & (v0 + 1 < height)
    u0 = u0.clamp(0, width - 2)
    v0 = v0.clamp(0, height - 2)
    value = (gray[v0, u0] * (1 - du) * (1 - dv) +
             gray[v0, u0 + 1] * du * (1 - dv) +
             gray[v0 + 1, u0] * (1 - du) * dv +
             gray[v0 + 1, u0 + 1] * du * dv)
    return value, valid


def _torch_mask_samples(mask, u, v):
    height, width = mask.shape
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    ui = torch.round(u).to(torch.long).clamp(0, width - 1)
    vi = torch.round(v).to(torch.long).clamp(0, height - 1)
    return valid & mask[vi, ui]


def _torch_candidate_normals(n0, tilt_deg):
    if tilt_deg <= 0:
        return n0.unsqueeze(0)
    e = torch.zeros_like(n0)
    e[:, 2] = 1.0
    e[n0[:, 2].abs() > 0.9] = torch.tensor([1.0, 0.0, 0.0], device=n0.device, dtype=n0.dtype)
    t1 = torch.linalg.cross(n0, e)
    t1 = t1 / (torch.linalg.vector_norm(t1, dim=1, keepdim=True) + 1e-9)
    t2 = torch.linalg.cross(n0, t1)
    t2 = t2 / (torch.linalg.vector_norm(t2, dim=1, keepdim=True) + 1e-9)
    delta = np.tan(np.deg2rad(tilt_deg))
    offsets = [(0.0, 0.0)] + [(a, b) for a in (-delta, 0.0, delta)
                              for b in (-delta, 0.0, delta) if (a, b) != (0.0, 0.0)]
    candidates = []
    for a, b in offsets:
        normal = n0 + a * t1 + b * t2
        candidates.append(normal / (torch.linalg.vector_norm(normal, dim=1, keepdim=True) + 1e-9))
    return torch.stack(candidates, 0)


def _torch_ncc_rows(first, second):
    first = first - first.mean(1, keepdim=True)
    second = second - second.mean(1, keepdim=True)
    first_norm = torch.sqrt((first ** 2).sum(1))
    second_norm = torch.sqrt((second ** 2).sum(1))
    textured = (first_norm > 1e-2) & (second_norm > 1e-2)
    ncc = (first * second).sum(1) / (first_norm * second_norm + 1e-9)
    return ncc, textured


def _forced_reference_scores_cuda(points, normals, refs, views, patch_radius=2, tilt_deg=20.0,
                                  k_src=4, min_src=2, batch_size=768):
    """Same score as the CPU path, with projection, warp, and NCC on CUDA."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA scoring requested but torch.cuda.is_available() is false")
    device = torch.device("cuda")
    dtype = torch.float64
    gpu_views = [{
        "Rwc": torch.as_tensor(view["Rwc"], device=device, dtype=dtype),
        "t": torch.as_tensor(view["t"], device=device, dtype=dtype),
        "Cw": torch.as_tensor(view["Cw"], device=device, dtype=dtype),
        "K": torch.as_tensor(view["K"], device=device, dtype=dtype),
        "Kinv": torch.as_tensor(view["Kinv"], device=device, dtype=dtype),
        "gray": torch.as_tensor(view["gray"], device=device, dtype=dtype),
        "fg": torch.as_tensor(view["fg"], device=device, dtype=torch.bool),
        "W": view["W"], "H": view["H"],
    } for view in views]
    result = np.full(len(points), np.nan, np.float32)
    offsets = torch.arange(-patch_radius, patch_radius + 1, device=device, dtype=dtype)
    dv, du = torch.meshgrid(offsets, offsets, indexing="ij")
    du = du.reshape(-1)
    dv = dv.reshape(-1)
    pixels = len(du)
    view_count = len(gpu_views)

    for start in range(0, len(points), batch_size):
        stop = min(start + batch_size, len(points))
        X0 = torch.as_tensor(points[start:stop], device=device, dtype=dtype)
        n0 = torch.as_tensor(normals[start:stop], device=device, dtype=dtype)
        ref_ids = torch.as_tensor(refs[start:stop], device=device, dtype=torch.long)
        size = len(X0)
        pu = torch.empty((view_count, size), device=device, dtype=dtype)
        pv = torch.empty((view_count, size), device=device, dtype=dtype)
        valid = torch.zeros((view_count, size), device=device, dtype=torch.bool)
        directions = torch.empty((view_count, size, 3), device=device, dtype=dtype)
        for vi, view in enumerate(gpu_views):
            X = X0 @ view["Rwc"].T + view["t"]
            z = X[:, 2]
            front = z > 1e-6
            zz = torch.where(front, z, torch.ones_like(z))
            u = view["K"][0, 0] * X[:, 0] / zz + view["K"][0, 2]
            v = view["K"][1, 1] * X[:, 1] / zz + view["K"][1, 2]
            to_camera = view["Cw"].unsqueeze(0) - X0
            to_camera = to_camera / (torch.linalg.vector_norm(to_camera, dim=1, keepdim=True) + 1e-9)
            pu[vi], pv[vi] = u, v
            directions[vi] = to_camera
            facing = (n0 * to_camera).sum(1)
            valid[vi] = front & (u >= 0) & (u < view["W"]) & (v >= 0) & (v < view["H"]) & (facing > 0.1)

        normal_candidates = _torch_candidate_normals(n0, tilt_deg)
        normal_count = len(normal_candidates)
        count = torch.zeros((normal_count, size), device=device, dtype=torch.int16)
        maximum = torch.full((normal_count, size), -torch.inf, device=device, dtype=dtype)
        second = torch.full((normal_count, size), -torch.inf, device=device, dtype=dtype)
        for ref in range(view_count):
            selected = torch.nonzero((ref_ids == ref) & valid[ref], as_tuple=False).flatten()
            if not len(selected):
                continue
            ref_view = gpu_views[ref]
            dref = directions[ref, selected]
            alignment = torch.full((view_count, len(selected)), -9.0, device=device, dtype=dtype)
            for source in range(view_count):
                if source == ref:
                    continue
                good = valid[source, selected]
                alignment[source, good] = (directions[source, selected][good] * dref[good]).sum(1)
            order = torch.argsort(-alignment, dim=0)[:k_src]
            ur, vr = pu[ref, selected], pv[ref, selected]
            ref_u = torch.round(ur[:, None] + du[None]).to(torch.long)
            ref_v = torch.round(vr[:, None] + dv[None]).to(torch.long)
            in_patch = ((ref_u >= 0) & (ref_u < ref_view["W"]) &
                        (ref_v >= 0) & (ref_v < ref_view["H"]))
            ref_u = ref_u.clamp(0, ref_view["W"] - 1)
            ref_v = ref_v.clamp(0, ref_view["H"] - 1)
            ref_fg = in_patch & ref_view["fg"][ref_v, ref_u]
            ref_patch = ref_view["gray"][ref_v, ref_u]
            Xref = X0[selected] @ ref_view["Rwc"].T + ref_view["t"]
            normals_ref = torch.einsum("cmj,jk->cmk", normal_candidates[:, selected], ref_view["Rwc"].T)
            plane_d = (normals_ref * Xref.unsqueeze(0)).sum(-1)
            ref_hom = torch.stack([ur[:, None] + du[None], vr[:, None] + dv[None],
                                   torch.ones((len(selected), pixels), device=device, dtype=dtype)], dim=-1)

            for rank in range(k_src):
                choices = order[rank]
                for source in torch.unique(choices).tolist():
                    if source == ref:
                        continue
                    local = torch.nonzero((choices == source) & (alignment[source] > -1), as_tuple=False).flatten()
                    if not len(local):
                        continue
                    source_view = gpu_views[source]
                    Rsr = source_view["Rwc"] @ ref_view["Rwc"].T
                    tsr = source_view["t"] - source_view["Rwc"] @ ref_view["Rwc"].T @ ref_view["t"]
                    hom = ref_hom[local]
                    candidate_rows = selected[local]
                    for ci in range(normal_count):
                        nr = normals_ref[ci, local]
                        depth = plane_d[ci, local]
                        good_plane = depth.abs() > 1e-6
                        matrix = Rsr.unsqueeze(0) + (tsr[None, :, None] * nr[:, None, :]) / depth[:, None, None]
                        homography = torch.einsum("ij,njk,kl->nil", source_view["K"], matrix, ref_view["Kinv"])
                        warped = torch.einsum("nij,npj->npi", homography, hom)
                        su = warped[:, :, 0] / warped[:, :, 2]
                        sv = warped[:, :, 1] / warped[:, :, 2]
                        source_patch, source_ok = _torch_bilinear(source_view["gray"], su, sv)
                        patch_ok = source_ok & in_patch[local] & ref_fg[local]
                        patch_ok = patch_ok.all(1) & good_plane & _torch_mask_samples(source_view["fg"], su, sv).all(1)
                        passing = torch.nonzero(patch_ok, as_tuple=False).flatten()
                        if not len(passing):
                            continue
                        ncc, textured = _torch_ncc_rows(ref_patch[local][passing], source_patch[passing])
                        accepted = torch.nonzero(textured & ~torch.isnan(ncc), as_tuple=False).flatten()
                        if not len(accepted):
                            continue
                        target = candidate_rows[passing[accepted]]
                        value = ncc[accepted]
                        count[ci, target] += 1
                        previous = maximum[ci, target]
                        second[ci, target] = torch.where(value > previous, previous, torch.maximum(second[ci, target], value))
                        maximum[ci, target] = torch.maximum(previous, value)

        values = torch.where(count >= min_src, second, torch.full_like(second, torch.nan))
        has_value = torch.isfinite(values).any(0)
        best = torch.full((size,), torch.nan, device=device, dtype=dtype)
        best[has_value] = torch.max(torch.nan_to_num(values[:, has_value], nan=-torch.inf), dim=0).values
        result[start:stop] = best.to(torch.float32).cpu().numpy()
    return result


def forced_reference_scores(points, normals, refs, views, patch_radius=2, tilt_deg=20.0,
                            k_src=4, min_src=2, batch_size=768, device="cpu"):
    if device == "cpu":
        return _forced_reference_scores_cpu(points, normals, refs, views, patch_radius, tilt_deg,
                                            k_src, min_src, batch_size)
    if device == "cuda":
        return _forced_reference_scores_cuda(points, normals, refs, views, patch_radius, tilt_deg,
                                             k_src, min_src, batch_size)
    raise ValueError(f"unknown scoring device: {device}")
def original_pixels(surface_indices, source_rows, refs, grid_min, voxel_size, views):
    points = grid_min + (surface_indices[source_rows] + 0.5) * voxel_size
    pixels = np.full(len(points), -1, np.int64)
    for ref, view in enumerate(views):
        rows = np.flatnonzero(refs == ref)
        if len(rows) == 0:
            continue
        X = points[rows] @ view["Rwc"].T + view["t"]
        u = np.rint(view["K"][0, 0] * X[:, 0] / X[:, 2] + view["K"][0, 2]).astype(int)
        v = np.rint(view["K"][1, 1] * X[:, 1] / X[:, 2] + view["K"][1, 2]).astype(int)
        in_image = (u >= 0) & (u < view["W"]) & (v >= 0) & (v < view["H"])
        pixels[rows[in_image]] = v[in_image] * view["W"] + u[in_image]
    return pixels


def build_sparse_maps(refs, pixels, selected_points, selected_scores, view_count):
    """Map each reference pixel to its strongest independently selected point."""
    maps = [dict() for _ in range(view_count)]
    for row, (ref, pixel) in enumerate(zip(refs, pixels)):
        if pixel < 0 or not np.isfinite(selected_scores[row]):
            continue
        previous = maps[ref].get(int(pixel))
        if previous is None or selected_scores[row] > previous[1]:
            maps[ref][int(pixel)] = (selected_points[row], float(selected_scores[row]))
    return maps


def source_support(point, ref, maps, views, tolerance_m):
    """Number of other independent depth maps selecting this same 3-D point."""
    support = 0
    for source, view in enumerate(views):
        if source == ref:
            continue
        X = point @ view["Rwc"].T + view["t"]
        if X[2] <= 1e-6:
            continue
        u = int(round(view["K"][0, 0] * X[0] / X[2] + view["K"][0, 2]))
        v = int(round(view["K"][1, 1] * X[1] / X[2] + view["K"][1, 2]))
        found = False
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                candidate = maps[source].get((v + dy) * view["W"] + u + dx)
                if candidate is not None and np.linalg.norm(candidate[0] - point) <= tolerance_m:
                    found = True
                    break
            if found:
                break
        support += int(found)
    return support


def run(scene, candidate_root, out_root, ncc_reject, ncc_accept, ncc_margin,
        min_sources, tolerance_voxels, min_cluster):
    source = EVAL / "srp_hull_mv2_v12_am1" / scene / "hull.npz"
    candidate_path = EVAL / candidate_root / scene / "multi_ref_candidates.npz"
    hull = np.load(source, allow_pickle=False)
    candidates = np.load(candidate_path, allow_pickle=False)
    occupancy = hull["occupancy"].astype(bool)
    grid_min = hull["grid_min"].astype(float)
    voxel_size = float(hull["voxel_size"])
    surface_indices = candidates["surface_indices"].astype(np.int32)
    source_rows = candidates["source_surface_rows"].astype(np.int32)
    refs = candidates["reference_views"].astype(np.int16)
    layers = candidates["layers"].astype(np.int32)
    views, names = load_views(scene)
    if list(candidates["view_names"]) != names:
        raise ValueError("candidate and scoring view order differ")

    normal_grid = IC.hull_normals(occupancy)
    row_normals = normal_grid[tuple(surface_indices[source_rows].T)]
    valid = (layers >= 0).all(2)
    rows, columns = np.nonzero(valid)
    candidate_indices = layers[rows, columns]
    points = grid_min + (candidate_indices + 0.5) * voxel_size
    t0 = time.time()
    flat_scores = forced_reference_scores(points, row_normals[rows], refs[rows], views)
    score_seconds = time.time() - t0
    scores = np.full(valid.shape, np.nan, np.float32)
    scores[rows, columns] = flat_scores
    best_layer = np.argmax(np.where(np.isfinite(scores), scores, -np.inf), axis=1)
    best_scores = scores[np.arange(len(layers)), best_layer]
    best_indices = layers[np.arange(len(layers)), best_layer]
    selected_points = grid_min + (best_indices + 0.5) * voxel_size
    selected_points[~np.isfinite(best_scores)] = np.nan
    pixels = original_pixels(surface_indices, source_rows, refs, grid_min, voxel_size, views)
    maps = build_sparse_maps(refs, pixels, selected_points, best_scores, len(views))

    outer_scores = scores[:, 0]
    proposal = (
        (best_layer > 0) & np.isfinite(outer_scores) & np.isfinite(best_scores)
        & (outer_scores < ncc_reject) & (best_scores >= ncc_accept)
        & (best_scores >= outer_scores + ncc_margin)
    )
    supports = np.zeros(len(layers), np.int8)
    for row in np.flatnonzero(proposal):
        supports[row] = source_support(selected_points[row], refs[row], maps, views,
                                       tolerance_voxels * voxel_size)
    accepted = proposal & (supports >= min_sources)
    selected_layer = np.where(accepted, best_layer, 0)
    centers = np.asarray([view["Cw"] for view in views])
    raw_remove = removal_rays(occupancy, grid_min, voxel_size,
                              surface_indices[source_rows], refs, layers,
                              selected_layer, centers)
    remove = connected_keep(raw_remove, min_cluster)
    carved = occupancy & ~remove

    output = EVAL / out_root / scene
    output.mkdir(parents=True, exist_ok=True)
    records = np.flatnonzero(np.isfinite(best_scores))
    np.savez_compressed(output / "depth_maps_sparse.npz", row=records, reference_views=refs[records],
                        pixels=pixels[records], points=selected_points[records], scores=best_scores[records])
    meta = {
        "src": "srp_hull_mv2_v12_am1", "candidate_root": candidate_root,
        "method": "interval_forced_ref_ncc_reprojection_consistency",
        "views": names, "ncc_reject": ncc_reject, "ncc_accept": ncc_accept,
        "ncc_margin": ncc_margin, "min_source_support": min_sources,
        "depth_tolerance_voxels": tolerance_voxels, "min_cluster": min_cluster,
        "reference_rows": int(len(layers)), "candidate_points": int(len(points)),
        "scored_points": int(np.isfinite(flat_scores).sum()), "depth_map_entries": int(sum(len(m) for m in maps)),
        "proposals": int(proposal.sum()), "accepted": int(accepted.sum()),
        "raw_removed": int(raw_remove.sum()), "removed": int(remove.sum()),
        "ncc_seconds": score_seconds,
    }
    np.savez_compressed(output / "hull.npz", occupancy=carved, grid_min=grid_min,
                        voxel_size=np.float64(voxel_size), observed=hull["observed"],
                        build_meta=json.dumps(meta))
    (output / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenes", nargs="+")
    parser.add_argument("--candidate-root", default="hull_multi_ref_candidates_mv2_v12_am1")
    parser.add_argument("--out-root", default="srp_hull_mv2_v12_am1_interval_depthcons")
    parser.add_argument("--ncc-reject", type=float, default=0.10)
    parser.add_argument("--ncc-accept", type=float, default=0.25)
    parser.add_argument("--ncc-margin", type=float, default=0.10)
    parser.add_argument("--min-sources", type=int, default=2)
    parser.add_argument("--tolerance-voxels", type=float, default=2.0)
    parser.add_argument("--min-cluster", type=int, default=3)
    args = parser.parse_args()
    for scene in args.scenes:
        run(scene, args.candidate_root, args.out_root, args.ncc_reject, args.ncc_accept,
            args.ncc_margin, args.min_sources, args.tolerance_voxels, args.min_cluster)


if __name__ == "__main__":
    main()
