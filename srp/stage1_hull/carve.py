#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""carve.py — Stage 1 visual hull carving(規格實作,GPU/torch)。

單一真相來源:plan/visual_hull_carving_spec.md(§2 契約、§3 慣例、§4 演算法、§5 介面)。
通過 test_carve.py 的 T1–T10 即視為與規格目的一致。

核心:把每個 3D voxel 投影到每張 2D 影像,檢查落前景/背景;唯有「所有」視角都落前景才保留
(交集 → 保守)。額外輸出 observed(是否曾被任一視角看進視錐 in_front&in_bounds),供場景 3a。

計算用 torch:device='auto' 時有 CUDA 走 GPU,否則 CPU。輸入/輸出皆 numpy(介面不變)。
不用深度、不做影像配準;對齊只靠共同世界座標系內外參。
"""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class VisualHull:
    occupancy: np.ndarray      # (Nx,Ny,Nz) bool
    observed: np.ndarray       # (Nx,Ny,Nz) bool
    grid_min: np.ndarray       # (3,) float, 公尺
    voxel_size: float          # 公尺
    frame: str = "world"

    def voxel_centers(self) -> np.ndarray:
        nx, ny, nz = self.occupancy.shape
        gi, gj, gk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
        idx = np.stack([gi, gj, gk], axis=-1).astype(np.float64)
        return self.grid_min + (idx + 0.5) * self.voxel_size

    def volume(self) -> float:
        return float(self.occupancy.sum()) * self.voxel_size ** 3


def c2w_to_w2c(R_c2w: np.ndarray, t_c2w: np.ndarray):
    """camera→world 轉 world→camera(規格 C1)。X_cam = R@X_world + t。"""
    R = np.asarray(R_c2w).T
    t = -R @ np.asarray(t_c2w)
    return R, t


def resolve_device(device="auto"):
    if device == "cuda" or (device == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def _grid_dims(grid_min, grid_max, voxel_size):
    grid_min = np.asarray(grid_min, dtype=np.float64)
    grid_max = np.asarray(grid_max, dtype=np.float64)
    dims = np.maximum(np.round((grid_max - grid_min) / voxel_size).astype(int), 1)
    return grid_min, tuple(int(d) for d in dims)


def carve_visual_hull(
    masks,                              # [V] (H,W) bool/uint, True=前景
    intrinsics,                         # [V] (3,3) K
    extrinsics_w2c,                     # [V] (R(3,3), t(3,)); X_cam = R@X_world + t
    grid_min,                           # (3,) 公尺
    grid_max,                           # (3,) 公尺
    voxel_size: float,                  # 公尺
    table_z=None,                       # 支撐平面世界 z;None=不封底
    outside_is_background: bool = True,  # 規格 C4
    allow_miss: int = 0,                 # soft carving:容忍幾個視角漏檢(0=硬交集,規格核心)
    device="auto",
) -> VisualHull:
    """見規格 §4。回傳世界座標系的佔據網格 + observed(numpy)。
    allow_miss>0 = soft hull(plan B4/[v2]):voxel 通過視角數 ≥ V−allow_miss 即保留,抗 SAM 漏檢。"""
    V = len(masks)
    assert len(intrinsics) == V and len(extrinsics_w2c) == V
    dev = resolve_device(device)
    gmin_np, shape = _grid_dims(grid_min, grid_max, voxel_size)
    nx, ny, nz = shape

    gmin = torch.tensor(gmin_np, dtype=torch.float32, device=dev)
    ax = torch.arange(nx, device=dev, dtype=torch.float32)
    ay = torch.arange(ny, device=dev, dtype=torch.float32)
    az = torch.arange(nz, device=dev, dtype=torch.float32)
    gi, gj, gk = torch.meshgrid(ax, ay, az, indexing="ij")
    P = torch.stack([gi.reshape(-1), gj.reshape(-1), gk.reshape(-1)], dim=1)
    P = gmin + (P + 0.5) * voxel_size            # (M,3) 世界座標 voxel 中心
    M = P.shape[0]

    keep_votes = torch.zeros(M, dtype=torch.int16, device=dev)   # 每 voxel 通過(前景)的視角數
    observed = torch.zeros(M, dtype=torch.bool, device=dev)

    for i in range(V):
        K = np.asarray(intrinsics[i], dtype=np.float64)
        R, t = extrinsics_w2c[i]
        Rt = torch.tensor(np.asarray(R), dtype=torch.float32, device=dev)
        tt = torch.tensor(np.asarray(t), dtype=torch.float32, device=dev)
        mask = torch.as_tensor(np.ascontiguousarray(np.asarray(masks[i]) > 0),
                               dtype=torch.bool, device=dev)
        H, W = mask.shape
        fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

        X = P @ Rt.T + tt                        # (M,3)
        z = X[:, 2]
        in_front = z > 0.0                       # C3
        zz = torch.where(in_front, z, torch.ones_like(z))
        u = fx * X[:, 0] / zz + cx
        v = fy * X[:, 1] / zz + cy
        ui = torch.round(u).long()
        vi = torch.round(v).long()
        in_bounds = in_front & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)   # C4

        observed |= in_bounds

        fg = torch.zeros(M, dtype=torch.bool, device=dev)
        sel = in_bounds
        flat = vi[sel] * W + ui[sel]             # 攤平索引取遮罩值
        fg[sel] = mask.reshape(-1)[flat]

        keep = (in_bounds & fg) if outside_is_background else ((~in_bounds) | (in_front & fg))
        keep_votes += keep.to(torch.int16)

    occupancy = keep_votes >= (V - allow_miss)   # allow_miss=0 → 全數通過 = 硬交集
    if table_z is not None:                      # A3 封底
        occupancy &= ~(P[:, 2] < float(table_z))

    occ = occupancy.reshape(shape).cpu().numpy()
    obs = observed.reshape(shape).cpu().numpy()
    return VisualHull(occupancy=occ, observed=obs,
                      grid_min=gmin_np, voxel_size=float(voxel_size), frame="world")
