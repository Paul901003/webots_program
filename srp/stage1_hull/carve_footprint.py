#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""carve_footprint.py — carve.py 的變體:前景判定改用「整個 voxel footprint(OR)」而非中心單點。

★ 與 carve.py 唯一差別(2026-09-12,使用者要求「用整個 voxel 算」,選 OR 版):
  carve.py:voxel 中心投影成 1 個像素,該像素在遮罩內才算此視角通過。
  本檔:voxel 立方體投影成一小塊 footprint(半徑 r=clip(round(fx·vs/2 / z),0,rmax),
        與 cg_associate.zbuffer_visible 同尺度);**footprint 內任一像素落遮罩內**即算此視角通過(OR,最寬鬆)。
  → hull 會變大、更容易保留邊界/細長物,鬼影也會增多(這是 OR 的預期方向,非 bug)。
  allow_miss、封底、observed、介面、回傳全部與 carve.py 相同。不取代 carve.py。

單一真相來源仍是 plan/visual_hull_carving_spec.md;本檔只是把 §4 的點取樣換成 footprint OR 取樣做對照。
"""

from dataclasses import dataclass

import numpy as np
import torch

RMAX = 8   # footprint 半徑上限(像素),與 zbuffer_visible 一致


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
    outside_is_background: bool = True,  # 規格 C4(本檔 footprint OR 下語意:footprint 全出界即背景)
    allow_miss: int = 0,                 # soft carving:容忍幾個視角漏檢(0=硬交集)
    device="auto",
) -> VisualHull:
    """見 carve.py。差別只在前景判定用 footprint OR(見檔頭)。"""
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
        maskflat = mask.reshape(-1)

        X = P @ Rt.T + tt                        # (M,3)
        z = X[:, 2]
        in_front = z > 0.0                       # C3
        zz = torch.where(in_front, z, torch.ones_like(z))
        u = fx * X[:, 0] / zz + cx
        v = fy * X[:, 1] / zz + cy
        ui = torch.round(u).long()
        vi = torch.round(v).long()

        # ★ 每 voxel footprint 半徑(像素),與 zbuffer_visible 同公式
        r = torch.zeros(M, dtype=torch.long, device=dev)
        r[in_front] = torch.clamp(torch.round(fx * (voxel_size * 0.5) / z[in_front]).long(), 0, RMAX)
        Rm = int(r.max().item()) if M else 0

        fg_any = torch.zeros(M, dtype=torch.bool, device=dev)   # footprint 任一像素落遮罩內
        obs_any = torch.zeros(M, dtype=torch.bool, device=dev)  # footprint 任一像素在畫面內
        for dy in range(-Rm, Rm + 1):
            for dx in range(-Rm, Rm + 1):
                off = in_front & (abs(dx) <= r) & (abs(dy) <= r)   # 此 offset 在該 voxel footprint 內
                if not bool(off.any()):
                    continue
                uu = ui + dx
                vv = vi + dy
                inb = off & (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
                if not bool(inb.any()):
                    continue
                obs_any |= inb
                idxs = inb.nonzero(as_tuple=True)[0]
                hit = maskflat[vv[idxs] * W + uu[idxs]]
                if bool(hit.any()):
                    fg_any[idxs[hit]] = True

        observed |= obs_any
        # outside_is_background=True:footprint 全出界(obs_any=False)視為背景=不通過(與 carve.py C4 對齊)
        keep = (in_front & fg_any) if outside_is_background else ((~obs_any) | (in_front & fg_any))
        keep_votes += keep.to(torch.int16)

    occupancy = keep_votes >= (V - allow_miss)   # allow_miss=0 → 全數通過 = 硬交集
    if table_z is not None:                      # A3 封底
        occupancy &= ~(P[:, 2] < float(table_z))

    occ = occupancy.reshape(shape).cpu().numpy()
    obs = observed.reshape(shape).cpu().numpy()
    return VisualHull(occupancy=occ, observed=obs,
                      grid_min=gmin_np, voxel_size=float(voxel_size), frame="world")
