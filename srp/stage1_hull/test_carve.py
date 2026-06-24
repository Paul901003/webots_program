#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""test_carve.py — Stage 1 carve_visual_hull 驗收測試(對應規格 §6 的 T1–T10)。

合成資料:給定 ground-truth 形狀(球/方塊)的體素佔據,把佔據體素中心投影回各視角
渲染剪影遮罩(與 carve 同一投影模型,保證自洽),再 carve 還原並檢驗。
不依賴 pytest:直接執行,逐項印 PASS/FAIL,最後回傳退出碼。

規格鐵則:先過 T2(方向)、T6(對齊),再信任其餘。
用法: ./srp/stage1_hull/test_carve.py
"""

import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from carve import carve_visual_hull, c2w_to_w2c, VisualHull   # noqa: E402

RNG = np.random.default_rng(0)


# ── 相機 / 渲染 helper ───────────────────────────────────────────────────
def intrinsics(W, H, fov_deg=60.0):
    f = (W / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    return np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1.0]])


def look_at_c2w(eye, target, up=(0, 0, 1)):
    """OpenCV 相機(z 前、x 右、y 下)的 camera→world 旋轉與位置。"""
    eye = np.asarray(eye, float); target = np.asarray(target, float)
    up = np.asarray(up, float)
    z = target - eye; z /= np.linalg.norm(z)            # forward
    if abs(np.dot(z, up)) > 0.999:                      # 近天頂,換參考 up
        up = np.array([0.0, 1.0, 0.0])
    x = np.cross(z, up); x /= np.linalg.norm(x)         # right
    y = np.cross(z, x)                                  # down
    R_c2w = np.column_stack([x, y, z])
    return R_c2w, eye


def hemisphere_eyes(center, radius, n_az=8, elevs=(30, 55, 80)):
    eyes = []
    for el in elevs:
        e = np.radians(el)
        for k in range(n_az):
            a = 2 * np.pi * k / n_az
            eyes.append(np.asarray(center) + radius * np.array(
                [np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)]))
    eyes.append(np.asarray(center) + radius * np.array([0, 0, 1.0]))   # 天頂
    return eyes


def render_mask(occ_centers, K, R_w2c, t, H, W, dilate=1):
    """把佔據 voxel 中心投影成剪影遮罩(bool)。"""
    X = occ_centers @ R_w2c.T + t
    z = X[:, 2]; ok = z > 0
    u = np.round(K[0, 0] * X[:, 0] / np.where(ok, z, 1) + K[0, 2]).astype(int)
    v = np.round(K[1, 1] * X[:, 1] / np.where(ok, z, 1) + K[1, 2]).astype(int)
    inb = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    m = np.zeros((H, W), bool)
    m[v[inb], u[inb]] = True
    if dilate:
        m = ndimage.binary_dilation(m, iterations=dilate)
    return m


def sphere_occ(P, c, r):
    return np.linalg.norm(P - np.asarray(c), axis=1) <= r


def cube_occ(P, c, half):
    d = np.abs(P - np.asarray(c))
    return np.all(d <= half, axis=1)


def make_views(eyes, target, W=200, H=200, as_w2c=True):
    """回傳 (Ks, extr)。as_w2c=True → 正確 world→camera;False → 故意給 c2w(測 T2)。"""
    K = intrinsics(W, H)
    Ks, extr = [], []
    for eye in eyes:
        R_c2w, C = look_at_c2w(eye, target)
        Ks.append(K)
        if as_w2c:
            extr.append(c2w_to_w2c(R_c2w, C))
        else:
            extr.append((R_c2w, C))        # 錯誤:把 c2w 當 w2c
    return Ks, extr


def grid_P(grid_min, grid_max, vs):
    gmin = np.asarray(grid_min, float); gmax = np.asarray(grid_max, float)
    dims = np.maximum(np.round((gmax - gmin) / vs).astype(int), 1)
    ax = [(np.arange(d) + 0.5) * vs + gmin[i] for i, d in enumerate(dims)]
    gi, gj, gk = np.meshgrid(*ax, indexing="ij")
    return np.stack([gi.ravel(), gj.ravel(), gk.ravel()], axis=1), tuple(int(d) for d in dims)


# ── 測試 ─────────────────────────────────────────────────────────────────
RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")


def common_sphere(vs=0.01, as_w2c=True):
    C = np.array([0.0, 0.0, 0.20]); r = 0.10
    gmin, gmax = [-0.2, -0.2, 0.0], [0.2, 0.2, 0.4]
    P, shape = grid_P(gmin, gmax, vs)
    gt = sphere_occ(P, C, r)
    eyes = hemisphere_eyes(C, 0.6)
    Ks, extr = make_views(eyes, C, as_w2c=as_w2c)
    masks = [render_mask(P[gt], Ks[i], *extr[i], 200, 200) for i in range(len(eyes))]
    return C, r, gmin, gmax, vs, gt, shape, Ks, extr, masks


def t1_sphere():
    C, r, gmin, gmax, vs, gt, shape, Ks, extr, masks = common_sphere()
    hull = carve_visual_hull(masks, Ks, extr, gmin, gmax, vs)
    P, _ = grid_P(gmin, gmax, vs)
    occ = hull.occupancy.ravel()
    pts = P[occ]
    ext = pts.max(0) - pts.min(0)
    cen = pts.mean(0)
    bbox_ok = np.all(np.abs(ext - 2 * r) <= 2 * vs)
    cen_ok = np.all(np.abs(cen - C) <= 2 * vs)
    vol = occ.sum() * vs ** 3
    sph = 4 / 3 * np.pi * r ** 3; cube = (2 * r) ** 3
    vol_ok = sph * 0.8 <= vol <= cube * 1.1
    check("T1 標準球-bbox≈2r", bbox_ok, f"ext={np.round(ext,3)} 期望≈{2*r}")
    check("T1 標準球-中心", cen_ok, f"cen={np.round(cen,3)}")
    check("T1 標準球-體積介於球與立方", vol_ok, f"vol={vol:.5f} [{sph:.5f},{cube:.5f}]")


def t2_direction():
    # 正確設定
    C, r, gmin, gmax, vs, gt, shape, Ks, extr, masks = common_sphere(as_w2c=True)
    good = carve_visual_hull(masks, Ks, extr, gmin, gmax, vs).occupancy.sum()
    # 錯誤:把 c2w 當 w2c
    _, _, _, _, _, _, _, Ks2, extr2, masks2 = common_sphere(as_w2c=False)
    bad = carve_visual_hull(masks2, Ks2, extr2, gmin, gmax, vs).occupancy.sum()
    check("T2 方向慣例(錯向→塌縮)", bad < 0.05 * max(good, 1),
          f"正確={good} 錯向={bad}")


def t3_monotonic():
    C, r, gmin, gmax, vs, gt, shape, Ks, extr, masks = common_sphere()
    ok = True; last = None
    for k in range(2, len(masks) + 1):
        occ = carve_visual_hull(masks[:k], Ks[:k], extr[:k], gmin, gmax, vs).occupancy
        if last is not None:
            ok = ok and bool(np.all(occ <= last))   # 加第 k 視角後須為前一步子集
        last = occ
    check("T3 單調性(加視角只縮不增)", ok)


def t4_two_cubes():
    vs = 0.01
    gmin, gmax = [-0.3, -0.3, 0.0], [0.3, 0.3, 0.3]
    P, shape = grid_P(gmin, gmax, vs)
    c1 = [-0.12, 0, 0.1]; c2 = [0.12, 0, 0.1]; half = [0.05, 0.05, 0.05]
    gt = cube_occ(P, c1, half) | cube_occ(P, c2, half)
    eyes = hemisphere_eyes([0, 0, 0.1], 0.7)
    Ks, extr = make_views(eyes, [0, 0, 0.1])
    masks = [render_mask(P[gt], Ks[i], *extr[i], 200, 200) for i in range(len(eyes))]
    hull = carve_visual_hull(masks, Ks, extr, gmin, gmax, vs)
    lab, n = ndimage.label(hull.occupancy, ndimage.generate_binary_structure(3, 1))
    check("T4 兩方塊→2 連通元件", n == 2, f"得到 {n} 個元件")


def t5_table():
    C, r, gmin, gmax, vs, gt, shape, Ks, extr, masks = common_sphere()
    tz = 0.20
    hull = carve_visual_hull(masks, Ks, extr, gmin, gmax, vs, table_z=tz)
    centers = hull.voxel_centers().reshape(-1, 3)
    occ = hull.occupancy.ravel()
    below = occ & (centers[:, 2] < tz)
    check("T5 封底(table_z 下方全空)", below.sum() == 0, f"低於桌面仍佔據={below.sum()}")


def t6_alignment():
    C, r, gmin, gmax, vs, gt, shape, Ks, extr, masks = common_sphere()
    hull = carve_visual_hull(masks, Ks, extr, gmin, gmax, vs)
    centers = hull.voxel_centers().reshape(-1, 3)
    occ_centers = centers[hull.occupancy.ravel()]
    ious = []
    for i in range(len(masks)):
        rm = render_mask(occ_centers, Ks[i], *extr[i], 200, 200, dilate=1)  # 與輸入同 dilation
        inter = (rm & masks[i]).sum(); union = (rm | masks[i]).sum()
        ious.append(inter / max(union, 1))
    miou = float(np.mean(ious))
    check("T6 對齊自檢(重投影 IoU>0.9)", miou > 0.9, f"mean IoU={miou:.3f}")


def t7_reproducible():
    C, r, gmin, gmax, vs, gt, shape, Ks, extr, masks = common_sphere()
    a = carve_visual_hull(masks, Ks, extr, gmin, gmax, vs).occupancy
    b = carve_visual_hull(masks, Ks, extr, gmin, gmax, vs).occupancy
    check("T7 可重現(兩次完全相等)", np.array_equal(a, b))


def t10_observed():
    """observed=in_front&in_bounds(視錐內)。用涵蓋視錐外的大 grid:
    驗 ①視錐內背景→observed且空 ②存在視錐外未觀測 voxel ③未觀測⇒未佔據。"""
    C = np.array([0.0, 0.0, 0.10]); r = 0.06; vs = 0.02
    gmin, gmax = [-0.6, -0.6, 0.0], [0.6, 0.6, 0.5]      # 遠大於相機框住範圍
    P, _ = grid_P(gmin, gmax, vs)
    gt = sphere_occ(P, C, r)
    eyes = hemisphere_eyes(C, 0.5, n_az=8, elevs=(40, 70))  # 緊框中心
    Ks, extr = make_views(eyes, C)
    masks = [render_mask(P[gt], Ks[i], *extr[i], 200, 200) for i in range(len(eyes))]
    hull = carve_visual_hull(masks, Ks, extr, gmin, gmax, vs)
    centers = hull.voxel_centers().reshape(-1, 3)
    occ = hull.occupancy.ravel(); obs = hull.observed.ravel()
    near = (np.linalg.norm(centers - C, axis=1) > r + 2 * vs) & \
           (np.linalg.norm(centers - C, axis=1) < r + 5 * vs)
    obs_empty = near & obs & ~occ
    check("T10 視錐內背景→observed且空", obs_empty.sum() > 0, f"數量={obs_empty.sum()}")
    check("T10 存在未觀測空間(observed=False)", np.any(~obs),
          f"未觀測 voxel={int((~obs).sum())}/{obs.size}")
    check("T10 未觀測⇒未佔據", not np.any(~obs & occ),
          f"未觀測且佔據={int((~obs & occ).sum())}")


def main():
    print("Stage 1 carve_visual_hull 驗收測試\n")
    print("[關鍵] 先看 T2 / T6:")
    t2_direction()
    t6_alignment()
    print("[其餘]")
    t1_sphere()
    t3_monotonic()
    t4_two_cubes()
    t5_table()
    t7_reproducible()
    t10_observed()
    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n== {n_pass}/{len(RESULTS)} 通過 ==")
    sys.exit(0 if n_pass == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
