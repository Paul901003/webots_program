#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""bench_resolution.py — 比較 64³/128³/256³ 體素解析度的計算速度。

同一盒(0.7×0.7×0.35),對給定場景做完整一輪:建格 → 投影 → 投票雕刻 → 6-連通,
分別計時各階段 + 總時間,並印體素數/保留數/元件數。可給多場景取平均。
用法: ./instance_hull/bench_resolution.py [n3_scene0030 ...] [--res 64 128 256]
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
from scipy import ndimage
sys.path.insert(0, str(Path(__file__).resolve().parent))
import hull_common as HC
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
_B2O = torch.tensor(HC.av.BODY_TO_OPENCV.T, dtype=torch.float32, device=DEV)


def run_gpu(scene, res):
    """GPU(torch)版:建格 + 投影 + 投票雕刻(連通仍用 CPU scipy)。回傳 (各階段時間, vox, kept, n)。"""
    views = HC.load_views(scene)
    t = {}
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    xs = torch.linspace(*HC.BOX_X, res, device=DEV); ys = torch.linspace(*HC.BOX_Y, res, device=DEV); zs = torch.linspace(*HC.BOX_Z, res, device=DEV)
    gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="ij")
    P = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], 1)
    torch.cuda.synchronize(); t["建格"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    votes = torch.zeros(P.shape[0], dtype=torch.int16, device=DEV); nv = 0
    for vn, v in views.items():
        fg = HC.foreground(HC.load_masks(scene, vn), v["H"], v["W"])
        if fg is None:
            continue
        fgt = torch.from_numpy(fg).to(DEV)
        C = torch.tensor(v["C"], dtype=torch.float32, device=DEV)
        R = torch.tensor(v["R"], dtype=torch.float32, device=DEV)
        X = (P - C) @ (R @ _B2O)
        z = X[:, 2]; ok = z > 1e-6
        zz = torch.where(ok, z, torch.ones_like(z))
        u = (v["fx"] * X[:, 0] / zz + v["cx"]).round().long()
        w = (v["fx"] * X[:, 1] / zz + v["cy"]).round().long()
        inb = ok & (u >= 0) & (u < v["W"]) & (w >= 0) & (w < v["H"])
        hit = torch.zeros(P.shape[0], dtype=torch.bool, device=DEV)
        idx = torch.nonzero(inb, as_tuple=True)[0]
        hit[idx] = fgt[w[idx], u[idx]]
        votes += hit.to(torch.int16); nv += 1
    occ = votes >= HC.vote_threshold(nv)
    torch.cuda.synchronize(); t["投影+雕刻"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    occ_cpu = occ.cpu().numpy()
    lab, n = ndimage.label(occ_cpu.reshape((res, res, res)), structure=ndimage.generate_binary_structure(3, 1))
    t["連通(CPU)"] = time.perf_counter() - t0
    t["總計"] = sum(t.values())
    return t, P.shape[0], int(occ_cpu.sum()), n


def build_grid(res):
    xs = np.linspace(*HC.BOX_X, res); ys = np.linspace(*HC.BOX_Y, res); zs = np.linspace(*HC.BOX_Z, res)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1).astype(np.float32), (res, res, res)


def run(scene, res):
    views = HC.load_views(scene)
    t = {}
    t0 = time.perf_counter()
    P, shape = build_grid(res); t["建格"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    proj = {vn: HC.project(P, v) for vn, v in views.items()}; t["投影"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    votes = np.zeros(len(P), np.int16); nv = 0
    for vn, v in views.items():
        fg = HC.foreground(HC.load_masks(scene, vn), v["H"], v["W"])
        if fg is None:
            continue
        ui, wi, inb = proj[vn]
        hit = np.zeros(len(P), bool); hit[inb] = fg[wi[inb], ui[inb]]
        votes += hit; nv += 1
    occ = votes >= HC.vote_threshold(nv); t["雕刻"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    lab, n = ndimage.label(occ.reshape(shape), structure=ndimage.generate_binary_structure(3, 1))
    t["連通"] = time.perf_counter() - t0

    t["總計"] = sum(t.values())
    return t, len(P), int(occ.sum()), n


def avg_run(fn, scenes, res):
    agg = {}; vox = 0
    for sc in scenes:
        t = fn(sc, res)[0]
        for kk, vv in t.items():
            agg[kk] = agg.get(kk, 0) + vv
        vox = fn.__name__  # placeholder
    return {kk: vv / len(scenes) for kk, vv in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*", default=["n3_scene0030"])
    ap.add_argument("--res", nargs="*", type=int, default=[64, 128, 256])
    args = ap.parse_args()
    scenes = HC.resolve_scenes(args.scenes)
    print(f"裝置 GPU={DEV} | 場景數 {len(scenes)}:{scenes}")
    if DEV == "cuda":
        run_gpu(scenes[0], 64)   # CUDA 暖機(不計時)
    print(f"\n{'解析度':>7}{'體素數':>12}{'CPU投影+雕刻':>14}{'GPU投影+雕刻':>14}{'加速':>7}{'CPU總計':>9}{'GPU總計':>9}")
    for res in args.res:
        cpu = {}; gpu = {}; vox = 0
        for sc in scenes:
            tc, v, _, _ = run(sc, res)
            for k, val in tc.items():
                cpu[k] = cpu.get(k, 0) + val
            vox = v
            tg = run_gpu(sc, res)[0]
            for k, val in tg.items():
                gpu[k] = gpu.get(k, 0) + val
        n = len(scenes)
        cpu_pc = (cpu["投影"] + cpu["雕刻"]) / n
        gpu_pc = gpu["投影+雕刻"] / n
        cpu_tot = cpu["總計"] / n; gpu_tot = gpu["總計"] / n
        sp = cpu_pc / gpu_pc if gpu_pc > 0 else 0
        print(f"{res}³".rjust(7) + f"{vox:>12,}" + f"{cpu_pc:>13.2f}s" + f"{gpu_pc:>13.2f}s"
              + f"{sp:>6.1f}x" + f"{cpu_tot:>8.2f}s" + f"{gpu_tot:>8.2f}s")


if __name__ == "__main__":
    main()
