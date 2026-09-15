#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""fill_solid.py — 把表面 voxel labels 填回實心:每個內部 occupied voxel 繼承「最近表面 voxel」的 id。
讀 <in_root>/<scene>/instances.npz(表面 labels) + srp_hull_v12/<scene>/hull.npz(occupancy)
→ <out_root>/<scene>/instances.npz(實心 labels)。只用於用 eval.py(3D IoU)和四方法公平比較,
不改原表面輸出。
用法: ./fill_solid.py --in-root srp_hull_cg --out-root srp_hull_cg_solid [scene|group|(空=全部)]
"""
import argparse
import glob
import os
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

REPO = Path(__file__).resolve().parents[2]
HULL = REPO / "data" / "eval" / os.environ.get("HULL_ROOT", "srp_hull_v12")   # env 可覆寫(mobilesam 用 srp_hull_mobilesamv2)


def fill(scene, in_root, out_root):
    ip = REPO / "data" / "eval" / in_root / scene / "instances.npz"
    hp = HULL / scene / "hull.npz"
    if not (ip.is_file() and hp.is_file()):
        return False
    z = np.load(ip); lab = z["labels"]; gm = z["grid_min"]; vs = float(z["voxel_size"])
    occ = np.load(hp)["occupancy"].astype(bool)
    if lab.shape != occ.shape:
        return False
    if (lab > 0).sum() == 0:
        filled = lab.astype(np.int32)
    else:
        _, inds = distance_transform_edt(lab == 0, return_indices=True)   # 每 voxel ← 最近有 label 者
        filled = lab[tuple(inds)].astype(np.int32)
        filled[~occ] = 0                                                  # 只保留 hull 實心內
    out = REPO / "data" / "eval" / out_root / scene
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "instances.npz", labels=filled, grid_min=gm, voxel_size=vs)
    return True


def resolve(t):
    if not t:
        return sorted(Path(p).parent.name for p in glob.glob(str(HULL / "*_scene*/hull.npz")))
    out = []
    for a in t:
        if "scene" in a:
            out.append(a)
        else:
            out += [Path(p).parent.name for p in glob.glob(str(HULL / f"{a}_scene*/hull.npz"))]
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("targets", nargs="*")
    args = ap.parse_args()
    n = 0
    for sc in resolve(args.targets):
        if fill(sc, args.in_root, args.out_root):
            n += 1
    print(f"填實心 {n} 場 → data/eval/{args.out_root}/")


if __name__ == "__main__":
    main()
