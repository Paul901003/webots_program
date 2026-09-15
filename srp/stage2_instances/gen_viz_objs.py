#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""gen_viz_objs.py — 為 hull_viz supervisor 產生「srp hull(每 instance 上色)+ 真實 YCB GT」obj+manifest。

讀 data/eval/srp_hull/<scene>/instances.npz(labels 網格)→ 每 instance marching cubes → 世界座標 .obj。
GT:由 amodal/actual annotations 取各物體位姿,擺真實 YCB mesh(textured.obj,半透明)以供對照。
輸出 manifest.json(格式與 hull_viz_supervisor 相容:items=hull 殼、ycb_items=真實模型)。

用法: ./srp/stage2_instances/gen_viz_objs.py n3_scene0030 --out <objs_dir> [--no-gt]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import mcubes

REPO = Path(__file__).resolve().parents[2]
HULL_ROOT = REPO / "data" / "eval" / "srp_hull"
import sys as _s, pathlib as _pl; _s.path.insert(0, str(_pl.Path(__file__).resolve().parents[2] / "srp" / "io")); from labels import LABELS  # data/labels 分層(類別/數量/場景)
ASSETS = REPO / "urdfs" / "ycb_assets"
GEO = json.loads((REPO / "controllers" / "ycb_supervisor" / "ycb_geometries.json").read_text())


def ycb_center(name):
    c = GEO.get(name, {}).get("center", {"x": 0, "y": 0, "z": 0})
    return np.array([c["x"], c["y"], c["z"]], dtype=float)


def axis_angle_to_mat(axis, angle):
    axis = np.asarray(axis, float); n = np.linalg.norm(axis)
    if n < 1e-9 or abs(angle) < 1e-12:
        return np.eye(3)
    x, y, z = axis / n; c, s = np.cos(angle), np.sin(angle); C = 1 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def gt_objects(scene):
    """actual/annotations.json 的靜態物體位姿 [{name, position_m, rotation_axis_angle}]。"""
    ann = LABELS / scene / "actual" / "annotations.json"
    if not ann.is_file():
        return []
    d = json.loads(ann.read_text())
    return d["images"][0].get("objects", []) if d.get("images") else []

# 與 gen_hull_gt_report.py 的 PALETTE 同一組顏色(0-255→0-1),hull k 兩邊同色
PALETTE = [[0.902, 0.235, 0.235], [0.235, 0.627, 0.902], [0.235, 0.784, 0.353],
           [0.902, 0.627, 0.157], [0.667, 0.353, 0.863], [0.157, 0.784, 0.784],
           [0.902, 0.392, 0.667], [0.588, 0.588, 0.235], [0.392, 0.392, 0.902],
           [0.235, 0.902, 0.588]]


def inst_obj(occ_bool, grid_min, vs, path):
    verts, faces = mcubes.marching_cubes(occ_bool.astype(np.float32), 0.5)
    if len(verts) == 0:
        return False
    world = grid_min + (verts + 0.5) * vs
    lines = [f"v {x:.5f} {y:.5f} {z:.5f}" for x, y, z in world]
    lines += [f"f {a+1} {b+1} {c+1}" for a, b, c in faces.astype(int)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


from scipy.ndimage import binary_erosion, generate_binary_structure
_ST6 = generate_binary_structure(3, 1)   # 6-鄰居(面相鄰)


def surface_of(occ):
    """表面 voxel = occupied 且至少一個 6-鄰居是空(內部挖掉)。"""
    occ = occ.astype(bool)
    return occ & ~binary_erosion(occ, structure=_ST6)


def cubes_obj(mask, grid_min, vs, path):
    """每個 True voxel 生一個邊長 vs 的 cube;只畫『外表面』(相鄰為空的面),內部重疊面不畫(免 z-fighting/黑塗)。
    三角 winding 已使法線朝外。"""
    mask = np.asarray(mask, bool)
    idx = np.argwhere(mask)
    if len(idx) == 0:
        return False
    centers = grid_min + (idx + 0.5) * vs
    h = vs / 2
    cv = np.array([[-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h],
                   [-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h]])
    # (鄰居方向 (x,y,z), 該面顯式法線 index(1-6), 兩三角[法線朝外]);鄰居實心 → 內部面不畫
    FACES = [((0, 0, -1), 1, [(0, 2, 1), (0, 3, 2)]), ((0, 0, 1), 2, [(4, 5, 6), (4, 6, 7)]),
             ((0, -1, 0), 3, [(0, 5, 4), (0, 1, 5)]), ((1, 0, 0), 4, [(1, 6, 5), (1, 2, 6)]),
             ((0, 1, 0), 5, [(2, 7, 6), (2, 3, 7)]), ((-1, 0, 0), 6, [(3, 4, 7), (3, 0, 4)])]
    occ = set(map(tuple, idx.tolist()))
    # 顯式法線(6 個面方向),寫進 obj → Webots 不自己平滑 90° 硬邊(平滑會讓面算成不受光→黑)
    lines = ["vn 0 0 -1", "vn 0 0 1", "vn 0 -1 0", "vn 1 0 0", "vn 0 1 0", "vn -1 0 0"]
    for c in centers:
        for x, y, z in cv + c:
            lines.append(f"v {x:.5f} {y:.5f} {z:.5f}")
    for n, (i, j, k) in enumerate(idx.tolist()):
        b = n * 8
        for (dx, dy, dz), ni, tris in FACES:
            if (i + dx, j + dy, k + dz) in occ:
                continue                    # 鄰居實心 → 這是內部面,不畫
            for a, bb, cc in tris:
                lines.append(f"f {b+a+1}//{ni} {b+bb+1}//{ni} {b+cc+1}//{ni}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


def ycb_mesh_path(name):
    for fn in ("textured.obj", "nontextured.ply", "nontextured.stl"):
        p = ASSETS / name / "google_16k" / fn
        if p.is_file():
            return str(p.resolve())
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-gt", action="store_true")
    ap.add_argument("--root", default="srp_hull",
                    help="讀 instances 的根目錄 data/eval/<root>/")
    ap.add_argument("--tag", default="", help="instances 檔名後綴(如 am1_cvsmall)")
    ap.add_argument("--surface", action="store_true",
                    help="改顯示 hull 表面 voxel(cube-per-voxel,讀 srp_hull_v12/<scene>/hull.npz 的 surface),而非 instances")
    ap.add_argument("--cubes", action="store_true",
                    help="每個 instance 用逐-voxel 立方體(cubes_obj)呈現,不做 marching cubes 平滑(看真實離散 voxel)")
    ap.add_argument("--gray-fill", action="store_true", dest="gray_fill",
                    help="灰色定位層:occupancy 中無語意(label 0)的 voxel 用灰色顯示(內部/沒被看到=只定位不上語意)")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for old in list(out.glob("inst_*.obj")) + list(out.glob("surface.obj")):
        old.unlink()

    suf = f"_{args.tag}" if args.tag else ""
    ip = REPO / "data" / "eval" / args.root / args.scene / f"instances{suf}.npz"
    items = []
    if not ip.is_file():
        print(f"[gen_viz] 無 {ip} → 只顯示 GT(hull 空,可能是空 hull/未跑 associate)")
    else:
        z = np.load(ip)
        labels = z["labels"]; gm = z["grid_min"]; vs = float(z["voxel_size"])
        k_ids = [k for k in range(1, int(labels.max()) + 1) if (labels == k).any()]
        if not k_ids:
            print(f"[gen_viz] {args.scene}: instances 為空(空 hull)→ 只顯示 GT")
        for idx, k in enumerate(k_ids):
            mask = (labels == k)
            if args.surface:                 # 每個 instance 挖空(只留表面 voxel);顏色/半透明/分instance 全同原本
                mask = surface_of(mask)
            f = f"inst_{k:02d}.obj"
            render = cubes_obj if args.cubes else inst_obj   # --cubes:逐voxel立方體(不平滑)
            if render(mask, gm, vs, out / f):
                items.append({"file": f, "color": PALETTE[(k - 1) % len(PALETTE)],  # 用 hull 編號 k,和報告一致
                              "transparency": 0.30,  # 半透明(cubes/marching 同),看得到內部結構
                              "name": f"inst_{k:02d}"})
        if args.gray_fill and "occupancy" in z:      # 灰色定位層:occupancy 無語意(label 0)的 voxel
            gray = np.asarray(z["occupancy"]) & (labels == 0)
            if gray.any() and cubes_obj(gray, gm, vs, out / "inst_gray.obj"):
                items.append({"file": "inst_gray.obj", "color": [0.55, 0.55, 0.55],
                              "transparency": 0.15, "name": "location_gray"})

    ycb_items = []
    if not args.no_gt:
        for o in gt_objects(args.scene):
            name = o["name"]
            mesh = ycb_mesh_path(name)
            if not mesh:
                continue
            aa = o.get("rotation_axis_angle", [0, 1, 0, 0])
            R = axis_angle_to_mat(aa[:3], aa[3]) if len(aa) == 4 else np.eye(3)
            tr = (np.asarray(o["position_m"], float) - R @ ycb_center(name)).tolist()
            ycb_items.append({"mesh": mesh, "translation": [float(x) for x in tr],
                              "rotation": [float(x) for x in aa],
                              "color": [0.6, 0.6, 0.6], "transparency": 0.55,
                              "name": f"gt_{name}"})

    mani = {"source": args.scene, "hull": len(items), "gt": 0, "ycb": len(ycb_items),
            "items": items, "ycb_items": ycb_items}
    (out / "manifest.json").write_text(json.dumps(mani, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[gen_viz] {args.scene}: hull instance {len(items)} + 真實YCB {len(ycb_items)} → {out}")


if __name__ == "__main__":
    main()
