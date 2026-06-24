#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""carve_instances.py — B 方法第二步:每個 instance 用自己的遮罩單獨雕 per-object hull。

讀 associate.py 產的 instances.json(各 instance → 各 view 關聯到的遮罩),
每個 view 把該 instance 的(過度分割)遮罩 union 成單一輪廓,以該 instance 中心
為心用緊緻 cube 單獨 carve → 乾淨 per-object hull(只用單一物體輪廓,無跨物體幻影)。

完全不用深度:cube 用 instance 中心 + 固定邊長;姿態只讀 pose.json。

輸出: data/eval/instance_hull/<scene>/visual_hull_inst_NN.obj
需在 webots_visual_hull 環境(torch / torchhull)。
用法: ./instance_hull/carve_instances.py n3_scene0001 [--cube 0.22] [--level 9]
"""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
CAPTURES = REPO / "data" / "captures"
SAM_ROOT = REPO / "data" / "eval" / "sam_only"
INST_ROOT = REPO / "data" / "eval" / "instance_hull"
HFOV_RAD = 1.4746


def rpy_to_R(roll, pitch, yaw):
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=np.float32)


def load_pose(pose_path):
    meta = json.loads(pose_path.read_text(encoding="utf-8"))
    if "position_m" not in meta and isinstance(meta.get("camera"), dict):
        meta = meta["camera"]
    p = meta["position_m"]
    C = np.array([p["x"], p["y"], p["z"]], dtype=np.float32)
    r = meta["rotation_rpy_rad"]
    return C, rpy_to_R(r["roll"], r["pitch"], r["yaw"])


def intrinsics(W, H):
    fx = W / (2.0 * math.tan(HFOV_RAD / 2.0))
    return np.array([[fx, 0, W / 2.0], [0, fx, H / 2.0], [0, 0, 1]], dtype=np.float32)


def make_transform(K, C, R):
    body_to_opencv = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float32)
    w2c = np.eye(4, dtype=np.float32)
    Rwo = body_to_opencv @ R.T
    w2c[:3, :3] = Rwo
    w2c[:3, 3] = -Rwo @ C
    K4 = np.eye(4, dtype=np.float32)
    K4[:3, :3] = K
    return K4 @ w2c


def resolve_scenes(targets):
    scenes = []
    for a in targets:
        if "scene" in a:
            scenes.append(a)
        else:
            scenes += [d.name for d in sorted((CAPTURES / f"multi_n{a}").glob(f"n{a}_scene*"))]
    return scenes


def process_scene(scene, args, device):
    group = scene.split("_")[0]
    scene_dir = CAPTURES / f"multi_{group}" / scene
    inst_dir = INST_ROOT / scene
    inst_json = inst_dir / "instances.json"
    if not inst_json.is_file():
        print(f"[skip] {scene}: 找不到 {inst_json}(先跑 associate.py)")
        return

    data = json.loads(inst_json.read_text(encoding="utf-8"))
    instances = data["instances"]
    import torchhull

    report = [f"scene: {scene}", f"instances: {len(instances)}",
              f"cube 邊長: {args.cube}m  level: {args.level}", ""]
    for old in inst_dir.glob("visual_hull_inst_*.obj"):
        old.unlink()

    for k, inst in enumerate(instances):
        center = np.array(inst["center"], dtype=np.float32)
        corner = (center - args.cube / 2.0).tolist()
        masks, transforms, used = [], [], []
        for vname, files in inst["masks"].items():
            pose_path = scene_dir / f"{vname}_pose.json"
            if not pose_path.is_file():
                continue
            # union 該 view 此 instance 的所有(過度分割)遮罩
            seg = None
            for f in files:
                mp = SAM_ROOT / scene / vname / "masks" / f
                m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                if m is None:
                    continue
                m = (m > 127).astype(np.float32)
                seg = m if seg is None else np.maximum(seg, m)
            if seg is None or seg.sum() == 0:
                continue
            H, W = seg.shape
            K = intrinsics(W, H)
            C, R = load_pose(pose_path)
            masks.append(seg[..., None])
            transforms.append(make_transform(K, C, R))
            used.append(vname)

        if len(masks) < 2:
            report.append(f"inst_{k:02d}: 有效 view < 2,跳過")
            continue
        masks_t = torch.from_numpy(np.stack(masks, 0)).to(device=device, dtype=torch.float32)
        tf_t = torch.from_numpy(np.stack(transforms, 0)).to(device=device, dtype=torch.float32)
        verts, faces = torchhull.visual_hull(
            masks=masks_t, transforms=tf_t, level=args.level,
            cube_corner_bfl=corner, cube_length=args.cube,
            masks_partial=True, transforms_convention="opencv", unique_verts=True)
        verts = verts.detach().cpu().numpy()
        faces = faces.detach().cpu().numpy()
        if verts.size == 0 or faces.size == 0:
            report.append(f"inst_{k:02d}: 空 hull,跳過")
            continue
        out = inst_dir / f"visual_hull_inst_{k:02d}.obj"
        lines = [f"v {v[0]} {v[1]} {v[2]}" for v in verts]
        lines += [f"f {f[0]+1} {f[1]+1} {f[2]+1}" for f in faces]
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        # 尺寸
        ext = verts.max(0) - verts.min(0)
        report.append(f"inst_{k:02d}: {len(used)} views, verts={len(verts)}, "
                      f"尺寸=({ext[0]:.3f}x{ext[1]:.3f}x{ext[2]:.3f}) → {out.name}")

    txt = "\n".join(report)
    print(txt)
    (inst_dir / "carve_report.txt").write_text(txt + "\n", encoding="utf-8")
    print(f"\n→ {inst_dir}/visual_hull_inst_*.obj")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*", default=["n3_scene0001"], help="場景名或組號")
    ap.add_argument("--cube", type=float, default=0.22, help="每物體 cube 邊長(m)")
    ap.add_argument("--level", type=int, default=8, help="torchhull octree level")
    ap.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = ap.parse_args()
    device = torch.device("cuda" if (args.device != "cpu" and torch.cuda.is_available()) else "cpu")
    scenes = resolve_scenes(args.scenes or ["n3_scene0001"])
    if not scenes:
        sys.exit("沒有場景")
    for i, scene in enumerate(scenes, 1):
        print(f"\n===== [{i}/{len(scenes)}] {scene} =====")
        try:
            process_scene(scene, args, device)
        except Exception as e:
            print(f"[error] {scene}: {e}")


if __name__ == "__main__":
    main()
