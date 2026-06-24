#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""split_hull.py — 前景遮罩 → (固定 cube)雕殼 → 3D 連通元件分物體。

完全不用 depth:雕刻立方體(carving cube)直接用「已知工作空間幾何」寫死,
只讀 view_XX_pose.json 的相機位姿 + view_XX_mask_foreground.png 的前景遮罩,
呼叫 torchhull.visual_hull 雕出一坨合併 hull;再用 trimesh 連通元件(mesh.split)
切成各物體(物體空間不重疊 → 每坨 = 一個物體)。

需在 webots_visual_hull 環境(有 torch / torchhull / trimesh)。
用法:
  ./foreground_hull/split_hull.py n3_scene0001        # 單一場景
  ./foreground_hull/split_hull.py 3                    # 整組 n3
  ./foreground_hull/split_hull.py n3_scene0001 --level 9
"""

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh

REPO = Path(__file__).resolve().parents[1]
CAPTURES = REPO / "data" / "captures"
FG_ROOT = REPO / "data" / "eval" / "foreground"

# ── 工作空間幾何(寫死,完全不用 depth)──────────────────────────────────────────
# 物體分布:中心 [0.35,0,0]、工作空間半徑 0.35m;桌面 z=0,物體都站桌上。
# torchhull 雕刻區是正立方體,邊長取水平跨度(2*半徑)+ 邊距,角點 = 最小角。
WS_CENTER_X = 0.35
WS_CENTER_Y = 0.0
WS_RADIUS = 0.35
TABLE_Z = 0.0
MARGIN = 0.05
HFOV_RAD = 1.4746          # IntelRealsense D455 水平視場(與拍攝設定一致)

CUBE_LENGTH = 2 * WS_RADIUS + 2 * MARGIN                 # 0.80 m
CUBE_CORNER_BFL = [
    WS_CENTER_X - WS_RADIUS - MARGIN,                    # x_min = -0.05
    WS_CENTER_Y - WS_RADIUS - MARGIN,                    # y_min = -0.40
    TABLE_Z - MARGIN,                                    # z_min = -0.05
]

# ── 連通元件分離參數 ──────────────────────────────────────────────────────────
MIN_VERTS = 50          # 小於此頂點數 → 視為雜訊碎塊
GHOST_VOL_FRAC = 0.05   # 體積 < 最大塊的此比例 → 標記疑似鬼影/碎塊


# ── 位姿 / 內參 / transform(與 build_torchhull 一致)──────────────────────────
def rpy_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float32,
    )


def load_pose(pose_path: Path):
    import json
    meta = json.loads(pose_path.read_text(encoding="utf-8"))
    # 新資料 view_XX_pose.json 把相機位姿包在 "camera" 下;舊資料直接在頂層
    if "position_m" not in meta and isinstance(meta.get("camera"), dict):
        meta = meta["camera"]
    position = np.array(
        [meta["position_m"]["x"], meta["position_m"]["y"], meta["position_m"]["z"]],
        dtype=np.float32,
    )
    rot = meta["rotation_rpy_rad"]
    camera_to_world = rpy_to_rotation_matrix(rot["roll"], rot["pitch"], rot["yaw"])
    return position, camera_to_world


def build_intrinsics(width: int, height: int, hfov_rad: float) -> np.ndarray:
    fx = width / (2.0 * math.tan(hfov_rad / 2.0))
    return np.array(
        [[fx, 0.0, width / 2.0], [0.0, fx, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def make_torchhull_transform(intrinsics, camera_position_world, camera_to_world):
    # Webots body frame (x=前, y=左, z=上) → OpenCV optical (x=右, y=下, z=前)
    body_to_opencv = np.array(
        [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]], dtype=np.float32
    )
    world_to_camera = np.eye(4, dtype=np.float32)
    rotation_world_to_opencv = body_to_opencv @ camera_to_world.T
    world_to_camera[:3, :3] = rotation_world_to_opencv
    world_to_camera[:3, 3] = -rotation_world_to_opencv @ camera_position_world
    intrinsics_4x4 = np.eye(4, dtype=np.float32)
    intrinsics_4x4[:3, :3] = intrinsics
    return intrinsics_4x4 @ world_to_camera


# ── 雕殼(固定 cube,不用 depth)────────────────────────────────────────────────
def carve_foreground(scene: str, level: int, device: torch.device) -> trimesh.Trimesh:
    group = scene.split("_")[0]
    scene_dir = CAPTURES / f"multi_{group}" / scene
    fg_dir = FG_ROOT / scene
    pose_paths = sorted(
        p for p in scene_dir.glob("view_*_pose.json") if p.name != "scene_objects_pose.json"
    )
    if not pose_paths:
        sys.exit(f"找不到位姿檔: {scene_dir}/view_*_pose.json")

    masks, transforms, used, skipped = [], [], [], []
    intrinsics = None
    for pose_path in pose_paths:
        view = pose_path.stem.removesuffix("_pose")
        mask_path = fg_dir / f"{view}_mask_foreground.png"
        if not mask_path.is_file():
            skipped.append(view); continue
        mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            skipped.append(view); continue
        mask = (mask_img > 127).astype(np.float32)
        if mask.sum() == 0.0:               # 空前景 → 不納入,避免削空整個 hull
            skipped.append(view); continue
        if intrinsics is None:
            h, w = mask.shape
            intrinsics = build_intrinsics(w, h, HFOV_RAD)
        position, camera_to_world = load_pose(pose_path)
        masks.append(mask[..., None])
        transforms.append(make_torchhull_transform(intrinsics, position, camera_to_world))
        used.append(view)

    if not masks:
        sys.exit(f"{scene} 無有效前景遮罩(先跑 make_foreground.py)")

    import torchhull
    masks_t = torch.from_numpy(np.stack(masks, 0)).to(device=device, dtype=torch.float32)
    transforms_t = torch.from_numpy(np.stack(transforms, 0)).to(device=device, dtype=torch.float32)
    verts, faces = torchhull.visual_hull(
        masks=masks_t,
        transforms=transforms_t,
        level=level,
        cube_corner_bfl=CUBE_CORNER_BFL,
        cube_length=CUBE_LENGTH,
        masks_partial=False,
        transforms_convention="opencv",
        unique_verts=True,
    )
    verts = verts.detach().cpu().numpy()
    faces = faces.detach().cpu().numpy()
    if verts.size == 0 or faces.size == 0:
        sys.exit(f"{scene} 雕出空 hull(檢查前景遮罩 / cube 是否罩到物體)")

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.export(str(fg_dir / "visual_hull_foreground.obj"))
    print(f"  雕殼: 用 {len(used)} views, 跳過 {len(skipped)} → "
          f"verts={len(verts)}  (cube 邊長 {CUBE_LENGTH:.2f}m, level {level})")
    return mesh


# ── 連通元件分物體 ────────────────────────────────────────────────────────────
def split_mesh(scene: str, mesh: trimesh.Trimesh):
    out_dir = FG_ROOT / scene / "components"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("obj_*.obj"):
        old.unlink()

    comps = mesh.split(only_watertight=False)
    comps = [c for c in comps if len(c.vertices) >= MIN_VERTS]
    comps.sort(key=lambda c: abs(c.volume), reverse=True)
    if not comps:
        sys.exit("沒有有效連通元件")
    max_vol = abs(comps[0].volume) or 1.0

    lines = [f"scene: {scene}", f"連通元件數(過濾後): {len(comps)}",
             f"cube: corner={CUBE_CORNER_BFL} length={CUBE_LENGTH:.3f}m", ""]
    for i, c in enumerate(comps):
        ctr = c.bounds.mean(axis=0)
        ext = c.extents
        vol = abs(c.volume)
        ghost = "  <- 疑似鬼影/碎塊" if vol < GHOST_VOL_FRAC * max_vol else ""
        c.export(str(out_dir / f"obj_{i:03d}.obj"))
        lines.append(f"obj_{i:03d}: verts={len(c.vertices):6d} vol={vol*1e6:8.1f}cm^3 "
                     f"中心=({ctr[0]:+.3f},{ctr[1]:+.3f},{ctr[2]:+.3f}) "
                     f"尺寸=({ext[0]:.3f}x{ext[1]:.3f}x{ext[2]:.3f}){ghost}")
    report = "\n".join(lines)
    (FG_ROOT / scene / "report.txt").write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\n→ {out_dir}/obj_*.obj、report.txt")


def resolve_scenes(targets):
    scenes = []
    for a in targets:
        if "scene" in a:
            scenes.append(a)
        else:
            scenes += [d.name for d in sorted((CAPTURES / f"multi_n{a}").glob(f"n{a}_scene*"))]
    return scenes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="*", default=["n3_scene0001"],
                    help="場景名(n3_scene0001)或組號(3)")
    ap.add_argument("--level", type=int, default=9, help="torchhull octree level(cube 變大,預設提一階補解析度)")
    ap.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = ap.parse_args()
    targets = args.targets or ["n3_scene0001"]

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    scenes = resolve_scenes(targets)
    if not scenes:
        sys.exit("沒有場景")
    for scene in scenes:
        print(f"\n== {scene} ==")
        mesh = carve_foreground(scene, args.level, device)
        split_mesh(scene, mesh)


if __name__ == "__main__":
    main()
