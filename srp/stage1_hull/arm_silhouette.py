#!/home/cho/.pyenv/versions/3.10.10/bin/python3
"""arm_silhouette.py — 用 FK 算每視角「手臂+夾爪」剪影(供 visual hull 前景排除手臂)。

重用 tools/generate_labels.py 的 FK + 相機 + SEG render(與 GT 幾何一致,實測 vs GT ur5e IoU=1.0),
但只渲手臂/夾爪、讀 captures_fast 的 scene_manifest.json(joint_deg + camera,list 格式),
**不碰物體 GT**。輸出 data/eval/srp_arm_masks/<scene>/<view>_arm.png(白=手臂)。

用法: ./arm_silhouette.py n3_scene0001 / n3 / occ3 / (無參數=全部 captures_fast)
env: CAPTURES_ROOT(預設 captures_fast) ARM_MASK_ROOT(預設 data/eval/srp_arm_masks) FORCE
"""
import contextlib
import glob
import io
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pyrender
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
import generate_labels as GL  # noqa: E402

CAP = Path(os.environ.get("CAPTURES_ROOT", str(REPO / "data" / "captures_fast")))
OUT = Path(os.environ.get("ARM_MASK_ROOT", str(REPO / "data" / "eval" / "srp_arm_masks")))
FORCE = os.environ.get("FORCE") == "1"
K = GL.camera_intrinsics()


def render_arm(joint_deg, cam_pos, rpy):
    jr = [math.radians(d) for d in joint_deg]
    cam_pose = GL.webots_camera_pose(np.array(cam_pos, float), rpy)
    with contextlib.redirect_stdout(io.StringIO()):   # 吞掉 load_gripper_nodes 的除錯輸出
        nodes = (GL.load_robot_scene_nodes(GL.PROTO_MESH_DIR, jr)
                 + GL.load_gripper_nodes(GL.GRIPPER_MESH_DIR, jr, 0.0))
    scene = pyrender.Scene(bg_color=[0, 0, 0, 255])
    smap = {}
    for mesh, tf in nodes:
        node = scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False), pose=tf)
        smap[node] = GL.SEG_ROBOT
    scene.add(pyrender.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
                                        znear=0.01, zfar=10.0), pose=cam_pose)
    r = pyrender.OffscreenRenderer(GL.CAM_WIDTH, GL.CAM_HEIGHT)
    try:
        seg, _ = r.render(scene, flags=pyrender.RenderFlags.SEG, seg_node_map=smap)
    finally:
        r.delete()
    return (seg[:, :, 0] == 1).astype(np.uint8) * 255


def scene_dirs(targets):
    out = []
    if not targets:
        out = sorted(glob.glob(str(CAP / "multi_*" / "*_scene*")))
    for a in targets:
        if "scene" in a:
            g = a.split("_")[0]; out.append(str(CAP / f"multi_{g}" / a))
        else:
            g = f"n{a}" if a.isdigit() else a
            out += sorted(glob.glob(str(CAP / f"multi_{g}" / f"{g}_scene*")))
    return [Path(p) for p in out if Path(p).is_dir()]


def main():
    dirs = scene_dirs(sys.argv[1:])
    print(f"待處理 {len(dirs)} 場景 → {OUT}", flush=True)
    for i, sd in enumerate(dirs, 1):
        scene = sd.name
        man_path = sd / "scene_manifest.json"
        if not man_path.exists():
            print(f"  [略過] 無 manifest: {scene}", flush=True); continue
        odir = OUT / scene
        odir.mkdir(parents=True, exist_ok=True)
        vps = json.load(open(man_path))["actual"]["viewpoints"]
        n = 0
        for vp in vps:
            view = Path(vp["files"]["rgb"]).stem
            op = odir / f"{view}_arm.png"
            if op.exists() and not FORCE:
                continue
            cam = vp["camera"]
            mask = render_arm(vp["joint_deg"], cam["position_m"], cam["rotation_rpy_rad"])
            Image.fromarray(mask).save(str(op))
            n += 1
        if i % 20 == 0 or i == len(dirs):
            print(f"  [{i}/{len(dirs)}] {scene}: +{n} 視角", flush=True)
    print("手臂剪影完成。", flush=True)


if __name__ == "__main__":
    main()
