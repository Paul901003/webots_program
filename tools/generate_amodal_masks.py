#!/home/cho/.pyenv/versions/3.10.10/bin/python3
"""generate_amodal_masks.py — 產生「完整(amodal)」GT 遮罩:每物體單獨渲染,無手臂、無其他物體。

對每場景每視角(用 <view>_pose.json 的相機,與拍攝一致)、每個 GT 物體:
  只把該物體 mesh 放進 pyrender scene → SEG 渲染 → 該物體的完整輪廓(不被任何東西遮擋)。
輸出 COCO: data/labels/<scene>/amodal/annotations.json(不覆蓋 actual/ 的 modal 版)。
重用 generate_labels 的相機/位姿/mesh 數學。需 pyrender 環境(3.10.10)。
用法: ./tools/generate_amodal_masks.py n3_scene0030 [更多場景 或 組號 3 4 5]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pyrender
from pycocotools import mask as mask_utils

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import generate_labels as GL  # 重用 ycb_center / camera_intrinsics / webots_camera_pose / load_ycb_mesh / _axis_angle_to_mat / 常數

CAPTURES = REPO / "data" / "captures"
LABELS = REPO / "data" / "labels"
ASSETS = str(REPO / "urdfs" / "ycb_assets")


def resolve_scenes(targets):
    out = []
    for a in targets:
        if "scene" in str(a):
            out.append(a)
        else:
            out += [d.name for d in sorted((CAPTURES / f"multi_n{a}").glob(f"n{a}_scene*"))]
    return out


def load_pose(p):
    m = json.loads(Path(p).read_text(encoding="utf-8"))
    if "position_m" not in m and isinstance(m.get("camera"), dict):
        m = m["camera"]
    pos = m["position_m"]; r = m["rotation_rpy_rad"]
    return (np.array([pos["x"], pos["y"], pos["z"]], float),
            (r["roll"], r["pitch"], r["yaw"]))


def render_one(renderer, mesh, tf, cam_pose, K):
    scene = pyrender.Scene(bg_color=[0, 0, 0, 255])
    node = scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False), pose=tf)
    seg_map = {node: (0, 1, 0)}
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    scene.add(pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.05, zfar=10.0), pose=cam_pose)
    seg, _ = renderer.render(scene, flags=pyrender.RenderFlags.SEG, seg_node_map=seg_map)
    return (seg[:, :, 1] == 1).astype(np.uint8)


def process_scene(scene, renderer):
    g = scene.split("_")[0]
    sdir = CAPTURES / f"multi_{g}" / scene
    mani = sdir / "scene_manifest.json"
    if not mani.is_file():
        print(f"[skip] {scene}: 無 manifest"); return
    objs = json.loads(mani.read_text())["actual"]["viewpoints"][0]["objects"]
    views = sorted(sdir.glob("view_*_pose.json"))
    K = GL.camera_intrinsics()

    names = sorted({o["name"] for o in objs})
    cat_id = {nm: i + 1 for i, nm in enumerate(names)}
    mesh_cache = {nm: GL.load_ycb_mesh(ASSETS, nm) for nm in names}   # 每物體只讀一次
    coco = {"images": [], "annotations": [],
            "categories": [{"id": cat_id[nm], "name": nm} for nm in names]}
    ann = 1
    for vp in views:
        vname = vp.stem.replace("_pose", "")          # view_01
        vid = int(vname.split("_")[1])
        cam_pos, rpy = load_pose(vp)
        cam_pose = GL.webots_camera_pose(cam_pos, rpy)
        coco["images"].append({"id": vid, "file_name": f"images/{vname}.png",
                               "width": GL.CAM_WIDTH, "height": GL.CAM_HEIGHT})
        for o in objs:
            nm = o["name"]
            mesh = mesh_cache.get(nm)
            if mesh is None:
                continue
            p = o["position_m"]; pos = np.array([p[0], p[1], p[2]] if isinstance(p, list) else [p["x"], p["y"], p["z"]], float)
            aa = o.get("rotation_axis_angle", [0, 1, 0, 0])
            R = GL._axis_angle_to_mat(np.array(aa[:3], float), aa[3])
            tf = np.eye(4); tf[:3, :3] = R; tf[:3, 3] = pos - R @ GL.ycb_center(nm)
            mask = render_one(renderer, mesh, tf, cam_pose, K)
            if mask.sum() == 0:
                continue
            rle = mask_utils.encode(np.asfortranarray(mask)); rle["counts"] = rle["counts"].decode("utf-8")
            coco["annotations"].append({"id": ann, "image_id": vid, "category_id": cat_id[nm],
                                        "segmentation": rle, "area": float(mask_utils.area(rle)),
                                        "bbox": mask_utils.toBbox(rle).tolist(), "iscrowd": 0})
            ann += 1
    out = LABELS / scene / "amodal"
    out.mkdir(parents=True, exist_ok=True)
    (out / "annotations.json").write_text(json.dumps(coco, ensure_ascii=False), encoding="utf-8")
    print(f"[{scene}] views {len(views)} objs {len(objs)} → {ann-1} amodal 遮罩 → {out}/annotations.json")


def main():
    targets = sys.argv[1:] or ["n3_scene0030"]
    scenes = resolve_scenes(targets)
    renderer = pyrender.OffscreenRenderer(GL.CAM_WIDTH, GL.CAM_HEIGHT)   # 共用一個,避免每次建/刪 EGL context
    try:
        for i, sc in enumerate(scenes, 1):
            print(f"[{i}/{len(scenes)}]", end=" ")
            try:
                process_scene(sc, renderer)
            except Exception as e:
                import traceback; traceback.print_exc(); print(f"[err] {sc}: {e}")
    finally:
        renderer.delete()


if __name__ == "__main__":
    main()
