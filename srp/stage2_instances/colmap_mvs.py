#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""colmap_mvs.py — 純 RGB dense MVS(給 GT 位姿只做 MVS),用於對比 visual hull。

流程(位姿固定不估,只做光度重建):
  影像 + 前景 mask → extract_features → match_exhaustive → 建已知位姿 recon
  → triangulate_points → undistort_images → patch_match_stereo(CUDA) → stereo_fusion。
輸出 data/eval/colmap_mvs/<scene>/fused.ply(+ work/ 中間檔)。可視化: worlds/srp_colmap_viz.wbt。

前提:pip install pycolmap-cuda12(CUDA dense)。用法: ./colmap_mvs.py <scene> [scene2 ...]
env: SAM_ROOT(sam_only_fast) CAPTURES_ROOT(captures_fast)
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srp" / "io"))
import camera as cam           # noqa: E402
import masks as MK             # noqa: E402
import viewpoints as VP        # noqa: E402  (A-3 selected_n{N},與 stage1/2 共用)

CAPTURES = Path(os.environ.get("CAPTURES_ROOT", str(REPO / "data" / "captures")))
SAM_ROOT = Path(os.environ.get("SAM_ROOT", str(REPO / "data" / "eval" / "sam_only")))
OUT_ROOT = REPO / "data" / "eval" / "colmap_mvs"


def process(scene, num_views=None, max_size=-1, tag=""):
    group = scene.split("_")[0]
    CAP = CAPTURES / f"multi_{group}" / scene
    SAM = SAM_ROOT / scene
    out = (OUT_ROOT / scene / tag) if tag else (OUT_ROOT / scene)
    work = out / "work"
    IMG = work / "images"; MASK = work / "masks"; DB = work / "database.db"; SPARSE = work / "sparse"; DENSE = work / "dense"
    if work.exists():
        shutil.rmtree(work)
    for d in (IMG, MASK, SPARSE):
        d.mkdir(parents=True, exist_ok=True)

    # 1. 影像 + 前景 mask(COLMAP mask 命名: <image_name>.png,白=保留)
    want = VP.selected_view_names(num_views) if num_views else None
    fg = None
    for vd in sorted(SAM.glob("view_*")):
        if want is not None and vd.name not in want:
            continue
        pf = CAP / f"{vd.name}_pose.json"; img = CAP / f"{vd.name}.png"
        if not (pf.is_file() and img.is_file()):
            continue
        km = MK.kept_object_masks(vd)
        if not km:
            continue
        m = None
        for b, _ in km:
            m = b if m is None else (m | b)
        if m is None or not m.any():
            continue
        fg = m
        shutil.copy(img, IMG / f"{vd.name}.png")
        cv2.imwrite(str(MASK / f"{vd.name}.png.png"), (m.astype(np.uint8) * 255))
    if fg is None:
        print(f"[skip] {scene}: 無有效視角"); return None
    H, W = fg.shape
    K = cam.intrinsics(W, H)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # 2. 特徵抽取(已知內參 + mask)+ 匹配
    ro = pycolmap.ImageReaderOptions()
    ro.camera_model = "PINHOLE"; ro.camera_params = f"{fx},{fy},{cx},{cy}"; ro.mask_path = str(MASK)
    pycolmap.extract_features(DB, IMG, camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=ro, device=pycolmap.Device.auto)
    pycolmap.match_exhaustive(DB)

    # 3. 建已知位姿 reconstruction
    with pycolmap.Database.open(str(DB)) as db:
        cameras = db.read_all_cameras(); dbimgs = db.read_all_images()
    rec = pycolmap.Reconstruction()
    for c in cameras:
        rec.add_camera_with_trivial_rig(c)
    for im in dbimgs:
        C, Rb = cam.load_pose(CAP / f"{im.name[:-4]}_pose.json")
        Rwc, t = cam.pose_to_w2c(C, Rb)
        q = Rotation.from_matrix(Rwc).as_quat()          # [x,y,z,w]
        image = pycolmap.Image(name=im.name, camera_id=im.camera_id, image_id=im.image_id)
        rec.add_image_with_trivial_frame(image)
        rec.frame(im.image_id).set_cam_from_world(im.camera_id, pycolmap.Rigid3d(pycolmap.Rotation3d(q), t))
        rec.register_frame(im.image_id)

    # 4. 三角化(已知位姿)
    pycolmap.triangulate_points(rec, str(DB), str(IMG), str(SPARSE))

    # 5. dense: undistort → patch_match(CUDA) → fusion
    if DENSE.exists():
        shutil.rmtree(DENSE)
    pycolmap.undistort_images(str(DENSE), str(SPARSE), str(IMG))
    pmo = pycolmap.PatchMatchOptions()
    pmo.max_image_size = max_size          # -1=原尺寸;否則縮到長邊=max_size
    pycolmap.patch_match_stereo(str(DENSE), options=pmo)
    fused_model = DENSE / "fused_model"; fused_model.mkdir(exist_ok=True)
    fused = pycolmap.stereo_fusion(output_path=str(fused_model), workspace_path=str(DENSE))

    # 存 ply(保留)
    ply = out / "fused.ply"
    fused.export_PLY(str(ply))
    n = len(fused.points3D)
    print(f"[{scene}] dense 融合 {n} 點 → {ply}")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="+")
    ap.add_argument("--num-views", type=int, default=None, dest="num_views",
                    help="只用 A-3 selected_n{N} 挑的視角(如 12,對齊 srp_hull_v12);預設全用")
    ap.add_argument("--max-image-size", type=int, default=-1, dest="max_size",
                    help="dense patch_match 的影像長邊上限(-1=原1280;960/640/320=縮小加速)")
    ap.add_argument("--tag", default="", help="輸出子目錄(分開儲存不同設定,如 v6_s320)")
    args = ap.parse_args()
    for sc in args.scenes:
        try:
            process(sc, args.num_views, args.max_size, args.tag)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"[err] {sc}: {e}")


if __name__ == "__main__":
    main()
