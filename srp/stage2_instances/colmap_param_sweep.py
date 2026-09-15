#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""colmap_param_sweep.py — 固定 v12_s1280,單因子掃描 dense 參數,量「加速 vs 品質」。

複製 colmap_mvs.py 的 sparse 準備(不動原檔,見 dont-overwrite 原則)。每場只做一次
sparse+undistort,之後只重跑會變的 dense(patch_match + fusion),隔離 dense 參數的加速效果。

品質(vs GT mesh,GT 位姿重建→同世界座標、免對齊):
  accuracy    = 每個重建點到 GT mesh 表面最近距離的中位數(mm,越小越好)
  completeness= GT 表面取樣點中,τ=5mm 內被重建點覆蓋的比例(%,越高越好)

掃描變體(對照 baseline 各動一個):
  geom_off  geom_consistency True→False(fusion 改 photometric)   iter3  num_iterations 5→3
  wstep2    window_step 1→2                                        samp10 num_samples 15→10
  wrad3     window_radius 5→3                                      depth  depth_min/max auto→[0.30,0.85]

env: SAM_ROOT(sam_only_fast) CAPTURES_ROOT(captures_fast)
用法: ./colmap_param_sweep.py stack3_scene0001 stack4_scene0001 stack5_scene0001
輸出: data/eval/colmap_mvs/param_sweep.csv(scene,variant,dense_sec,n_points,accuracy_mm,completeness_pct)
"""
import argparse
import csv
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pycolmap
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srp" / "io"))
import camera as cam            # noqa: E402
import masks as MK              # noqa: E402
import viewpoints as VP         # noqa: E402
import eval_mesh as EM          # noqa: E402  (load_mesh/ycb_center/aa_to_mat/gt_objects)

CAPTURES = Path(os.environ.get("CAPTURES_ROOT", str(REPO / "data" / "captures")))
SAM_ROOT = Path(os.environ.get("SAM_ROOT", str(REPO / "data" / "eval" / "sam_only")))
OUT_ROOT = REPO / "data" / "eval" / "colmap_mvs"

NUM_VIEWS = 12
MAX_SIZE = 1280
TAU = 0.005            # completeness 覆蓋門檻(公尺)
N_GT_SAMPLE = 200000   # GT mesh 表面取樣點數


# ── sparse 準備(複製 colmap_mvs.process 的 step 1–6,只到 undistort)──
def prepare_workspace(scene):
    group = scene.split("_")[0]
    CAP = CAPTURES / f"multi_{group}" / scene
    SAM = SAM_ROOT / scene
    out = OUT_ROOT / scene / "_sweep"
    work = out / "work"
    IMG = work / "images"; MASK = work / "masks"; DB = work / "database.db"
    SPARSE = work / "sparse"; DENSE = work / "dense"
    if work.exists():
        shutil.rmtree(work)
    for d in (IMG, MASK, SPARSE):
        d.mkdir(parents=True, exist_ok=True)

    want = VP.selected_view_names(NUM_VIEWS)
    fg = None
    for vd in sorted(SAM.glob("view_*")):
        if vd.name not in want:
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
        raise RuntimeError(f"{scene}: 無有效視角")
    H, W = fg.shape
    K = cam.intrinsics(W, H)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    ro = pycolmap.ImageReaderOptions()
    ro.camera_model = "PINHOLE"; ro.camera_params = f"{fx},{fy},{cx},{cy}"; ro.mask_path = str(MASK)
    pycolmap.extract_features(DB, IMG, camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=ro, device=pycolmap.Device.auto)
    pycolmap.match_exhaustive(DB)

    with pycolmap.Database.open(str(DB)) as db:
        cameras = db.read_all_cameras(); dbimgs = db.read_all_images()
    rec = pycolmap.Reconstruction()
    for c in cameras:
        rec.add_camera_with_trivial_rig(c)
    for im in dbimgs:
        C, Rb = cam.load_pose(CAP / f"{im.name[:-4]}_pose.json")
        Rwc, t = cam.pose_to_w2c(C, Rb)
        q = Rotation.from_matrix(Rwc).as_quat()
        image = pycolmap.Image(name=im.name, camera_id=im.camera_id, image_id=im.image_id)
        rec.add_image_with_trivial_frame(image)
        rec.frame(im.image_id).set_cam_from_world(im.camera_id, pycolmap.Rigid3d(pycolmap.Rotation3d(q), t))
        rec.register_frame(im.image_id)
    pycolmap.triangulate_points(rec, str(DB), str(IMG), str(SPARSE))

    if DENSE.exists():
        shutil.rmtree(DENSE)
    pmo0 = pycolmap.PatchMatchOptions(); pmo0.max_image_size = MAX_SIZE  # undistort 尺寸一致
    pycolmap.undistort_images(str(DENSE), str(SPARSE), str(IMG))
    return DENSE


def clear_stereo(DENSE):
    """清掉上一輪 depth/normal maps,強制 patch_match 重算(否則 COLMAP 見檔即跳過)。"""
    st = DENSE / "stereo"
    for sub in ("depth_maps", "normal_maps", "consistency_graphs"):
        d = st / sub
        if d.exists():
            for f in d.iterdir():
                if f.is_file():
                    f.unlink()


# ── dense(patch_match + fusion),回傳點雲 + 秒數 ──
def make_patch_opts(variant):
    o = pycolmap.PatchMatchOptions()
    o.max_image_size = MAX_SIZE
    if variant == "geom_off":
        o.geom_consistency = False
    elif variant == "wstep2":
        o.window_step = 2
    elif variant == "iter3":
        o.num_iterations = 3
    elif variant == "samp10":
        o.num_samples = 10
    elif variant == "wrad3":
        o.window_radius = 3
    elif variant == "depth":
        o.depth_min = 0.30; o.depth_max = 0.85
    return o


def run_dense(DENSE, variant):
    clear_stereo(DENSE)
    o = make_patch_opts(variant)
    input_type = "photometric" if variant == "geom_off" else "geometric"
    t0 = time.time()
    pycolmap.patch_match_stereo(str(DENSE), options=o)
    fused_model = DENSE / "fused_model";
    if fused_model.exists():
        shutil.rmtree(fused_model)
    fused_model.mkdir(exist_ok=True)
    fused = pycolmap.stereo_fusion(output_path=str(fused_model), workspace_path=str(DENSE),
                                   input_type=input_type)
    sec = time.time() - t0
    pts = np.array([p.xyz for p in fused.points3D.values()]) if len(fused.points3D) else np.zeros((0, 3))
    return pts, sec


# ── GT mesh 表面點(世界座標,同 solid_mesh_occ 的位姿變換)──
def gt_surface_points(scene):
    allp = []
    for o in EM.gt_objects(scene):
        name = o["name"]; m = EM.load_mesh(name)
        if m is None:
            continue
        aa = o.get("rotation_axis_angle", [0, 1, 0, 0])
        R = EM.aa_to_mat(aa[:3], aa[3])
        Vw = (m.vertices - EM.ycb_center(name)) @ R.T + np.asarray(o["position_m"], float)
        mw = trimesh.Trimesh(vertices=Vw, faces=m.faces, process=False)
        pts, _ = trimesh.sample.sample_surface(mw, N_GT_SAMPLE // max(1, len(EM.gt_objects(scene))))
        allp.append(np.asarray(pts))
    return np.vstack(allp) if allp else np.zeros((0, 3))


def eval_quality(P, GT):
    if len(P) == 0 or len(GT) == 0:
        return float("nan"), float("nan")
    d_acc, _ = cKDTree(GT).query(P)              # 每重建點→GT 最近
    d_comp, _ = cKDTree(P).query(GT)             # 每 GT 點→重建 最近
    accuracy_mm = float(np.median(d_acc)) * 1000
    completeness = float((d_comp < TAU).mean()) * 100
    return accuracy_mm, completeness


VARIANTS = ["baseline", "geom_off", "wstep2", "iter3", "samp10", "wrad3", "depth"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="+")
    args = ap.parse_args()
    csv_path = OUT_ROOT / "param_sweep.csv"
    rows = []
    for sc in args.scenes:
        print(f"\n######## {sc}:sparse 準備(一次)########")
        DENSE = prepare_workspace(sc)
        GT = gt_surface_points(sc)
        print(f"  GT 表面點 {len(GT)}")
        for v in VARIANTS:
            try:
                pts, sec = run_dense(DENSE, v)
                acc, comp = eval_quality(pts, GT)
                rows.append([sc, v, round(sec, 1), len(pts), round(acc, 2), round(comp, 1)])
                print(f"  [{v:>8}] {sec:6.1f}s  {len(pts):>7} 點  acc {acc:5.2f}mm  comp {comp:5.1f}%")
            except Exception as e:
                import traceback; traceback.print_exc(); print(f"  [err {v}] {e}")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene", "variant", "dense_sec", "n_points", "accuracy_mm", "completeness_pct"])
        w.writerows(rows)
    print(f"\n寫入 {csv_path}")


if __name__ == "__main__":
    main()
