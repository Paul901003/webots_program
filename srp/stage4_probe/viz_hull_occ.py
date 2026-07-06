#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""viz_hull_occ.py — 同一張圖疊「SAM 遮罩」與「hull 重投影遮罩」。

對任一場景的每個視角:
  - hull 重投影遮罩 = 每個 instance 的體素投影 + 閉運算成實心輪廓(完整輪廓,含被擋部分),各 instance 一色填色;
  - SAM 遮罩 = sam_only class-agnostic 全部遮罩,以白色輪廓疊上(只切到可見部分)。
hull 填色超出 SAM 輪廓的部分 = 投影得到、但畫面看不到(被擋/重建多出)的區域。不需指定目標物,任何場景可跑。
用法: ./srp/stage4_probe/viz_hull_occ.py <scene> [scene2 ...]   不給場景則跑 srp_hull 下全部有 instances 的場景。
"""
import glob
import sys
from pathlib import Path

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
from matplotlib import font_manager   # noqa: E402

_FP = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"
try:
    font_manager.fontManager.addfont(_FP)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=_FP).get_name()
except Exception:
    pass

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srp" / "io"))
import camera as cam            # noqa: E402

HULL = REPO / "data" / "eval" / "srp_hull"
SAM = REPO / "data" / "eval" / "sam_only"
CAPTURES = REPO / "data" / "captures"
OUT = REPO / "data" / "eval" / "_diag" / "occ_viz"
DS = 2


def centers(occ, gm, vs):
    i, j, k = np.nonzero(occ)
    return gm + (np.stack([i, j, k], 1) + 0.5) * vs


def raster(c, K, Rwc, t, H, W):
    """hull 體素重投影 → 閉運算補洞的實心遮罩。"""
    X = c @ Rwc.T + t; z = X[:, 2]; ok = z > 1e-9; zz = np.where(ok, z, 1.0)
    u = np.round(K[0, 0] * X[:, 0] / zz + K[0, 2]).astype(int)
    v = np.round(K[1, 1] * X[:, 1] / zz + K[1, 2]).astype(int)
    inb = ok & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    mask = np.zeros((H, W), np.uint8)
    mask[v[inb], u[inb]] = 1
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)).astype(bool)


def load_sam(scene, view):
    d = SAM / scene / view / "masks"
    return [cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) > 127 for f in sorted(d.glob("mask_*.png"))]


def viz(scene):
    hp = HULL / scene / "hull.npz"; ip = HULL / scene / "instances.npz"
    if not (hp.is_file() and ip.is_file()):
        print(f"[skip] {scene}: 缺 hull/instances"); return
    z = np.load(hp); gm = z["grid_min"]; vs = float(z["voxel_size"]); shape = z["occupancy"].shape
    labels = np.load(ip)["labels"]
    insts = [k for k in range(1, int(labels.max()) + 1) if (labels == k).any()]
    if not insts:
        print(f"[skip] {scene}: 無 instance"); return
    cents = {k: centers(labels == k, gm, vs) for k in insts}
    cmap = plt.get_cmap("tab10")
    colors = {k: np.array(cmap((i % 10)))[:3] * 255 for i, k in enumerate(insts)}

    grp = scene.split("_")[0]; sdir = CAPTURES / f"multi_{grp}" / scene
    poses = sorted(sdir.glob("view_*_pose.json"))
    panels = []
    for pf in poses:
        v = pf.name.split("_pose")[0]
        rgb = cv2.imread(str(sdir / f"{v}.png"))
        if rgb is None:
            continue
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        C, Rb = cam.load_pose(pf); Rwc, t = cam.pose_to_w2c(C, Rb); K = cam.intrinsics(W, H)
        ov = rgb.astype(float)
        for k in insts:                                   # hull 重投影:各 instance 半透明填色
            m = raster(cents[k], K, Rwc, t, H, W)
            ov[m] = 0.55 * ov[m] + 0.45 * colors[k]
        for s in load_sam(scene, v):                      # SAM 遮罩:白色輪廓疊上
            if s.shape != (H, W):
                s = cv2.resize(s.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
            cnts, _ = cv2.findContours(s.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(ov, cnts, -1, (255, 255, 255), 2)
        panels.append((v, np.clip(ov, 0, 255).astype(np.uint8)[::DS, ::DS]))

    if not panels:
        print(f"[skip] {scene}: 無影像"); return
    n = len(panels); ncol = 4; nrow = (n + ncol - 1) // ncol
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.0 * nrow), squeeze=False)
    for ax, (v, img) in zip(axes.ravel(), panels):
        ax.imshow(img); ax.set_title(v, fontsize=8); ax.axis("off")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(f"{scene}  填色=各 instance hull 重投影  白輪廓=SAM 遮罩(可見部分)", fontsize=11)
    fig.tight_layout()
    p = OUT / f"{scene}.png"; fig.savefig(p, dpi=110); plt.close(fig)
    print(f"→ {p}")


def main():
    scenes = sys.argv[1:]
    if not scenes:
        scenes = sorted(Path(p).parent.name for p in glob.glob(str(HULL / "*" / "instances.npz")))
    for s in scenes:
        try:
            viz(s)
        except Exception as e:
            print(f"[err] {s}: {e}")


if __name__ == "__main__":
    main()
