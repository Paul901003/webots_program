#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""viz_morph.py — hull occupancy 膨脹/侵蝕 互動檢視器(純檢視,不改任何檔)。

用途:把某場景某 hull 的 occupancy 用 pyvista 3D 檢視,滑桿即時調整
侵蝕/膨脹次數與 6/18/26 連通,每個 hull voxel 依「與 GT 真實 mesh 的關係」上色:
  - 鬼影  ghost    : voxel 八個角點「全部」落在所有 mesh 之外 → 半透明紅
  - 邊界  boundary : voxel 八個角點「有的在 mesh 內、有的在外」 → 半透明黃
  - 真實  real     : voxel 八個角點「全部」落在某 mesh 之內       → 半透明灰
(角點分類只看 voxel 位置,與 hull/morphology 無關 → 一次預算 class 場,
 滑桿只重算 hull 集合再查表,故即時。)

資料來源(不覆蓋、不修改):
  - hull occupancy : data/eval/<hull_root>/<scene>/hull.npz 的 occupancy/grid_min/voxel_size
  - 真實 mesh      : GT 位姿 + YCB mesh(與 eval_mesh 同一套轉換:gt_objects/load_mesh/
                     ycb_center/aa_to_mat),於同一網格上判定角點內外。視覺化「不套」GLOBAL_EXCLUDE。

一致性:mesh 與 hull 共用 hull.npz 的 grid_min / voxel_size / shape;三類 voxel 數
即時顯示,real+boundary+ghost == 當前 hull voxel 數。

用法:
  srp/stage2_instances/viz_morph.py <scene> <hull_root> [conn]      # 開視窗(需 pyvista)
  srp/stage2_instances/viz_morph.py <scene> <hull_root> [conn] --check   # 無視窗,印數字驗證
例:
  srp/stage2_instances/viz_morph.py stack4_scene0002 srp_hull_mv2_v12_am1 26

provenance: 本檔為新增檢視工具,不動 hull/eval 既有檔;morphology 只作用於記憶體中的
occupancy 副本,原 hull.npz 永不被寫回。
"""
import sys
import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage
import trimesh  # noqa: F401  (eval_mesh 轉換需要)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_mesh import gt_objects, load_mesh, ycb_center, aa_to_mat  # 同 eval 的 mesh 轉換

ROOT = HERE.parent.parent  # webots_program/
EVAL = ROOT / "data" / "eval"

CONN2RANK = {6: 1, 18: 2, 26: 3}
# 半透明 RGB(0..1);opacity 另外給
COLORS = {
    "ghost": (0.90, 0.15, 0.15),     # 紅
    "boundary": (0.95, 0.80, 0.10),  # 黃
    "real": (0.60, 0.60, 0.62),      # 灰
}


def load_hull(scene, hull_root):
    p = EVAL / hull_root / scene / "hull.npz"
    if not p.is_file():
        sys.exit(f"[錯誤] 找不到 hull:{p}")
    z = np.load(p, allow_pickle=True)
    occ = np.asarray(z["occupancy"], bool)
    gm = np.asarray(z["grid_min"], float)
    vs = float(z["voxel_size"])
    return occ, gm, vs, p


def build_world_meshes(scene):
    """回傳 [(name, trimesh(world))],與 eval_mesh.solid_mesh_occ 同一套轉換。"""
    out = []
    for o in gt_objects(scene):
        name = o["name"]
        m = load_mesh(name)
        if m is None:
            out.append((name, None))
            continue
        aa = o.get("rotation_axis_angle", [0, 1, 0, 0])
        R = aa_to_mat(aa[:3], aa[3])
        Vw = (m.vertices - ycb_center(name)) @ R.T + np.asarray(o["position_m"], float)
        out.append((name, trimesh.Trimesh(vertices=Vw, faces=m.faces, process=False)))
    return out


def corner_inside_lattice(meshes, gm, vs, shape):
    """角點格 (nx+1,ny+1,nz+1) 每點是否落在任一 mesh 內。只在 mesh 聯集 bbox(+1格)內
    實測 contains,bbox 外一律 False(=外),省成本。"""
    nx, ny, nz = shape
    C = np.zeros((nx + 1, ny + 1, nz + 1), bool)
    valid = [(n, m) for n, m in meshes if m is not None]
    if not valid:
        return C, 0
    # 聯集 bbox → 角點索引範圍
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for _, m in valid:
        b = m.bounds  # (2,3)
        lo = np.minimum(lo, b[0])
        hi = np.maximum(hi, b[1])
    i0 = np.floor((lo - gm) / vs).astype(int) - 1
    i1 = np.ceil((hi - gm) / vs).astype(int) + 1
    i0 = np.maximum(i0, 0)
    i1 = np.minimum(i1, [nx, ny, nz])
    if np.any(i1 < i0):
        return C, len(valid)
    ax = np.arange(i0[0], i1[0] + 1)
    ay = np.arange(i0[1], i1[1] + 1)
    az = np.arange(i0[2], i1[2] + 1)
    gx, gy, gz = np.meshgrid(ax, ay, az, indexing="ij")
    pts = np.stack([gx, gy, gz], -1).reshape(-1, 3) * vs + gm
    inside = np.zeros(len(pts), bool)
    for _, m in valid:
        try:
            inside |= m.contains(pts)
        except Exception:
            # contains 失敗(非水密等)→ 用 ray fallback 已在 trimesh 內部處理;真失敗則跳過該 mesh
            pass
    sub = inside.reshape(len(ax), len(ay), len(az))
    C[i0[0]:i1[0] + 1, i0[1]:i1[1] + 1, i0[2]:i1[2] + 1] = sub
    return C, len(valid)


def class_field(C, shape):
    """每 cell 八角點內數 n_in → 分類場:0=ghost,1=boundary,2=real。"""
    nx, ny, nz = shape
    n_in = np.zeros(shape, np.uint8)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                n_in += C[dx:dx + nx, dy:dy + ny, dz:dz + nz].astype(np.uint8)
    cls = np.ones(shape, np.uint8)          # 預設 boundary
    cls[n_in == 0] = 0                      # ghost
    cls[n_in == 8] = 2                      # real
    return cls


def morph(occ, erode, dilate, conn):
    st = ndimage.generate_binary_structure(3, CONN2RANK[conn])
    out = occ
    if erode > 0:
        out = ndimage.binary_erosion(out, st, iterations=erode, border_value=0)
    if dilate > 0:
        out = ndimage.binary_dilation(out, st, iterations=dilate, border_value=0)
    return out


def counts(hull, cls):
    idx = np.where(hull)
    c = cls[idx]
    return dict(ghost=int((c == 0).sum()),
                boundary=int((c == 1).sum()),
                real=int((c == 2).sum()),
                total=int(hull.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("hull_root")
    ap.add_argument("conn", nargs="?", type=int, default=26, choices=[6, 18, 26])
    ap.add_argument("--check", action="store_true", help="無視窗,印各設定的三類 voxel 數")
    args = ap.parse_args()

    occ, gm, vs, hp = load_hull(args.scene, args.hull_root)
    shape = occ.shape
    meshes = build_world_meshes(args.scene)
    n_no_mesh = sum(1 for _, m in meshes if m is None)
    C, n_mesh = corner_inside_lattice(meshes, gm, vs, shape)
    cls = class_field(C, shape)

    print(f"[hull] {hp}")
    print(f"  shape={shape} voxel_size={vs} occupancy={int(occ.sum())}")
    print(f"  GT 物體 {len(meshes)} 個,有 mesh {n_mesh}、無 mesh {n_no_mesh}"
          + ("  ⚠ 無 mesh 者其 voxel 會被判成鬼影" if n_no_mesh else ""))

    if args.check:
        base = counts(occ, cls)
        print(f"[check] 原始(e0 d0)三類:{base}"
              f"  → real+boundary+ghost={base['real']+base['boundary']+base['ghost']}"
              f" == total {base['total']}")
        for e, d in [(1, 0), (0, 1), (1, 1), (2, 0), (0, 2)]:
            h = morph(occ, e, d, args.conn)
            print(f"[check] conn{args.conn} e{e} d{d}: {counts(h, cls)}")
        return

    # ---- 視窗模式 ----
    try:
        import pyvista as pv
    except ModuleNotFoundError:
        sys.exit("[錯誤] 需要 pyvista:請先在 webots_visual_hull 環境安裝(pip install pyvista)。"
                 "或用 --check 先看數字。")

    centers_cube = pv.Cube(x_length=vs, y_length=vs, z_length=vs)
    state = {"e": 0, "d": 0, "conn": args.conn}
    pl = pv.Plotter()
    pl.set_background("white")

    def rebuild():
        hull = morph(occ, state["e"], state["d"], state["conn"])
        cc = counts(hull, cls)
        pl.remove_actor("hull_g", render=False)
        pl.remove_actor("hull_b", render=False)
        pl.remove_actor("hull_r", render=False)
        idx = np.argwhere(hull)
        if len(idx):
            world = idx * vs + gm + vs / 2.0  # voxel 中心
            cvals = cls[hull]
            for key, code, opac in (("real", 2, 0.35), ("boundary", 1, 0.55), ("ghost", 0, 0.75)):
                sel = world[cvals == code]
                if len(sel):
                    glyph = pv.PolyData(sel).glyph(geom=centers_cube, scale=False, orient=False)
                    pl.add_mesh(glyph, color=COLORS[key], opacity=opac,
                                name=f"hull_{key[0]}")
        pl.add_text(
            f"{args.scene}  {args.hull_root}\n"
            f"conn={state['conn']}  erode={state['e']}  dilate={state['d']}\n"
            f"鬼影(紅)={cc['ghost']}  邊界(黃)={cc['boundary']}  真實(灰)={cc['real']}"
            f"  總={cc['total']}",
            name="info", position="upper_left", font_size=10, color="black")
        pl.render()

    def set_e(v):
        state["e"] = int(round(v)); rebuild()

    def set_d(v):
        state["d"] = int(round(v)); rebuild()

    def set_conn(v):
        state["conn"] = {0: 6, 1: 18, 2: 26}[int(round(v))]; rebuild()

    pl.add_slider_widget(set_e, [0, 3], value=0, title="erode 次數",
                         pointa=(0.02, 0.10), pointb=(0.32, 0.10), fmt="%.0f")
    pl.add_slider_widget(set_d, [0, 3], value=0, title="dilate 次數",
                         pointa=(0.35, 0.10), pointb=(0.65, 0.10), fmt="%.0f")
    pl.add_slider_widget(set_conn, [0, 2], value={6: 0, 18: 1, 26: 2}[args.conn],
                         title="連通 0:6 1:18 2:26",
                         pointa=(0.68, 0.10), pointb=(0.98, 0.10), fmt="%.0f")
    rebuild()
    pl.add_axes()
    pl.show()


if __name__ == "__main__":
    main()
