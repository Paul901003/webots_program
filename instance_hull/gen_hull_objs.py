#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""gen_hull_objs.py — 為「某方法某場景」產生 hull(+GT)的 .obj 與 manifest.json(供 supervisor 載入)。

每 instance:優先用 association 存的 occ.npz 真實元件殼;無則從遮罩重雕。marching cubes(PyMCubes)→ 世界座標 .obj。
GT 也雕一份(半透明灰對照)。輸出到 --out 目錄:inst_NN.obj / gt_<name>.obj / manifest.json。
用法: ./instance_hull/gen_hull_objs.py n3_scene0030 --root v3/instance_hull_voxel --out <dir> [--no-gt]
"""
import argparse, json, math, sys
from pathlib import Path
import cv2, numpy as np, mcubes
from pycocotools import mask as mu
sys.path.insert(0, str(Path(__file__).resolve().parent))
import hull_common as HC
import eval_clip_match as E

import sys as _s, pathlib as _pl; _s.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "srp" / "io")); from labels import LABELS  # data/labels 分層(類別/數量/場景)


def load_gt_amodal(scene):
    p = LABELS / scene / "amodal" / "annotations.json"
    if not p.is_file():
        return None
    coco = json.loads(p.read_text()); cats = {c["id"]: c["name"] for c in coco["categories"]}
    out = {}
    for a in coco["annotations"]:
        nm = cats[a["category_id"]]
        if nm == "ur5e":
            continue
        rle = a["segmentation"]
        if isinstance(rle["counts"], str):
            rle = {"counts": rle["counts"].encode(), "size": rle["size"]}
        out.setdefault(nm, {})[f"view_{int(a['image_id']):02d}"] = mu.decode(rle).astype(bool)
    return out

YCB_ASSETS = HC.REPO / "urdfs" / "ycb_assets"
_GEO = json.loads((HC.REPO / "controllers" / "ycb_supervisor" / "ycb_geometries.json").read_text(encoding="utf-8"))


def axis_angle_to_mat(axis, ang):
    a = np.asarray(axis, float); n = np.linalg.norm(a)
    if n < 1e-9 or abs(ang) < 1e-12:
        return np.eye(3)
    x, y, z = a / n; c, s, t = math.cos(ang), math.sin(ang), 1 - math.cos(ang)
    return np.array([[t*x*x+c, t*x*y-s*z, t*x*z+s*y],
                     [t*x*y+s*z, t*y*y+c, t*y*z-s*x],
                     [t*x*z-s*y, t*y*z+s*x, t*z*z+c]])


def ycb_center(name):
    c = _GEO.get(name, {}).get("center", {"x": 0, "y": 0, "z": 0})
    return np.array([c["x"], c["y"], c["z"]])


def ycb_mesh(name):
    base = YCB_ASSETS / name / "google_16k"
    for f in ("textured.obj", "nontextured.ply", "nontextured.stl"):
        if (base / f).is_file():
            return base / f
    return None


def ycb_items(scene):
    """讀 GT 位姿,回傳真實 YCB 模型的擺放項(translation = pos - R@center,rotation = axis-angle)。"""
    g = scene.split("_")[0]
    mani = HC.CAPTURES / f"multi_{g}" / scene / "scene_manifest.json"
    if not mani.is_file():
        return []
    objs = json.loads(mani.read_text())["actual"]["viewpoints"][0]["objects"]
    out = []
    for o in objs:
        name = o["name"]; mp = ycb_mesh(name)
        if mp is None:
            continue
        p = o["position_m"]; pos = np.array([p[0], p[1], p[2]] if isinstance(p, list) else [p["x"], p["y"], p["z"]], float)
        aa = o.get("rotation_axis_angle", [0, 1, 0, 0])
        R = axis_angle_to_mat(aa[:3], aa[3])
        t = pos - R @ ycb_center(name)
        out.append({"name": f"ycb_{name}", "mesh": str(mp.resolve()),
                    "translation": [round(float(x), 5) for x in t],
                    "rotation": [float(aa[0]), float(aa[1]), float(aa[2]), float(aa[3])],
                    "color": [0.25, 0.5, 0.3], "transparency": 0.3})
    return out

PALETTE = [(0.9,0.2,0.2),(0.2,0.6,0.95),(0.2,0.8,0.3),(0.95,0.75,0.1),(0.7,0.3,0.9),
           (0.95,0.5,0.1),(0.1,0.8,0.8),(0.9,0.4,0.6),(0.5,0.7,0.2),(0.4,0.4,0.9)]


def load_occ_idx(root, scene):
    p = HC.EVAL_ROOT/root/scene/"occ.npz"
    if not p.is_file(): return None
    d = np.load(p); counts=d["counts"]; idx=d["idx"]; out=[]; off=0
    for c in counts: out.append(idx[off:off+int(c)]); off+=int(c)
    return out


def carve(view_masks, proj, P, allow_miss=None):
    votes=np.zeros(len(P),np.int16); nv=0
    for vn,seg in view_masks.items():
        if vn not in proj: continue
        ui,wi,inb=proj[vn]; h=np.zeros(len(P),bool); h[inb]=seg[wi[inb],ui[inb]]; votes+=h; nv+=1
    if nv<2: return None
    thr = HC.vote_threshold(nv) if allow_miss is None else (nv - allow_miss)
    occ=votes>=thr; return occ if occ.any() else None


def inst_masks(inst, scene):
    out={}
    for vn,files in inst["masks"].items():
        seg=None
        for f in files:
            m=cv2.imread(str(HC.SAM_ROOT/scene/vn/"masks"/f),cv2.IMREAD_GRAYSCALE)
            if m is None: continue
            b=m>127; seg=b if seg is None else (seg|b)
        if seg is not None: out[vn]=seg
    return out


def occ_to_obj(occ, shape, path):
    vol=np.pad(occ.reshape(shape).astype(np.float32),1)
    verts,faces=mcubes.marching_cubes(vol,0.5)
    if len(verts)==0: return False
    verts-=1.0
    sx=(HC.BOX_X[1]-HC.BOX_X[0])/(HC.RES-1); sy=(HC.BOX_Y[1]-HC.BOX_Y[0])/(HC.RES-1); sz=(HC.BOX_Z[1]-HC.BOX_Z[0])/(HC.RES-1)
    w=np.empty_like(verts)
    w[:,0]=HC.BOX_X[0]+verts[:,0]*sx; w[:,1]=HC.BOX_Y[0]+verts[:,1]*sy; w[:,2]=HC.BOX_Z[0]+verts[:,2]*sz
    lines=[f"v {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}" for p in w]+[f"f {a+1} {b+1} {c+1}" for a,b,c in faces]
    Path(path).write_text("\n".join(lines)+"\n",encoding="utf-8"); return True


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--root", default="v3/instance_hull_voxel")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-gt", action="store_true")
    args=ap.parse_args()
    scene=args.scene; out=Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.obj"): old.unlink()
    views=HC.load_views(scene); P,shape=HC.build_grid(); proj=HC.project_all(P,views); M=len(P)

    ij=HC.EVAL_ROOT/args.root/scene/"instances.json"
    if not ij.is_file(): sys.exit(f"找不到 {ij}")
    insts=json.loads(ij.read_text()).get("instances",[])
    occ_idx_list=load_occ_idx(args.root,scene)
    src="occ.npz" if occ_idx_list is not None else "recarve"
    manifest=[]; nh=0
    for i,inst in enumerate(insts):
        if occ_idx_list is not None and i<len(occ_idx_list):
            occ=np.zeros(M,bool); occ[occ_idx_list[i]]=True
        else:
            occ=carve(inst_masks(inst,scene),proj,P)
        if occ is None or not occ.any(): continue
        fn=f"inst_{i:02d}.obj"
        if occ_to_obj(occ,shape,out/fn):
            c=PALETTE[nh%len(PALETTE)]
            manifest.append({"file":fn,"name":f"inst_{i:02d}","color":list(c),"transparency":0.0}); nh+=1
    ng=0
    if not args.no_gt:
        gt=load_gt_amodal(scene)              # amodal(完整)GT,嚴格雕(allow_miss=0)
        if gt is None:
            print(f"[警告] {scene} 無 amodal GT(先跑 generate_amodal_masks.py),GT 殼跳過")
            gt={}
        for nm,vm in gt.items():
            occ=carve(vm,proj,P,allow_miss=0)
            if occ is None: continue
            fn=f"gt_{nm}.obj"
            if occ_to_obj(occ,shape,out/fn):
                manifest.append({"file":fn,"name":f"gt_{nm}","color":[0.6,0.6,0.6],"transparency":0.6}); ng+=1
    ycb = [] if args.no_gt else ycb_items(scene)
    (out/"manifest.json").write_text(json.dumps(
        {"scene":scene,"root":args.root,"source":src,"hull":nh,"gt":ng,"ycb":len(ycb),
         "items":manifest,"ycb_items":ycb},
        indent=2,ensure_ascii=False),encoding="utf-8")
    print(f"hull {nh} + GT殼 {ng} + 真實YCB {len(ycb)}(來源 {src})→ {out}")


if __name__=="__main__":
    main()
