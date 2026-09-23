"""同一前景hull:GT遮罩footprint投票=完美標籤;比方法標籤 precision/recall。遮擋=hull occupancy。排GEX(不當target)。"""
import numpy as np, json, sys, glob
from pathlib import Path
sys.path.insert(0,"srp/stage2_instances"); sys.path.insert(0,"srp/io")
import camera as cam, viewpoints as VP
import cg_associate as CG
from labels import label_dir
from pycocotools import mask as RLE
E=Path("data/eval"); CAP=Path("data/captures_fast"); GEXS={"bowl","skillet_lid","windex_bottle","colored_wood_blocks","dice"}
HULLS={"srp_hull_mv2_v12_am1":["瘦 fp drop (#5)|srp_hull_divB_t50_reNNfpSd_am1",
                               "瘦 fp 併最近|srp_hull_divB_t50_reNNfpS_am1",
                               "瘦 fp 侵蝕|srp_hull_divB_t50_reNNfpSe_am1",
                               "瘦 中心 併最近|srp_hull_divB_t50_reNNcS_am1",
                               "瘦 中心 drop (#6)|srp_hull_divB_t50_reNNcSd_am1",
                               "瘦 中心 侵蝕|srp_hull_divB_t50_reNNcSe_am1"],
       "srp_hull_mv2_v12_am1_fp":["胖 fp 併最近|srp_hull_divB_t50_reNNfpS_am1fp",
                                  "胖 fp drop|srp_hull_divB_t50_reNNfpSd_am1fp",
                                  "胖 fp 侵蝕|srp_hull_divB_t50_reNNfpSe_am1fp",
                                  "胖 中心 併最近|srp_hull_divB_t50_reNNcS_am1fp",
                                  "胖 中心 drop|srp_hull_divB_t50_reNNcSd_am1fp",
                                  "胖 中心 侵蝕|srp_hull_divB_t50_reNNcSe_am1fp"]}
def gt_masks(sc):
    d=label_dir(sc); ann=json.load(open(d/"actual"/"annotations.json"))
    id2n={c["id"]:c["name"].split("_",1)[-1] for c in ann["categories"] if c["name"]!="ur5e"}
    fn2img={Path(im["file_name"]).stem:im["id"] for im in ann["images"]}
    seg={}
    for a in ann["annotations"]:
        if a["category_id"] in id2n: seg.setdefault(fn2img_inv(fn2img,a["image_id"]),{})[id2n[a["category_id"]]]=a["segmentation"]
    return id2n, fn2img, {(a["image_id"],a["category_id"]):a["segmentation"] for a in ann["annotations"]}
def fn2img_inv(*a): return None
def build_gt(sc,hull):
    """回 (vox, surf_label_gt, vis, onames) 用GT遮罩footprint投票 + hull實心遮擋"""
    z=np.load(E/hull/sc/"hull.npz"); occ=z["occupancy"].astype(bool); surf=z["surface"].astype(bool)
    gm=z["grid_min"]; vs=float(z["voxel_size"]); shape=surf.shape
    vox=np.argwhere(surf); M=len(vox)
    ovox=np.argwhere(occ); oP=gm+(ovox+0.5)*vs
    sg=np.full(shape,-1,np.int64); sg[tuple(vox.T)]=np.arange(M); s_of_o=sg[tuple(ovox.T)]
    d=label_dir(sc); ann=json.load(open(d/"actual"/"annotations.json"))
    id2n={c["id"]:c["name"].split("_",1)[-1] for c in ann["categories"] if c["name"]!="ur5e"}
    onames=sorted(set(id2n.values())); oidx={o:i for i,o in enumerate(onames)}
    fn2img={Path(im["file_name"]).stem:im["id"] for im in ann["images"]}
    seg={(a["image_id"],a["category_id"]):a["segmentation"] for a in ann["annotations"]}
    g=sc.split("_")[0]; votes=np.zeros((M,len(onames))); vis=np.zeros(M,bool)
    for vn in sorted(VP.selected_view_names(12)):
        img=fn2img.get(vn); pf=CAP/f"multi_{g}"/sc/f"{vn}_pose.json"
        if img is None or not pf.is_file(): continue
        C,Rb=cam.load_pose(pf); va=CG.zbuffer_visible(oP,C,Rb,1280,720,vs)
        out=np.full(len(va),-1,np.int64); mm=va>=0; out[mm]=s_of_o[va[mm]]; vox_at=out.reshape(720,1280)
        ys,xs=np.where(vox_at>=0); own=vox_at[ys,xs]; vis[own]=True
        # 每 GT 物體遮罩
        for cid,oname in id2n.items():
            if (img,cid) not in seg: continue
            m=RLE.decode(seg[(img,cid)]).astype(bool)
            hit=m[ys,xs]; 
            if hit.any(): np.add.at(votes[:,oidx[oname]],own[hit],1.0)
    lab=np.full(M,-1,int); has=votes.max(1)>0; lab[has]=votes.argmax(1)[has]  # -1=無GT遮罩(過估區)
    return vox, lab, vis, onames
gcache={}
scenes=sorted(Path(p).parent.name for p in glob.glob(str(E/"srp_hull_divB_t50_reNNfpSd_am1fp/*_scene*/instances.npz")))
res={}
for hull,methods in HULLS.items():
    for spec in methods: res[spec]={"p_cor":0,"p_wr":0,"p_over":0,"r_cor":0,"r_wr":0,"r_miss":0}
for si,sc in enumerate(scenes):
    for hull,methods in HULLS.items():
        key=(sc,hull)
        try:
            if key not in gcache: gcache[key]=build_gt(sc,hull)
        except Exception as e:
            continue
        vox,gtlab,vis,onames=gcache[key]
        gex_idx={i for i,o in enumerate(onames) if o in GEXS}
        gt_valid=np.array([ (l>=0 and l not in gex_idx) for l in gtlab])  # GT標到非GEX物體
        for spec in methods:
            name,root=spec.split("|")
            f=E/root/sc/"instances.npz"
            if not f.is_file(): continue
            lab=np.load(f)["labels"]; z=np.load(E/hull/sc/"hull.npz")
            vl=lab[tuple(vox.T)]
            labeled=vis&(vl>0)
            # 方法 instance → GT物體(用gtlab眾數)
            dom={}
            for k in np.unique(vl[labeled]):
                mm=labeled&(vl==k)&(gtlab>=0)
                if mm.sum()==0: dom[int(k)]=-1; continue
                u,c=np.unique(gtlab[mm],return_counts=True); dom[int(k)]=int(u[c.argmax()])
            a=res[spec]
            # precision:方法可見標到的 voxel
            for p in np.where(labeled)[0]:
                d=dom.get(int(vl[p]),-1); gl=gtlab[p]
                if gl<0 or gl in gex_idx: a["p_over"]+=1        # GT沒標(過估區/GEX)→ 方法標了=過估
                elif d==gl: a["p_cor"]+=1
                else: a["p_wr"]+=1
            # recall:GT標到非GEX的可見 voxel
            for p in np.where(vis&gt_valid)[0]:
                k=int(vl[p]); 
                if k==0: a["r_miss"]+=1
                elif dom.get(k,-1)==gtlab[p]: a["r_cor"]+=1
                else: a["r_wr"]+=1
    if (si+1)%50==0: print(f"..{si+1}/{len(scenes)}",flush=True)
print("\n基準=同前景hull+GT遮罩完美標籤;遮擋=hull occupancy;排GEX不當target")
print("\n=== precision(分母=方法可見標到的voxel)===")
print(f"{'方法':16s} {'對%':>6} {'標錯物%':>7} {'過估(GT沒標)%':>12}")
for hull,methods in HULLS.items():
    for spec in methods:
        name=spec.split("|")[0]; a=res[spec]; t=max(a["p_cor"]+a["p_wr"]+a["p_over"],1)
        print(f"{name:16s} {a['p_cor']/t*100:6.1f} {a['p_wr']/t*100:7.1f} {a['p_over']/t*100:12.1f}")
print("\n=== recall(分母=GT完美標到的非GEX可見voxel)===")
print(f"{'方法':16s} {'對%':>6} {'標錯物%':>7} {'漏%':>6}")
for hull,methods in HULLS.items():
    for spec in methods:
        name=spec.split("|")[0]; a=res[spec]; t=max(a["r_cor"]+a["r_wr"]+a["r_miss"],1)
        print(f"{name:16s} {a['r_cor']/t*100:6.1f} {a['r_wr']/t*100:7.1f} {a['r_miss']/t*100:6.1f}")
