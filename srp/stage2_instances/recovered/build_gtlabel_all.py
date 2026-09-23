import numpy as np, json, sys, glob
from pathlib import Path
sys.path.insert(0,"srp/stage2_instances"); sys.path.insert(0,"srp/io")
import camera as cam, viewpoints as VP
import cg_associate as CG
from labels import label_dir
from pycocotools import mask as RLE
E=Path("data/eval"); CAP=Path("data/captures_fast")
def build(sc,hull,rootout):
    out=E/rootout/sc
    if (out/"instances.npz").is_file(): return "skip"
    hp=E/hull/sc/"hull.npz"
    if not hp.is_file(): return "no_hull"
    z=np.load(hp); occ=z["occupancy"].astype(bool); surf=z["surface"].astype(bool)
    gm=z["grid_min"]; vs=float(z["voxel_size"]); shape=surf.shape
    vox=np.argwhere(surf); M=len(vox); ovox=np.argwhere(occ); oP=gm+(ovox+0.5)*vs
    sg=np.full(shape,-1,np.int64); sg[tuple(vox.T)]=np.arange(M); s_of_o=sg[tuple(ovox.T)]
    d=label_dir(sc); af=d/"actual"/"annotations.json"
    if not af.is_file(): return "no_ann"
    ann=json.load(open(af))
    id2n={c["id"]:c["name"].split("_",1)[-1] for c in ann["categories"] if c["name"]!="ur5e"}
    onames=sorted(set(id2n.values())); oidx={o:i for i,o in enumerate(onames)}
    fn2img={Path(im["file_name"]).stem:im["id"] for im in ann["images"]}
    seg={(a["image_id"],a["category_id"]):a["segmentation"] for a in ann["annotations"]}
    g=sc.split("_")[0]; votes=np.zeros((M,len(onames)))
    for vn in sorted(VP.selected_view_names(12)):
        img=fn2img.get(vn); pf=CAP/f"multi_{g}"/sc/f"{vn}_pose.json"
        if img is None or not pf.is_file(): continue
        C,Rb=cam.load_pose(pf); va=CG.zbuffer_visible(oP,C,Rb,1280,720,vs)
        o2=np.full(len(va),-1,np.int64); mm=va>=0; o2[mm]=s_of_o[va[mm]]; vox_at=o2.reshape(720,1280)
        ys,xs=np.where(vox_at>=0); own=vox_at[ys,xs]
        for cid,oname in id2n.items():
            if (img,cid) not in seg: continue
            m=RLE.decode(seg[(img,cid)]).astype(bool); hit=m[ys,xs]
            if hit.any(): np.add.at(votes[:,oidx[oname]],own[hit],1.0)
    lab=np.zeros(M,int); has=votes.max(1)>0; lab[has]=votes.argmax(1)[has]+1
    grid=np.zeros(shape,np.int32); grid[tuple(vox.T)]=lab
    out.mkdir(parents=True,exist_ok=True)
    meta={"script":"GT完美標籤(前景hull+GT遮罩投票)","hull":hull,"labelmap":{i+1:o for i,o in enumerate(onames)}}
    np.savez_compressed(out/"instances.npz",labels=grid,grid_min=gm,voxel_size=vs,occupancy=occ,build_meta=json.dumps(meta,ensure_ascii=False))
    return "ok"
scenes=sorted(Path(p).parent.name for p in glob.glob(str(E/"srp_hull_mv2_v12_am1_fp/*_scene*/hull.npz")))
scenes=[s for s in scenes if not s.startswith("n1_")]
print(f"場景={len(scenes)}")
for hull,ro in [("srp_hull_mv2_v12_am1","srp_hull_gtlabel_am1"),("srp_hull_mv2_v12_am1_fp","srp_hull_gtlabel_am1fp")]:
    ok=skip=0
    for i,sc in enumerate(scenes):
        r=build(sc,hull,ro); ok+=(r=="ok"); skip+=(r=="skip")
        if (i+1)%50==0: print(f"  {ro} ..{i+1}/{len(scenes)} (ok={ok} skip={skip})",flush=True)
    print(f"{ro}: 完成 ok={ok} skip={skip}",flush=True)
print("DONE")
