"""基準=modal GT-mask-hull(per-object,allow_miss=1)。算 precision(方法voxel為分母)+ recall(GThull voxel為分母)。排GEX。"""
import numpy as np, json, sys, glob
from pathlib import Path
sys.path.insert(0,"srp/stage2_instances"); sys.path.insert(0,"srp/io")
import camera as cam, viewpoints as VP
import cg_associate as CG
from labels import label_dir
from pycocotools import mask as RLE
E=Path("data/eval"); CAP=Path("data/captures_fast"); GEXS={"bowl","skillet_lid","windex_bottle","colored_wood_blocks","dice"}; AM=1
M=[("瘦 fp   drop (#5)","srp_hull_divB_t50_reNNfpSd_am1","srp_hull_mv2_v12_am1"),
   ("瘦 fp   併最近","srp_hull_divB_t50_reNNfpS_am1","srp_hull_mv2_v12_am1"),
   ("瘦 fp   侵蝕","srp_hull_divB_t50_reNNfpSe_am1","srp_hull_mv2_v12_am1"),
   ("瘦 中心 併最近","srp_hull_divB_t50_reNNcS_am1","srp_hull_mv2_v12_am1"),
   ("瘦 中心 drop (#6)","srp_hull_divB_t50_reNNcSd_am1","srp_hull_mv2_v12_am1"),
   ("瘦 中心 侵蝕","srp_hull_divB_t50_reNNcSe_am1","srp_hull_mv2_v12_am1"),
   ("胖 fp   併最近","srp_hull_divB_t50_reNNfpS_am1fp","srp_hull_mv2_v12_am1_fp"),
   ("胖 fp   drop","srp_hull_divB_t50_reNNfpSd_am1fp","srp_hull_mv2_v12_am1_fp"),
   ("胖 fp   侵蝕","srp_hull_divB_t50_reNNfpSe_am1fp","srp_hull_mv2_v12_am1_fp"),
   ("胖 中心 併最近","srp_hull_divB_t50_reNNcS_am1fp","srp_hull_mv2_v12_am1_fp"),
   ("胖 中心 drop","srp_hull_divB_t50_reNNcSd_am1fp","srp_hull_mv2_v12_am1_fp"),
   ("胖 中心 侵蝕","srp_hull_divB_t50_reNNcSe_am1fp","srp_hull_mv2_v12_am1_fp")]
def surface(o):
    s=np.zeros_like(o); s[1:-1,1:-1,1:-1]=o[1:-1,1:-1,1:-1]&~(o[:-2,1:-1,1:-1]&o[2:,1:-1,1:-1]&o[1:-1,:-2,1:-1]&o[1:-1,2:,1:-1]&o[1:-1,1:-1,:-2]&o[1:-1,1:-1,2:]); return s
def gt_hulls(sc,gm,vs,shape):
    d=label_dir(sc); ann=json.load(open(d/"actual"/"annotations.json"))
    fn2img={Path(im["file_name"]).stem:im["id"] for im in ann["images"]}
    seg={(a["image_id"],a["category_id"]):a["segmentation"] for a in ann["annotations"]}
    views=sorted(VP.selected_view_names(12)); g=sc.split("_")[0]
    ii,jj,kk=np.meshgrid(np.arange(shape[0]),np.arange(shape[1]),np.arange(shape[2]),indexing="ij")
    Pall=(gm+(np.stack([ii,jj,kk],-1)+0.5)*vs).reshape(-1,3); N=len(Pall); out={}
    for c in ann["categories"]:
        if c["name"]=="ur5e": continue
        cid=c["id"]; cnt=np.zeros(N,np.int32); nv=0
        for vn in views:
            img=fn2img.get(vn)
            if img is None or (img,cid) not in seg: continue
            pf=CAP/f"multi_{g}"/sc/f"{vn}_pose.json"
            if not pf.is_file(): continue
            m=RLE.decode(seg[(img,cid)]).astype(bool); H,W=m.shape
            C,Rb=cam.load_pose(pf); Rwc,t=cam.pose_to_w2c(C,Rb); K=cam.intrinsics(W,H)
            X=Pall@Rwc.T+t; zc=X[:,2]; ok=zc>1e-9; zz=np.where(ok,zc,1.0)
            u=np.round(K[0,0]*X[:,0]/zz+K[0,2]).astype(int); v=np.round(K[1,1]*X[:,1]/zz+K[1,2]).astype(int)
            inb=ok&(u>=0)&(u<W)&(v>=0)&(v<H); idx=np.where(inb)[0]; hit=m[v[idx],u[idx]]; cnt[idx[hit]]+=1; nv+=1
        if nv==0: continue
        out[c["name"].split("_",1)[-1]]=(cnt>=nv-AM).reshape(shape)
    return out
def visible(occ,gm,vs,shape,vox,sc):
    ovox=np.argwhere(occ);oP=gm+(ovox+0.5)*vs
    sg=np.full(shape,-1,np.int64);sg[tuple(vox.T)]=np.arange(len(vox));s=sg[tuple(ovox.T)]
    g=sc.split("_")[0];vis=np.zeros(len(vox),bool)
    for vn in sorted(VP.selected_view_names(12)):
        pf=CAP/f"multi_{g}"/sc/f"{vn}_pose.json"
        if not pf.is_file():continue
        C,Rb=cam.load_pose(pf);va=CG.zbuffer_visible(oP,C,Rb,1280,720,vs)
        si=s[va[va>=0]];vis[si[si>=0]]=True
    return vis
scenes=sorted(Path(p).parent.name for p in glob.glob(str(E/"srp_hull_divB_t50_reNNfpSd_am1fp/*_scene*/instances.npz")))
P={k:{"cor":0,"cross":0,"out":0} for k,_,_ in M}     # precision(方法voxel)
Rc={k:{"cor":0,"wrong":0,"miss":0} for k,_,_ in M}   # recall(GThull voxel)
gcache={}
for si,sc in enumerate(scenes):
    for name,root,hull in M:
        f=E/root/sc/"instances.npz"
        if not f.is_file(): continue
        z=np.load(E/hull/sc/"hull.npz");occ=z["occupancy"].astype(bool);surf=z["surface"].astype(bool)
        gm=z["grid_min"];vs=float(z["voxel_size"]);shape=surf.shape;vox=np.argwhere(surf)
        gk=(sc,tuple(gm),vs,shape)
        if gk not in gcache:
            gh=gt_hulls(sc,gm,vs,shape)
            onames=[o for o in gh if o not in GEXS]
            gt_occ=np.zeros(shape,bool)
            for o in onames: gt_occ|=gh[o]
            gsurf=surface(gt_occ); gvox=np.argwhere(gsurf)
            gvis=visible(gt_occ,gm,vs,shape,gvox,sc) if len(gvox) else np.zeros(0,bool)
            # GThull surface voxel → 屬哪個物體(排GEX)
            gobj=np.full(len(gvox),-1,int)
            for oi,o in enumerate(onames):
                gobj[gh[o][tuple(gvox.T)]]=oi
            gcache[gk]=(gh,onames,gvox,gvis,gobj)
        gh,onames,gvox,gvis,gobj=gcache[gk]
        if not onames: continue
        labgrid=np.load(f)["labels"]
        # 方法可見表面
        mvis=visible(occ,gm,vs,shape,vox,sc)
        vl=labgrid[tuple(vox.T)]
        inobj=np.zeros((len(vox),len(onames)),bool)
        for oi,o in enumerate(onames): inobj[:,oi]=gh[o][tuple(vox.T)]
        labeled=mvis&(vl>0)
        dom={}
        for k in np.unique(vl[labeled]):
            cnts=inobj[labeled&(vl==k)].sum(0); dom[int(k)]=int(cnts.argmax()) if cnts.max()>0 else -1
        # precision
        for p in np.where(labeled)[0]:
            d=dom.get(int(vl[p]),-1); row=inobj[p]
            if d>=0 and row[d]: P[name]["cor"]+=1
            elif row.any(): P[name]["cross"]+=1
            else: P[name]["out"]+=1
        # recall:GThull 可見表面 voxel → 方法在該格標什麼
        if len(gvox):
            gml=labgrid[tuple(gvox.T)]
            sel=gvis&(gobj>=0)
            for gi in np.where(sel)[0]:
                o=gobj[gi]; k=int(gml[gi])
                if k==0: Rc[name]["miss"]+=1
                elif dom.get(k,-1)==o: Rc[name]["cor"]+=1
                else: Rc[name]["wrong"]+=1
    if (si+1)%50==0: print(f"..{si+1}/{len(scenes)}",flush=True)
print("\n=== precision(分母=方法可見表面被標voxel)===")
print(f"{'方法':18s} {'正確%':>7} {'跨物%':>7} {'GThull外%':>9}")
for name,_,_ in M:
    a=P[name]; t=max(a['cor']+a['cross']+a['out'],1)
    print(f"{name:18s} {a['cor']/t*100:7.1f} {a['cross']/t*100:7.1f} {a['out']/t*100:9.1f}")
print("\n=== recall(分母=GThull可見表面voxel)===")
print(f"{'方法':18s} {'正確覆蓋%':>9} {'覆蓋成別物%':>11} {'漏(沒覆蓋)%':>11}")
for name,_,_ in M:
    a=Rc[name]; t=max(a['cor']+a['wrong']+a['miss'],1)
    print(f"{name:18s} {a['cor']/t*100:9.1f} {a['wrong']/t*100:11.1f} {a['miss']/t*100:11.1f}")
