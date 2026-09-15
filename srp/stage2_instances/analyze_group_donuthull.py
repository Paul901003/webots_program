#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""analyze_group_donuthull.py — 每個語意群用其 donut 遮罩雕 hull,測「真物體雕得出、吻合 base;雜訊群雕不出/差很多」。
每群:各視角 donut 遮罩聯集當前景 → 在 base bbox 內 soft-carve(allow_miss=1)→ 量體積、落在 base 內比例、連通塊。
分 真物體群(gobj≠None,對到 GT 物體) vs 雜訊群(gobj=None)。堆疊場。
用法: SAM_ROOT=.. CAPTURES_ROOT=.. ./analyze_group_donuthull.py
"""
import os, sys, json
import numpy as np, cv2
from collections import Counter, defaultdict
from pathlib import Path
from scipy import ndimage
REPO=Path(__file__).resolve().parents[2]; EVAL=REPO/"data"/"eval"
sys.path.insert(0,str(REPO/"srp/io")); sys.path.insert(0,str(REPO/"srp/stage2_instances"))
import camera as cam, masks as MK, viewpoints as VP, labels as L
from voxel_sem_cluster_donut import donut_masks
MV2=Path("data/eval/mobilesamv2_fast"); CAP=Path("data/captures_fast"); ARM=Path("data/eval/srp_arm_masks")
INST="srp_hull_semcluster_surf_am1photo"; HULL="srp_hull_mv2_v12_am1_photo"
scenes=[p.name for g in ("stack3","stack4","stack5") for p in sorted((EVAL/INST).glob(f"{g}_scene*")) if (EVAL/HULL/p.name/"hull.npz").is_file()]
real=[]; spur=[]
for si,sc in enumerate(scenes):
    z=np.load(EVAL/HULL/sc/"hull.npz"); base=z["occupancy"]; gm=z["grid_min"]; vs=float(z["voxel_size"])
    bidx=np.argwhere(base)
    if len(bidx)==0: continue
    lo=np.clip(bidx.min(0)-12,0,None); hi=np.clip(bidx.max(0)+12,0,np.array(base.shape)-1)
    gx,gy,gz=[np.arange(lo[d],hi[d]+1) for d in range(3)]
    GX,GY,GZ=np.meshgrid(gx,gy,gz,indexing="ij"); vidx=np.stack([GX.ravel(),GY.ravel(),GZ.ravel()],1)
    Pw=gm+(vidx+0.5)*vs; base_reg=base[vidx[:,0],vidx[:,1],vidx[:,2]]
    mc=json.loads((EVAL/INST/sc/"instances.json").read_text()).get("mask_clusters",{})
    # reproj_labels 給群→GT物體
    rl=np.load(EVAL/INST/sc/"reproj_labels.npz",allow_pickle=True)
    og=rl["own_group"].astype(int); gt=rl["gt_obj"].astype(int)
    gobj={}
    for g in np.unique(og):
        if g<=0: continue
        o=gt[(og==g)&(gt>=0)]; gobj[int(g)]=Counter(o.tolist()).most_common(1)[0][0] if len(o) else None
    sdir=CAP/f"multi_{sc.split('_')[0]}"/sc
    # 每群各視角 silhouette:votes
    grp_votes=defaultdict(lambda: np.zeros(len(Pw),int)); grp_nv=defaultdict(int)
    for vn in sorted(VP.selected_view_names(12)):
        vd=MV2/sc/vn; pf=sdir/f"{vn}_pose.json"
        if not (vd.is_dir() and pf.is_file()): continue
        km=MK.kept_object_masks(vd); names=[n for _,n in km]; ms0=[m for m,_ in km]
        ap=ARM/sc/f"{vn}_arm.png"; arm=(cv2.imread(str(ap),0)>127) if ap.is_file() else None
        if arm is not None:
            keep=[k for k,m in enumerate(ms0) if (m&arm).sum()/max(int(m.sum()),1)<0.5]
            names=[names[k] for k in keep]; ms0=[ms0[k] for k in keep]
        if not ms0: continue
        don=donut_masks(ms0); H,W=ms0[0].shape
        C,Rb=cam.load_pose(pf); Rwc,t=cam.pose_to_w2c(C,Rb); K=cam.intrinsics(W,H)
        X=Pw@Rwc.T+t; zc=X[:,2]; ok=zc>1e-6; zz=np.where(ok,zc,1.0)
        px=np.round(K[0,0]*X[:,0]/zz+K[0,2]).astype(int); py=np.round(K[1,1]*X[:,1]/zz+K[1,2]).astype(int)
        inb=ok&(px>=0)&(px<W)&(py>=0)&(py<H)
        cl=mc.get(vn,{})
        gmask=defaultdict(lambda: np.zeros((H,W),bool))
        for k,nm in enumerate(names):
            gid=cl.get(nm)
            if gid is not None: gmask[gid]|=ms0[k]
        for gid,sil in gmask.items():
            v=np.zeros(len(Pw),bool); ii=np.where(inb)[0]
            v[ii]=sil[py[ii],px[ii]]
            grp_votes[gid]+=v; grp_nv[gid]+=1
    for gid,votes in grp_votes.items():
        nv=grp_nv[gid]
        if nv<3: continue
        ghull=votes>=nv-1                       # soft allow_miss=1
        vol=int(ghull.sum())
        if vol<5:
            rec=(sc,gid,vol,0.0,0)
        else:
            inside=int((ghull&base_reg).sum())/vol
            ncc=ndimage.label(ghull.reshape(GX.shape))[1]
            rec=(sc,gid,vol,inside,ncc)
        (real if gobj.get(gid) is not None else spur).append(rec)
    if (si+1)%15==0: print(f"  {si+1}/{len(scenes)}",flush=True)
def rep(nm,R):
    if not R: print(f"{nm}: n=0"); return
    vol=np.array([r[2] for r in R]); ins=np.array([r[3] for r in R]); ncc=np.array([r[4] for r in R])
    print(f"{nm}: n={len(R)} | 體積中位 {np.median(vol):.0f} | 落base內比例 中位 {np.median(ins):.2f} | 連通塊中位 {np.median(ncc):.0f} | 雕近空(<5vox) {(vol<5).mean()*100:.0f}%")
print(f"\n60 堆疊場 per-群 donut-hull:")
rep("真物體群(gobj≠None)",real)
rep("雜訊群(gobj=None)",spur)
