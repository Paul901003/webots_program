import numpy as np, glob, sys
from pathlib import Path
from scipy.optimize import linear_sum_assignment
sys.path.insert(0,"srp/stage2_instances"); sys.path.insert(0,"srp/io")
import eval_mesh as EM, camera as cam, viewpoints as VP
import cg_associate as CG
E=Path("data/eval"); CAP=Path("data/captures_fast")
GEX={"024_bowl","028_skillet_lid","022_windex_bottle","070-a_colored_wood_blocks","062_dice"}
ROOT="srp_hull_divB_t50_reNN_guard"; HULL="srp_hull_mv2_v12_am1"
def solid_vis(sc,occ,gm,vs,shape,vox):
    ovox=np.argwhere(occ);oP=gm+(ovox+0.5)*vs
    sg=np.full(shape,-1,np.int64);sg[tuple(vox.T)]=np.arange(len(vox));s_of_o=sg[tuple(ovox.T)]
    g=sc.split("_")[0];vis=np.zeros(len(vox),bool)
    for vn in sorted(VP.selected_view_names(12)):
        pf=CAP/f"multi_{g}"/sc/f"{vn}_pose.json"
        if not pf.is_file():continue
        C,Rb=cam.load_pose(pf);va=CG.zbuffer_visible(oP,C,Rb,1280,720,vs)
        si=s_of_o[va[va>=0]];vis[si[si>=0]]=True
    return vis
# 303 多物場(和其他線同一組)
scenes=sorted(Path(p).parent.name for p in glob.glob(str(E/"srp_hull_divB_t50_reNNfpSd_am1fp/*_scene*/instances.npz")))
a={"h5":0,"h7":0,"tot":0,"iou":0.0}; miss=0
for sc in scenes:
    f=E/ROOT/sc/"instances.npz"
    if not f.is_file(): miss+=1; continue
    z=np.load(E/HULL/sc/"hull.npz");occ=z["occupancy"].astype(bool);surf=z["surface"].astype(bool)
    gm=z["grid_min"];vs=float(z["voxel_size"]);shape=surf.shape;vox=np.argwhere(surf)
    vis=solid_vis(sc,occ,gm,vs,shape,vox)
    gt=EM.solid_mesh_occ(sc,gm,vs,shape);onames=[o for o in gt if o not in GEX]
    mlab=np.full(len(vox),-1,int)
    for oi,o in enumerate(onames):mlab[gt[o].astype(bool)[tuple(vox.T)]]=oi
    lab=np.load(f)["labels"];vlab=lab[tuple(vox.T)];sel=vis&(mlab>=0)
    gtm=[(mlab==oi)&sel for oi in range(len(onames))];ks=[k for k in np.unique(vlab) if k>0]
    predm=[(vlab==k)&sel for k in ks];Mx=np.zeros((len(gtm),len(predm)))
    for i,gg in enumerate(gtm):
        if gg.sum()==0:continue
        for j,pp in enumerate(predm):
            it=int((gg&pp).sum())
            if it:Mx[i,j]=it/int((gg|pp).sum())
    ri,cj=linear_sum_assignment(-Mx);best={i:Mx[i,j] for i,j in zip(ri,cj)}
    for i in range(len(onames)):
        if gtm[i].sum()==0:continue
        io=best.get(i,0);a["tot"]+=1;a["iou"]+=io
        if io>=0.5:a["h5"]+=1
        if io>=0.7:a["h7"]+=1
t=max(a["tot"],1)
print(f"#1 reNN_guard(瘦am1+中心+薄殼bug+併最近)物理正確 eval:缺場={miss}")
print(f"  found@0.5={a['h5']/t:.3f}  found@0.7={a['h7']/t:.3f}  mIoU={a['iou']/t:.3f}  obj={a['tot']}")
