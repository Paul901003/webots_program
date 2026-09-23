"""兩指標:①全域voxel三分法(歸對/歸錯污染/沒歸) ②相觸on對的主instance precision(原始,不設門檻)。
各線用自己hull的可見表面;排GEX。"""
import numpy as np, json, sys, glob
from pathlib import Path
sys.path.insert(0,"srp/stage2_instances"); sys.path.insert(0,"srp/io")
import eval_mesh as EM, camera as cam, viewpoints as VP
import cg_associate as CG
from labels import label_dir
E=Path("data/eval"); CAP=Path("data/captures_fast")
GEX={"024_bowl","028_skillet_lid","022_windex_bottle","070-a_colored_wood_blocks","070-b_colored_wood_blocks","062_dice"}
GEXS={g.split("_",1)[-1] for g in GEX}|{"colored_wood_blocks"}
# (顯示名, div root, hull)
M=[("瘦 fp   drop (#5)","srp_hull_divB_t50_reNNfpSd_am1","srp_hull_mv2_v12_am1"),
   ("瘦 fp   侵蝕","srp_hull_divB_t50_reNNfpSe_am1","srp_hull_mv2_v12_am1"),
   ("瘦 中心 drop (#6)","srp_hull_divB_t50_reNNcSd_am1","srp_hull_mv2_v12_am1"),
   ("胖 fp   併最近","srp_hull_divB_t50_reNNfpS_am1fp","srp_hull_mv2_v12_am1_fp"),
   ("胖 fp   drop","srp_hull_divB_t50_reNNfpSd_am1fp","srp_hull_mv2_v12_am1_fp"),
   ("胖 fp   侵蝕","srp_hull_divB_t50_reNNfpSe_am1fp","srp_hull_mv2_v12_am1_fp"),
   ("胖 中心 併最近","srp_hull_divB_t50_reNNcS_am1fp","srp_hull_mv2_v12_am1_fp"),
   ("胖 中心 drop","srp_hull_divB_t50_reNNcSd_am1fp","srp_hull_mv2_v12_am1_fp"),
   ("胖 中心 侵蝕","srp_hull_divB_t50_reNNcSe_am1fp","srp_hull_mv2_v12_am1_fp")]
def vis_surf(sc,occ,gm,vs,shape,vox):
    ovox=np.argwhere(occ);oP=gm+(ovox+0.5)*vs
    sg=np.full(shape,-1,np.int64);sg[tuple(vox.T)]=np.arange(len(vox));s=sg[tuple(ovox.T)]
    g=sc.split("_")[0];vis=np.zeros(len(vox),bool)
    for vn in sorted(VP.selected_view_names(12)):
        pf=CAP/f"multi_{g}"/sc/f"{vn}_pose.json"
        if not pf.is_file():continue
        C,Rb=cam.load_pose(pf);va=CG.zbuffer_visible(oP,C,Rb,1280,720,vs)
        si=s[va[va>=0]];vis[si[si>=0]]=True
    return vis
def on_pairs(sc):
    try: rel=json.loads((label_dir(sc)/"relations.json").read_text())
    except: return []
    out=[]
    for r in rel.get("relations",[]):
        if r.get("type")=="on":
            x=r.get("x","").split("_",1)[-1]; y=r.get("y","").split("_",1)[-1]
            if x and y: out.append((x,y))
    return out
scenes=sorted(Path(p).parent.name for p in glob.glob(str(E/"srp_hull_divB_t50_reNNfpSd_am1fp/*_scene*/instances.npz")))
G={k:{"cor":0,"wr":0,"un":0} for k,_,_ in M}       # 全域三分
S={k:{"up_p":[],"lo_p":[]} for k,_,_ in M}          # 相觸precision
cache={}
for sc in scenes:
    pairs=on_pairs(sc)
    for name,root,hull in M:
        f=E/root/sc/"instances.npz"
        if not f.is_file(): continue
        key=(sc,hull)
        if key not in cache:
            z=np.load(E/hull/sc/"hull.npz");occ=z["occupancy"].astype(bool);surf=z["surface"].astype(bool)
            gm=z["grid_min"];vs=float(z["voxel_size"]);shape=surf.shape;vox=np.argwhere(surf)
            vis=vis_surf(sc,occ,gm,vs,shape,vox)
            gt=EM.solid_mesh_occ(sc,gm,vs,shape)
            names=[o.split("_",1)[-1] for o in gt]
            objb={o.split("_",1)[-1]:gt[o].astype(bool)[tuple(vox.T)] for o in gt}
            # mlab: non-GEX GT index
            keep=[o for o in objb if o not in GEXS]
            mlab=np.full(len(vox),-1,int)
            for oi,o in enumerate(keep): mlab[objb[o]]=oi
            anyobj=np.zeros(len(vox),bool)
            for o in keep: anyobj|=objb[o]
            cache[key]=(vox,vis,mlab,keep,objb,anyobj)
        vox,vis,mlab,keep,objb,anyobj=cache[key]
        lab=np.load(f)["labels"]; vl=lab[tuple(vox.T)]
        selvis=vis
        # 每 instance 主導 GT(keep index)
        dominant={}
        for k in np.unique(vl[vl>0]):
            m=selvis&(vl==k)&(mlab>=0)
            if m.sum()==0: dominant[int(k)]=-1; continue
            u,c=np.unique(mlab[m],return_counts=True); dominant[int(k)]=int(u[c.argmax()])
        # 全域三分:可見 in-GT voxel
        ingt=selvis&(mlab>=0)
        for p in np.where(ingt)[0]:
            k=int(vl[p])
            if k==0: G[name]["un"]+=1
            elif dominant.get(k,-1)==mlab[p]: G[name]["cor"]+=1
            else: G[name]["wr"]+=1
        # 相觸 precision(用可見表面)
        for (u,l) in pairs:
            if u in GEXS or l in GEXS or u not in objb or l not in objb: continue
            for who,o in [("up_p",u),("lo_p",l)]:
                m=selvis&objb[o]
                ls=vl[m]; ls=ls[ls>0]
                if len(ls)==0: continue
                vals,cnts=np.unique(ls,return_counts=True); ck=int(vals[cnts.argmax()]); tp=int(cnts.max())
                inc=selvis&(vl==ck); den=int((inc&anyobj).sum())
                if den>0: S[name][who].append(tp/den)
print(f"({len(scenes)}場) 排GEX\n")
print("① 全域 voxel 三分法(可見in-GT voxel為分母)")
print(f"{'方法':18s} {'歸對%':>7} {'歸錯%(污染)':>11} {'沒歸%(留白)':>11}")
for name,_,_ in M:
    a=G[name]; t=max(a['cor']+a['wr']+a['un'],1)
    print(f"{name:18s} {a['cor']/t*100:7.1f} {a['wr']/t*100:11.1f} {a['un']/t*100:11.1f}")
print("\n② 相觸on對 主instance precision 原始平均(越低=融合污染越重;不設門檻)")
print(f"{'方法':18s} {'上物prec':>8} {'下物prec':>8} {'對數':>5}")
for name,_,_ in M:
    up=S[name]['up_p']; lo=S[name]['lo_p']
    print(f"{name:18s} {np.mean(up) if up else 0:8.3f} {np.mean(lo) if lo else 0:8.3f} {len(up):5d}")
