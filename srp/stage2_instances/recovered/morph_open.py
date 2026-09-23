"""測 opening(erode1→dilate1) 6/18/26連通對 hull occupancy 的影響:真實(mesh內)/鬼影(mesh外)voxel變化。基準=真實mesh(含所有物體)。"""
import numpy as np, sys, glob
from pathlib import Path
from scipy import ndimage
sys.path.insert(0,"srp/stage2_instances"); sys.path.insert(0,"srp/io")
import eval_mesh as EM
E=Path("data/eval")
SE={6:ndimage.generate_binary_structure(3,1),18:ndimage.generate_binary_structure(3,2),26:ndimage.generate_binary_structure(3,3)}
for HULL in ["srp_hull_mv2_v12_am1","srp_hull_mv2_v12_am1_fp"]:
    scenes=sorted(Path(p).parent.name for p in glob.glob(str(E/HULL/"*_scene*/hull.npz")))
    scenes=[s for s in scenes if not s.startswith("n1_")]
    agg={"base_real":0,"base_ghost":0}
    for c in SE: agg[f"{c}_real"]=0; agg[f"{c}_ghost"]=0
    n=0
    for sc in scenes:
        z=np.load(E/HULL/sc/"hull.npz"); occ=z["occupancy"].astype(bool); gm=z["grid_min"]; vs=float(z["voxel_size"]); shape=occ.shape
        gt=EM.solid_mesh_occ(sc,gm,vs,shape)
        mesh=np.zeros(shape,bool)
        for o in gt: mesh|=gt[o].astype(bool)   # 真實=任一物體mesh內(含GEX,都是真物體)
        agg["base_real"]+=int((occ&mesh).sum()); agg["base_ghost"]+=int((occ&~mesh).sum())
        for c,se in SE.items():
            op=ndimage.binary_opening(occ,se)
            agg[f"{c}_real"]+=int((op&mesh).sum()); agg[f"{c}_ghost"]+=int((op&~mesh).sum())
        n+=1
    print(f"\n===== {HULL} ({n}場) =====")
    br,bg=agg["base_real"],agg["base_ghost"]
    print(f"{'操作':10s} {'真實voxel':>12} {'鬼影voxel':>12} {'真實變化%':>9} {'鬼影變化%':>9} {'鬼影佔比%':>9}")
    print(f"{'原始':10s} {br:>12,} {bg:>12,} {'—':>9} {'—':>9} {bg/(br+bg)*100:>9.1f}")
    for c in [6,18,26]:
        r,g=agg[f"{c}_real"],agg[f"{c}_ghost"]
        print(f"{'open'+str(c):10s} {r:>12,} {g:>12,} {(r-br)/br*100:>9.1f} {(g-bg)/bg*100:>9.1f} {g/(r+g)*100:>9.1f}")
