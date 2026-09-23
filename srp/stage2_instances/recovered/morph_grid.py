"""erode E次→dilate D次(E,D∈1..3),26連通;對hull occupancy量真實(mesh內)/鬼影(mesh外)。同303多物場。"""
import numpy as np, sys, glob
from pathlib import Path
from scipy import ndimage
sys.path.insert(0,"srp/stage2_instances"); sys.path.insert(0,"srp/io")
import eval_mesh as EM
E=Path("data/eval"); se=ndimage.generate_binary_structure(3,3)  # 26連通
scenes=sorted(Path(p).parent.name for p in glob.glob(str(E/"srp_hull_divB_t50_reNNfpSd_am1fp/*_scene*/instances.npz")))
combos=[(e,d) for e in [1,2,3] for d in [1,2,3]]
for HULL in ["srp_hull_mv2_v12_am1","srp_hull_mv2_v12_am1_fp"]:
    agg={"br":0,"bg":0}; 
    for e,d in combos: agg[(e,d,'r')]=0; agg[(e,d,'g')]=0
    n=0
    for sc in scenes:
        hp=E/HULL/sc/"hull.npz"
        if not hp.is_file(): continue
        z=np.load(hp); occ=z["occupancy"].astype(bool); gm=z["grid_min"]; vs=float(z["voxel_size"]); shape=occ.shape
        gt=EM.solid_mesh_occ(sc,gm,vs,shape); mesh=np.zeros(shape,bool)
        for o in gt: mesh|=gt[o].astype(bool)
        agg["br"]+=int((occ&mesh).sum()); agg["bg"]+=int((occ&~mesh).sum())
        for e,d in combos:
            op=ndimage.binary_dilation(ndimage.binary_erosion(occ,se,iterations=e),se,iterations=d)
            agg[(e,d,'r')]+=int((op&mesh).sum()); agg[(e,d,'g')]+=int((op&~mesh).sum())
        n+=1
    br,bg=agg["br"],agg["bg"]
    print(f"\n===== {HULL} ({n}場) 26連通 =====")
    print(f"{'erode→dilate':12s} {'真實':>11} {'鬼影':>11} {'真實Δ%':>7} {'鬼影Δ%':>7} {'鬼影佔比%':>8}")
    print(f"{'原始':12s} {br:>11,} {bg:>11,} {'—':>7} {'—':>7} {bg/(br+bg)*100:>8.1f}")
    for e,d in combos:
        r,g=agg[(e,d,'r')],agg[(e,d,'g')]
        tag=f"e{e}→d{d}"+("(open)" if e==d else ("(淨縮)" if e>d else "(淨脹)"))
        print(f"{tag:12s} {r:>11,} {g:>11,} {(r-br)/br*100:>7.1f} {(g-bg)/bg*100:>7.1f} {g/(r+g)*100:>8.1f}")
