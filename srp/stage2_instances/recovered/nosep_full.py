"""正確『沒分開率』:兩物主instance相同=沒分開。全物體對 + on對,分n/occ/stack/all,12方法。用GT完美標籤(無zbuffer)。"""
import numpy as np, json, sys, glob, itertools
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,"srp/stage2_instances"); sys.path.insert(0,"srp/io")
from labels import label_dir
E=Path("data/eval"); GEXS={"bowl","skillet_lid","windex_bottle","colored_wood_blocks","dice"}
GT={"srp_hull_mv2_v12_am1":"srp_hull_gtlabel_am1","srp_hull_mv2_v12_am1_fp":"srp_hull_gtlabel_am1fp"}
HULLS={"srp_hull_mv2_v12_am1":[("瘦 fp 併最近","reNNfpS_am1"),("瘦 fp drop","reNNfpSd_am1"),("瘦 fp 侵蝕","reNNfpSe_am1"),
                               ("瘦 中心 併最近","reNNcS_am1"),("瘦 中心 drop","reNNcSd_am1"),("瘦 中心 侵蝕","reNNcSe_am1")],
 "srp_hull_mv2_v12_am1_fp":[("胖 fp 併最近","reNNfpS_am1fp"),("胖 fp drop","reNNfpSd_am1fp"),("胖 fp 侵蝕","reNNfpSe_am1fp"),
                            ("胖 中心 併最近","reNNcS_am1fp"),("胖 中心 drop","reNNcSd_am1fp"),("胖 中心 侵蝕","reNNcSe_am1fp")]}
def onpairs(sc):
    try: rel=json.loads((label_dir(sc)/"relations.json").read_text())
    except: return set()
    out=set()
    for r in rel.get("relations",[]):
        if r.get("type")=="on":
            x=r.get("x","").split("_",1)[-1]; y=r.get("y","").split("_",1)[-1]
            if x and y: out.add(frozenset((x,y)))
    return out
def grp(sc): return "stack" if sc.startswith("stack") else ("occ" if sc.startswith("occ") else "n")
scenes=sorted(Path(p).parent.name for p in glob.glob(str(E/"srp_hull_divB_t50_reNNfpSd_am1fp/*_scene*/instances.npz")))
allm=[m for ms in HULLS.values() for m in ms]
BK=["n","occ","stack","all"]
R={n:{b:{"af":0,"at":0,"of":0,"ot":0} for b in BK} for n,_ in allm}  # a=all pairs, o=on pairs; f=fail t=total
MINV=20
for si,sc in enumerate(scenes):
    G=grp(sc); ons=onpairs(sc)
    for hull,ms in HULLS.items():
        gtf=E/GT[hull]/sc/"instances.npz"
        if not gtf.is_file(): continue
        gz=np.load(gtf); glab=gz["labels"]; lmap={int(k):v for k,v in json.loads(str(gz["build_meta"]))["labelmap"].items()}
        idxs=np.argwhere(glab>0); gvals=glab[glab>0]
        objs=[i for i,nm in lmap.items() if nm not in GEXS]
        for name,rootsuf in ms:
            f=E/f"srp_hull_divB_t50_{rootsuf}"/sc/"instances.npz"
            if not f.is_file(): continue
            mv=np.load(f)["labels"][tuple(idxs.T)]
            objinst=defaultdict(lambda: defaultdict(int))
            for g,m in zip(gvals.tolist(),mv.tolist()):
                if m>0: objinst[g][m]+=1
            dom={}
            present=[]
            for oid in objs:
                di=objinst[oid]
                if di and sum(di.values())>=MINV:
                    dom[oid]=max(di,key=di.get); present.append(oid)
            for B in (G,"all"):
                a=R[name][B]
                for i in range(len(present)):
                    for j in range(i+1,len(present)):
                        oa,ob=present[i],present[j]
                        a["at"]+=1
                        if dom[oa]==dom[ob]: a["af"]+=1
                        pr=frozenset((lmap[oa],lmap[ob]))
                        if pr in ons:
                            a["ot"]+=1
                            if dom[oa]==dom[ob]: a["of"]+=1
    if (si+1)%80==0: print(f"..{si+1}",flush=True)
for B in BK:
    print(f"\n########## {B} 組 ##########")
    print(f"{'方法':16s} | {'全對沒分開':>7}/{'全對':<6} {'率%':>5} | {'on沒分開':>6}/{'on對':<4} {'on率%':>6}")
    for name,_ in allm:
        a=R[name][B]
        print(f"{name:16s} | {a['af']:>7}/{a['at']:<6} {a['af']/max(a['at'],1)*100:>5.1f} | {a['of']:>6}/{a['ot']:<4} {a['of']/max(a['ot'],1)*100:>6.1f}")
