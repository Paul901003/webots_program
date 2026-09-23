"""每物體『被混進別群』比例(連續,無門檻):該物體可見表面voxel中,落在『主導物體≠自己』instance的比例。
物體真實歸屬=重投影各視角比GT遮罩(即GT完美標籤)。報 per-object 平均。"""
import numpy as np, json, sys, glob
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
def grp(sc): return "stack" if sc.startswith("stack") else ("occ" if sc.startswith("occ") else "n")
def stackobjs(sc):
    try: rel=json.loads((label_dir(sc)/"relations.json").read_text())
    except: return set()
    s=set()
    for r in rel.get("relations",[]):
        if r.get("type")=="on":
            for kk in ("x","y"):
                nm=r.get(kk,"").split("_",1)[-1]
                if nm and nm not in GEXS: s.add(nm)
    return s
scenes=sorted(Path(p).parent.name for p in glob.glob(str(E/"srp_hull_divB_t50_reNNfpSd_am1fp/*_scene*/instances.npz")))
allm=[m for ms in HULLS.values() for m in ms]
BK=["n","occ","stack","all","stack物"]   # stack物=只算堆疊(on)相關物體
R={n:{b:[] for b in BK} for n,_ in allm}
for si,sc in enumerate(scenes):
    G=grp(sc); sobjs=stackobjs(sc)
    for hull,ms in HULLS.items():
        gtf=E/GT[hull]/sc/"instances.npz"
        if not gtf.is_file(): continue
        gz=np.load(gtf); glab=gz["labels"]; lmap={int(k):v for k,v in json.loads(str(gz["build_meta"]))["labelmap"].items()}
        idxs=np.argwhere(glab>0); gv=glab[glab>0]
        for name,rootsuf in ms:
            f=E/f"srp_hull_divB_t50_{rootsuf}"/sc/"instances.npz"
            if not f.is_file(): continue
            mv=np.load(f)["labels"][tuple(idxs.T)]
            # 每 instance 主導GT物體
            inst_obj=defaultdict(lambda: defaultdict(int))
            for g,m in zip(gv.tolist(),mv.tolist()):
                if m>0: inst_obj[m][g]+=1
            dom={k:max(d,key=d.get) for k,d in inst_obj.items()}
            # 每物體:被混進別群比例
            objtot=defaultdict(int); objleak=defaultdict(int)
            for g,m in zip(gv.tolist(),mv.tolist()):
                nm=lmap[g]
                if nm in GEXS: continue
                objtot[g]+=1
                if m>0 and dom.get(m)!=g: objleak[g]+=1   # 落在主導非自己的instance
            for g,tot in objtot.items():
                if tot<30: continue
                leak=objleak[g]/tot*100; nm=lmap[g]
                for B in (G,"all"):
                    R[name][B].append(leak)
                if nm in sobjs: R[name]["stack物"].append(leak)
    if (si+1)%80==0: print(f"..{si+1}",flush=True)
print("每物體『被混進別群』比例平均%(連續,無門檻;越低越好)")
print(f"{'方法':16s} {'n':>6} {'occ':>6} {'stack':>6} {'stack堆疊物':>10} {'all':>6}")
for name,_ in allm:
    a=R[name]
    def mv(b): return f"{np.mean(a[b]):.1f}" if a[b] else "–"
    print(f"{name:16s} {mv('n'):>6} {mv('occ'):>6} {mv('stack'):>6} {mv('stack物'):>10} {mv('all'):>6}")
