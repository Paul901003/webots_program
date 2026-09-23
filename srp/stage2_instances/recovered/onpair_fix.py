"""正確:on對『沒分開』=兩物主instance相同。另報污染(對方voxel佔自己主instance比例)。用GT完美標籤。"""
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
def onpairs(sc):
    try: rel=json.loads((label_dir(sc)/"relations.json").read_text())
    except: return []
    out=[]
    for r in rel.get("relations",[]):
        if r.get("type")=="on":
            x=r.get("x","").split("_",1)[-1]; y=r.get("y","").split("_",1)[-1]
            if x and y and x not in GEXS and y not in GEXS: out.append((x,y))
    return out
scenes=sorted(Path(p).parent.name for p in glob.glob(str(E/"srp_hull_divB_t50_reNNfpSd_am1fp/stack*_scene*/instances.npz")))
allm=[m for ms in HULLS.values() for m in ms]
R={n:{"fail":0,"tot":0,"contam":[]} for n,_ in allm}
perpair=defaultdict(dict)  # (sc,pair)->method->fail
for sc in scenes:
    ons=onpairs(sc)
    if not ons: continue
    for hull,ms in HULLS.items():
        gtf=E/GT[hull]/sc/"instances.npz"
        if not gtf.is_file(): continue
        gz=np.load(gtf); glab=gz["labels"]; lmap={int(k):v for k,v in json.loads(str(gz["build_meta"]))["labelmap"].items()}
        name2id={v:k for k,v in lmap.items()}
        idxs=np.argwhere(glab>0); gvals=glab[glab>0]
        for name,rootsuf in ms:
            f=E/f"srp_hull_divB_t50_{rootsuf}"/sc/"instances.npz"
            if not f.is_file(): continue
            mv=np.load(f)["labels"][tuple(idxs.T)]
            # 每物體→方法instance分佈
            objinst=defaultdict(lambda: defaultdict(int))
            for g,m in zip(gvals.tolist(),mv.tolist()):
                if m>0: objinst[g][m]+=1
            def dom(oname):
                oid=name2id.get(oname); 
                if oid is None or not objinst[oid]: return None,0,{}
                d=objinst[oid]; k=max(d,key=d.get); return k,d[k],d
            for (u,l) in ons:
                ku,cu,du=dom(u); kl,cl,dl=dom(l)
                R[name]["tot"]+=1
                if ku is None or kl is None: continue
                fail = (ku==kl)
                if fail: R[name]["fail"]+=1
                perpair[(sc,frozenset((u,l)))][name]=fail
                # 污染:u主instance裡有多少l的voxel / u主instance總(GT標到的)
                # (簡化:u主instance ku 裡 l 的voxel數 / ku 裡所有GT標到voxel)
    # (contam 省略聚合,主看 fail)
BKf=defaultdict(lambda:{"fail":0,"tot":0})
# 分組
def grp(sc): return "stack"
print("on對『沒分開』率(兩物主instance相同=沒分開;正確定義):\n")
print(f"{'方法':16s} {'沒分開':>6}/{'on對':<4} {'沒分開率%':>9}")
for name,_ in allm:
    a=R[name]; print(f"{name:16s} {a['fail']:>6}/{a['tot']:<4} {a['fail']/max(a['tot'],1)*100:>9.1f}")
# 每對被幾種方法「沒分開」
uniq=sorted(perpair.keys())
byN=defaultdict(int)
for k in uniq: byN[sum(perpair[k].values())]+=1
print(f"\n29 on對:被幾種方法『沒分開』分佈")
for n in range(13):
    if byN[n]: print(f"  {n}/12: {byN[n]} 對")
allf=sum(1 for k in uniq if sum(perpair[k].values())==12); nof=sum(1 for k in uniq if sum(perpair[k].values())==0)
print(f"12種全沒分開(幾何硬失敗)={allf}  全分開={nof}  方法有差={len(uniq)-allf-nof}")
