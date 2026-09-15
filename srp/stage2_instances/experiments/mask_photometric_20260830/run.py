#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""60 堆疊場:每視角遮罩對的 CLIP/ΔRGB/Δ色相hist/Δ紋理 分同異物體 + 記錄離群。
輸入 mobilesamv2_fast 遮罩 + captures_fast RGB + GT modal(貼標)。donut 後算。存 pairs.csv、印 AUC+離群。
env: SAM_ROOT CAPTURES_ROOT。用法: ./run.py"""
import numpy as np, sys, json, cv2, csv
from collections import defaultdict, Counter
from pathlib import Path
from pycocotools import mask as cocomask
from scipy.stats import rankdata
REPO=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(REPO/"srp/io")); sys.path.insert(0,str(REPO/"srp/stage2_instances"))
import masks as MK, viewpoints as VP, labels as L
import mask_clip_cluster as MC
from voxel_sem_cluster_donut import donut_masks
MV2=Path("data/eval/mobilesamv2_fast"); CAP=Path("data/captures_fast"); ARM=Path("data/eval/srp_arm_masks")
HERE=REPO/"srp/stage2_instances/experiments/mask_photometric_20260830"
FBG=MC.F_BG.astype(np.float64)
def debias(F):
    F=np.atleast_2d(F).astype(np.float64); F=F-(F@FBG)[:,None]*FBG[None,:]
    return F/(np.linalg.norm(F,axis=1,keepdims=True)+1e-9)
def auc(a,b):
    a,b=np.array(a),np.array(b)
    if len(a)==0 or len(b)==0: return float("nan")
    r=rankdata(np.concatenate([a,b])); U=r[:len(a)].sum()-len(a)*(len(a)+1)/2; return U/(len(a)*len(b))
def modal(sc):
    ann=json.loads((L.label_dir(sc)/"actual"/"annotations.json").read_text())
    cat={c["id"]:c["name"] for c in ann["categories"]}; id2v={im["id"]:Path(im["file_name"]).stem for im in ann["images"]}
    md=defaultdict(dict)
    for a in ann["annotations"]:
        if a["category_id"]==1: continue
        s=a["segmentation"]; c=s["counts"].encode() if isinstance(s["counts"],str) else s["counts"]
        md[id2v[a["image_id"]]][cat[a["category_id"]]]=cocomask.decode({"size":s["size"],"counts":c}).astype(bool)
    return md
scenes=[p.name for g in ("stack3","stack4","stack5") for p in sorted(MV2.glob(f"{g}_scene*"))]
rows=[]
for si,sc in enumerate(scenes):
    md=modal(sc); sdir=CAP/f"multi_{sc.split('_')[0]}"/sc
    for vn in sorted(VP.selected_view_names(12)):
        vd=MV2/sc/vn
        if not vd.is_dir(): continue
        km=MK.kept_object_masks(vd); ms0=[m for m,_ in km]
        ap=ARM/sc/f"{vn}_arm.png"; arm=(cv2.imread(str(ap),0)>127) if ap.is_file() else None
        if arm is not None:
            keep=[k for k,m in enumerate(ms0) if (m&arm).sum()/max(int(m.sum()),1)<0.5]; ms0=[ms0[k] for k in keep]
        gmv=md.get(vn,{})
        lab=[]
        for m in ms0:
            a=int(m.sum()); best=None;bc=0.5
            for on,om in gmv.items():
                cov=(m&om).sum()/max(a,1)
                if cov>bc: bc=cov; best=on
            lab.append(best)
        idx=[i for i,b in enumerate(lab) if b is not None]
        if len(idx)<2: continue
        ms=[ms0[i] for i in idx]; objs=[lab[i] for i in idx]
        rgb=cv2.cvtColor(cv2.imread(str(sdir/f"{vn}.png")),cv2.COLOR_BGR2RGB)
        hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV); gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(float)
        gmag=cv2.magnitude(cv2.Sobel(gray,cv2.CV_64F,1,0),cv2.Sobel(gray,cv2.CV_64F,0,1))
        don=donut_masks(ms); cf=MC.clip_feats(rgb,don,"mean")
        val=[k for k in range(len(cf)) if cf[k] is not None]
        if len(val)<2: continue
        ms=[ms[k] for k in val]; objs=[objs[k] for k in val]; don=[don[k] for k in val]
        feats=debias(np.array([cf[k] for k in val]))
        mrgb=[rgb[d].mean(0) if d.sum()>0 else np.zeros(3) for d in don]
        hist=[]
        for d in don:
            h=cv2.calcHist([hsv],[0,1],d.astype(np.uint8),[30,32],[0,180,0,256]); cv2.normalize(h,h); hist.append(h)
        tex=[gmag[d].mean() if d.sum()>0 else 0 for d in don]
        for i in range(len(ms)):
            for j in range(i+1,len(ms)):
                rows.append([sc,vn,objs[i].split("_",1)[-1],objs[j].split("_",1)[-1],int(objs[i]==objs[j]),
                             round(float(feats[i]@feats[j]),3),round(float(np.linalg.norm(mrgb[i]-mrgb[j])),1),
                             round(float(cv2.compareHist(hist[i],hist[j],cv2.HISTCMP_BHATTACHARYYA)),3),
                             round(float(abs(tex[i]-tex[j])),1)])
    if (si+1)%15==0: print(f"  {si+1}/{len(scenes)}",flush=True)
with open(HERE/"pairs.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(["scene","view","objA","objB","same","clipcos","drgb","dhist","dtex"]); w.writerows(rows)
A=np.array(rows,dtype=object); same=A[:,4].astype(int)==1
print(f"\n===== 60 堆疊場 (同 {same.sum()} / 異 {(~same).sum()} 對) =====")
print(f"{'量':<10}{'同中位':>8}{'異中位':>8}{'AUC':>8}")
for ci,nm,hi in [(5,"CLIPcos",False),(6,"ΔRGB",True),(7,"Δ色相hist",True),(8,"Δ紋理",True)]:
    v=A[:,ci].astype(float); s=v[same]; t=v[~same]
    a=auc(t,s) if hi else auc(s,t)
    print(f"{nm:<10}{np.median(s):>8.2f}{np.median(t):>8.2f}{a:>8.3f}")
# 離群:同物體對 dhist 最高(該低卻高)/ 異物體對 dhist 最低(該高卻低)
dh=A[:,7].astype(float)
si_hi=np.where(same)[0][np.argsort(-dh[same])[:15]]
di_lo=np.where(~same)[0][np.argsort(dh[~same])[:15]]
print("\n--- 離群① 同物體卻色相差大(dhist高)Top15 ---")
for k in si_hi: print(f"  {A[k,0]}/{A[k,1]} {A[k,2]}↔{A[k,3]} dhist={A[k,7]}")
print("--- 離群② 異物體卻色相近(dhist低)Top15 ---")
for k in di_lo: print(f"  {A[k,0]}/{A[k,1]} {A[k,2]}↔{A[k,3]} dhist={A[k,7]}")
print("\n--- 離群物體統計 ---")
print("① 同物體高dhist 最常見物體:",Counter(A[np.where(same)[0][np.argsort(-dh[same])[:60]],2]).most_common(6))
print("② 異物體低dhist 最常見物體對:",Counter(tuple(sorted((A[k,2],A[k,3]))) for k in np.where(~same)[0][np.argsort(dh[~same])[:60]]).most_common(6))
print(f"\npairs.csv 存於 {HERE}")
