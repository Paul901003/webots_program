#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""量 kept 遮罩對 GT modal 的最大覆蓋 p*,分類:乾淨單物/跨多物/物外(手臂夾爪背景)/部分。
env: CAPTURES_ROOT=captures_fast SAM_ROOT=mobilesamv2_fast。用法: ./measure_coverage.py"""
import json, numpy as np, sys
from collections import defaultdict
from pathlib import Path
from pycocotools import mask as cocomask
import cv2
sys.path.insert(0,"srp/io"); sys.path.insert(0,"srp/stage2_instances")
import masks as MK, viewpoints as VP, labels as L
MV2=Path("data/eval/mobilesamv2_fast"); ARM=Path("data/eval/srp_arm_masks")
scenes=[p.name for g in ("n3","stack3","stack4","stack5","occ4") for p in sorted(MV2.glob(f"{g}_scene*"))][:40]
def modal(sc):
    ann=json.loads((L.label_dir(sc)/"actual"/"annotations.json").read_text())
    cat={c["id"]:c["name"] for c in ann["categories"]}; id2v={im["id"]:Path(im["file_name"]).stem for im in ann["images"]}
    md=defaultdict(dict)
    for a in ann["annotations"]:
        if a["category_id"]==1: continue
        s=a["segmentation"]; c=s["counts"].encode() if isinstance(s["counts"],str) else s["counts"]
        md[id2v[a["image_id"]]][cat[a["category_id"]]]=cocomask.decode({"size":s["size"],"counts":c}).astype(bool)
    return md
cnt=defaultdict(int); arm_hit=0; arm_tot=0
for sc in scenes:
    try: md=modal(sc)
    except: continue
    for vn in sorted(VP.selected_view_names(12)):
        vd=MV2/sc/vn
        if not vd.is_dir(): continue
        km=MK.kept_object_masks(vd); ms=[m for m,_ in km]
        if not ms: continue
        gmv=md.get(vn,{})
        if not gmv: continue
        union=np.zeros_like(ms[0]);
        for om in gmv.values(): union|=om
        ap=ARM/sc/f"{vn}_arm.png"; arm=(cv2.imread(str(ap),0)>127) if ap.is_file() else None
        for m in ms:
            a=int(m.sum())
            if a==0: continue
            pstar=max([(m&om).sum()/a for om in gmv.values()], default=0)
            onu=(m&union).sum()/a
            if pstar>=0.7: cnt["clean_single_p>=0.7"]+=1
            elif onu>=0.7: cnt["spans_multi(onunion>=0.7,single<0.7)"]+=1
            elif onu<0.3:
                cnt["outside_objects(arm/gripper/bg)"]+=1
                if arm is not None:
                    arm_tot+=1
                    if (m&arm).sum()/a>=0.5: arm_hit+=1
            else: cnt["partial(0.3-0.7)"]+=1
tot=sum(cnt.values())
print(f"masks={tot}")
for k,v in sorted(cnt.items(),key=lambda x:-x[1]): print(f"  {k}: {v} ({v/tot*100:.1f}%)")
if arm_tot: print(f"outside masks in arm silhouette >=50%: {arm_hit}/{arm_tot}={arm_hit/arm_tot*100:.0f}%")
