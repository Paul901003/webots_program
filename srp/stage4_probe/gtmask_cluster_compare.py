#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""gtmask_cluster_compare.py — 在【GT 完美遮罩】下比較 CLIP vs DINOv2 兩種聚合法的語意分群品質(上界探針)。

★ 這是「語意特徵天花板」實驗,【不是】管線層級比較:
  - 餵的是 GT modal 遮罩(一物體一遮罩,完美),不是 MobileSAMv2 預測遮罩。
  - 因此數字【不可】與 srp_hull_cluster_mv2_noarm / _donut_noarm(SAM 遮罩、303場)相比。
  - 它回答的是:「就算遮罩完美,DINO 的語意特徵能不能比 CLIP 更分得開堆疊物?」

分群演算法與管線完全相同(voxel_sem_cluster_reassign_soliddrop.py 行 100-101):
    F  = debias(feats)
    cl = fcluster(linkage(pdist(F,"cosine"),"average"), t=thr, criterion="distance")
一場內把 12 視角的所有遮罩混在一起分一次;真值 = npz 的 names(該遮罩的真實物體)。

比較的四種特徵(同一批遮罩、同一演算法,只換特徵):
  clip_debias : CLIP 512d 扣 F_BG 去偏(★管線實際用法)
  clip_raw    : CLIP 512d 不去偏(對照,看去偏貢獻)
  dino_cw     : DINO 768d,cov>0 納入 + 面積加權
  dino_ce     : DINO 768d,cov>0 納入 + 等權平均
  ⚠ DINO 不做 F_BG 去偏:DINO 不摳圖不填灰(從原圖 patch token 取樣),沒有填充色偏置可扣。

指標(sklearn,逐場算後對場平均):
  homogeneity  低 = 混群(不同物體被併在一起)
  completeness 低 = 過切(同物體被拆成多群)
  v-measure    兩者調和平均
門檻:各特徵各自掃 0.05~0.95(步進 0.05),各報自己最佳 v-measure;另附 CLIP 管線預設 0.4。

拿什麼:data/eval/gt_mask_feats_v2/*.npz(60 場 stack3/4/5,2848 筆,12 視角 A-3 selected)。
排除項:GEX(skillet_lid/windex_bottle/colored_wood_blocks/dice)分開報,不預設濾除。
用法  : ./gtmask_cluster_compare.py        輸出 RESULT_gtmask_cluster_compare.md
"""
import sys
import glob
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import homogeneity_completeness_v_measure

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "srp" / "stage2_instances"))
import mask_clip_cluster as MC   # noqa: E402  只讀 F_BG 快取,不載 CLIP 模型

FEAT = REPO / "data" / "eval" / "gt_mask_feats_v2"
GEX = {"skillet_lid", "windex_bottle", "colored_wood_blocks", "dice"}
THRS = np.round(np.arange(0.05, 0.96, 0.05), 2)
VARIANTS = ["clip_debias", "clip_raw", "dino_cw", "dino_ce"]


def l2(F):
    return F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)


def debias_clip(F):
    """與管線 debias_feats 相同:扣掉 CLIP 純填充色特徵 F_BG 的分量後重新 L2。"""
    bg = MC.F_BG.astype(np.float64)
    F = F.astype(np.float64)
    return l2(F - (F @ bg)[:, None] * bg[None, :])


def load(exclude_gex):
    """回 [(scene, {variant: F}, labels)];exclude_gex=True 時濾掉 GEX 物體的筆。"""
    out = []
    for p in sorted(FEAT.glob("*.npz")):
        z = np.load(p)
        names = z["names"]
        keep = np.array([n.split("_", 1)[-1] not in GEX for n in names]) if exclude_gex \
            else np.ones(len(names), bool)
        if keep.sum() < 2 or len(set(names[keep].tolist())) < 2:
            continue                                   # 至少 2 筆且 ≥2 個物體才有分群意義
        F = {"clip_debias": debias_clip(z["clip"][keep]),
             "clip_raw": l2(z["clip"][keep].astype(np.float64)),
             "dino_cw": l2(z["dino_cw"][keep].astype(np.float64)),
             "dino_ce": l2(z["dino_ce"][keep].astype(np.float64))}
        out.append((p.stem, F, names[keep]))
    return out


def run(data, variant, thr):
    """逐場分群 → 逐場算 h/c/v 與群數,回各自的場平均。"""
    H, C, V, K, T = [], [], [], [], []
    for sc, F, lab in data:
        cl = fcluster(linkage(pdist(F[variant], "cosine"), "average"), t=thr, criterion="distance")
        h, c, v = homogeneity_completeness_v_measure(lab, cl)
        H.append(h); C.append(c); V.append(v); K.append(len(set(cl))); T.append(len(set(lab.tolist())))
    return dict(h=np.mean(H), c=np.mean(C), v=np.mean(V), k=np.mean(K), t=np.mean(T))


def block(data, tag, md):
    md.append(f"## {tag}(場數 {len(data)})\n")
    md.append("| 特徵 | 最佳門檻 | homogeneity | completeness | **v-measure** | 平均群數 | 平均真實物體數 |")
    md.append("|---|---|---|---|---|---|---|")
    best = {}
    for vn in VARIANTS:
        rs = [(run(data, vn, t), t) for t in THRS]
        r, t = max(rs, key=lambda x: x[0]["v"])
        best[vn] = (r, t, rs)
        md.append(f"| {vn} | {t:.2f} | {r['h']:.3f} | {r['c']:.3f} | **{r['v']:.3f}** | "
                  f"{r['k']:.1f} | {r['t']:.1f} |")
    md.append("")
    md.append("### 管線預設門檻 0.40 下(僅供對照;非各特徵最佳)\n")
    md.append("| 特徵 | homogeneity | completeness | v-measure | 平均群數 |")
    md.append("|---|---|---|---|---|")
    for vn in VARIANTS:
        r = run(data, vn, 0.40)
        md.append(f"| {vn} | {r['h']:.3f} | {r['c']:.3f} | {r['v']:.3f} | {r['k']:.1f} |")
    md.append("")
    return best


def main():
    if not FEAT.is_dir() or not list(FEAT.glob("*.npz")):
        print(f"缺 {FEAT}"); return
    md = ["# GT 完美遮罩下的語意分群:CLIP vs DINOv2(上界探針)\n",
          "- 建檔 2026-09-23;程式 `srp/stage4_probe/gtmask_cluster_compare.py`;可復現。",
          "- 資料 `data/eval/gt_mask_feats_v2/`(60 場 stack3/4/5、**12 視角 A-3 selected**、2848 筆物體×視角)。",
          "- 演算法與管線相同:去偏 → cosine → average linkage → 依距離門檻切(`fcluster criterion=distance`)。",
          "- 一場內 12 視角所有遮罩混在一起分一次;真值 = 該遮罩的真實物體名。逐場算指標後對場平均。",
          "- DINO 兩版【納入規則相同(cov>0)、只差加權】;DINO 不做 F_BG 去偏(不摳圖填灰,無該偏置)。\n",
          "> ⚠ **這是 GT 完美遮罩下的語意天花板,不是管線表現。**",
          "> 數字**不可**與 `srp_hull_cluster_mv2_noarm` / `_donut_noarm`(MobileSAMv2 遮罩、303 場)相比 —— 遮罩來源與場景母體都不同。",
          "> 也**不可**與 `CONCLUSION_voxeloverlap_semantic.md` 的 v-measure 0.674 相比 —— 那是 voxel重疊+語意、最終物體層級、SAM 遮罩。\n"]
    d_all = load(False); d_nog = load(True)
    block(d_all, "全部(含 GEX)", md)
    block(d_nog, "排除 GEX", md)
    md.append("**GEX = skillet_lid / windex_bottle / colored_wood_blocks / dice(方法不處理的物體)。**\n")
    out = HERE / "RESULT_gtmask_cluster_compare.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"[存檔] {out}\n\n" + "\n".join(md))


if __name__ == "__main__":
    main()
