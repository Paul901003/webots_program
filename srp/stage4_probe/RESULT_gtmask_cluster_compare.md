# GT 完美遮罩下的語意分群:CLIP vs DINOv2(上界探針)

- 建檔 2026-09-23;程式 `srp/stage4_probe/gtmask_cluster_compare.py`;可復現。
- 資料 `data/eval/gt_mask_feats_v2/`(60 場 stack3/4/5、**12 視角 A-3 selected**、2848 筆物體×視角)。
- 演算法與管線相同:去偏 → cosine → average linkage → 依距離門檻切(`fcluster criterion=distance`)。
- 一場內 12 視角所有遮罩混在一起分一次;真值 = 該遮罩的真實物體名。逐場算指標後對場平均。
- DINO 兩版【納入規則相同(cov>0)、只差加權】;DINO 不做 F_BG 去偏(不摳圖填灰,無該偏置)。

> ⚠ **這是 GT 完美遮罩下的語意天花板,不是管線表現。**
> 數字**不可**與 `srp_hull_cluster_mv2_noarm` / `_donut_noarm`(MobileSAMv2 遮罩、303 場)相比 —— 遮罩來源與場景母體都不同。
> 也**不可**與 `CONCLUSION_voxeloverlap_semantic.md` 的 v-measure 0.674 相比 —— 那是 voxel重疊+語意、最終物體層級、SAM 遮罩。

## 全部(含 GEX)(場數 60)

| 特徵 | 最佳門檻 | homogeneity | completeness | **v-measure** | 平均群數 | 平均真實物體數 |
|---|---|---|---|---|---|---|
| clip_debias | 0.35 | 0.944 | 0.841 | **0.886** | 5.9 | 4.0 |
| clip_raw | 0.15 | 0.979 | 0.788 | **0.868** | 7.2 | 4.0 |
| dino_cw | 0.40 | 0.887 | 0.871 | **0.872** | 5.1 | 4.0 |
| dino_ce | 0.35 | 0.852 | 0.821 | **0.827** | 5.6 | 4.0 |

### 管線預設門檻 0.40 下(僅供對照;非各特徵最佳)

| 特徵 | homogeneity | completeness | v-measure | 平均群數 |
|---|---|---|---|---|
| clip_debias | 0.898 | 0.875 | 0.880 | 5.0 |
| clip_raw | 0.018 | 0.995 | 0.025 | 1.1 |
| dino_cw | 0.887 | 0.871 | 0.872 | 5.1 |
| dino_ce | 0.795 | 0.845 | 0.810 | 4.8 |

## 排除 GEX(場數 59)

| 特徵 | 最佳門檻 | homogeneity | completeness | **v-measure** | 平均群數 | 平均真實物體數 |
|---|---|---|---|---|---|---|
| clip_debias | 0.35 | 0.944 | 0.813 | **0.867** | 5.1 | 3.4 |
| clip_raw | 0.15 | 0.977 | 0.744 | **0.836** | 6.4 | 3.4 |
| dino_cw | 0.40 | 0.896 | 0.865 | **0.873** | 4.3 | 3.4 |
| dino_ce | 0.40 | 0.838 | 0.843 | **0.831** | 4.1 | 3.4 |

### 管線預設門檻 0.40 下(僅供對照;非各特徵最佳)

| 特徵 | homogeneity | completeness | v-measure | 平均群數 |
|---|---|---|---|---|
| clip_debias | 0.879 | 0.848 | 0.847 | 4.3 |
| clip_raw | 0.036 | 0.994 | 0.043 | 1.1 |
| dino_cw | 0.896 | 0.865 | 0.873 | 4.3 |
| dino_ce | 0.838 | 0.843 | 0.831 | 4.1 |

**GEX = skillet_lid / windex_bottle / colored_wood_blocks / dice(方法不處理的物體)。**
