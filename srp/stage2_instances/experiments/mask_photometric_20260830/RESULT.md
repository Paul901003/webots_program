# 遮罩對光度/語意 分同異物體(60 堆疊場)

日期 2026-08-30。腳本 `run.py`(env SAM_ROOT=mobilesamv2_fast CAPTURES_ROOT=captures_fast)。逐對存 `pairs.csv`。

## 設置
每視角 kept 遮罩(去手臂夾爪、只留對到 GT 物體 cover>0.5)→ donut → 每對算:CLIP cos(去偏)、ΔRGB(平均色 L2)、Δ色相hist(HSV H-S 分佈 Bhattacharyya)、Δ紋理(Sobel 梯度均值差);同/異 = 兩遮罩 GT 物體同否。stack3/4/5 共 60 場、12 視角。

## 數據(同 24065 / 異 37510 對)
| 量 | 同中位 | 異中位 | AUC(分同異) |
|---|:--:|:--:|:--:|
| CLIP cos | 0.53 | 0.44 | 0.653 |
| ΔRGB | 63.6 | 87.9 | 0.646 |
| Δ色相hist | 0.88 | 0.96 | 0.676 |
| Δ紋理 | 31.7 | 41.6 | 0.562 |

## 離群(記錄)
- **① 同物體卻色相差大(false alarm)**:top60 有 55 是 `colored_wood_blocks`(本身多色→兩遮罩色分佈天差地別 dhist=1.0)。
- **② 異物體卻色相近(confusion)**:`cracker_box↔sugar_box`(20)、`apple↔strawberry`、`baseball↔cracker_box`、`knife↔windex_bottle`(顏色像的不同物)。

## 結論(事實)
- 光度(RGB/色相/紋理)在 60 場都**弱(AUC ~0.65)**,不比 CLIP(0.653)好。
- 單場 stack3_scene0001 曾得色相 hist AUC 0.974,是**特例**(那場同物體=均勻色 cups、異物顏色差大);60 場一驗即破。
- 失敗兩型已定位:**多色物體**破壞同物體、**同色異物**造成混淆。→ 光度不採用;相疊分離最好仍是 z重疊(0.805)。
