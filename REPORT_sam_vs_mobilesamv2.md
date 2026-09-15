# 分割方法對照報告：sam_only vs MobileSAMv2（efficientvit_l2 / tiny_vit）

> 資料集：`captures_fast`（UR5e 手臂移動拍攝，34 視角/場景，367 場景：n1 64、n3/4/5 各 61、occ3/4/5 各 20、stack3/4/5 各 20）。
> 日期：2026-07。GPU：RTX 4070 Ti。全程免深度（RGB-only）。

## 1. 目的與方法

比較三種 class-agnostic 分割在**同一份拍攝資料 + 同一套下游**（visual hull → 實例關聯 → vs GT 評估）下的表現：

| 方法 | 分割原理 | 影像編碼器 | 遮罩輸出 |
|---|---|---|---|
| **sam_only** | SAM 自動網格點（32×32）→ 所有遮罩 | SAM ViT-B | 過分割、每 view ~15-20 塊 |
| **MobileSAMv2 (efficientvit_l2)** | ObjectAwareModel(YOLO) 出物件框 → 框→遮罩 | efficientvit_l2 (235MB) | 物件級、每 view ~14-18 塊 |
| **MobileSAMv2 (tiny_vit)** | 同上（YOLO 框） | tiny_vit / mobile_sam (39MB) | 物件級 |

**兩個評估層級**：
- **遮罩層**（`sam_recall_fast.py`）：每 (場景,視角,GT 物體)，該物 GT modal 遮罩 vs 該視角任一遮罩 best IoU ≥ 0.5 即「找到」。
- **3D 實例層**（`eval.py`）：遮罩雕 voxel visual hull（排手臂、soft am2）→ 跨視角實例關聯 → 3D IoU 匈牙利配對 pred instance ↔ GT 物體（GT 用 amodal 雕）。

## 2. 遮罩層：sam_recall（各組）

recall@.5 / meanIoU（物視角數）：

| 組 | 物視角 | sam_only | MobileSAMv2 l2 | MobileSAMv2 tiny_vit |
|---|---|---|---|---|
| n1 | 2176 | 0.949 / 0.919 | 1.000 / 0.966 | 1.000 / 0.960 |
| n3 | 6216 | 0.965 / 0.898 | 0.994 / 0.921 | 0.995 / 0.922 |
| n4 | 8247 | 0.965 / 0.895 | 0.996 / 0.920 | 0.995 / 0.920 |
| n5 | 10301 | 0.958 / 0.885 | 0.992 / 0.913 | 0.993 / 0.913 |
| occ3 | 2007 | 0.966 / 0.928 | 0.994 / 0.955 | 0.995 / 0.951 |
| occ4 | 2696 | 0.961 / 0.924 | 0.993 / 0.954 | 0.993 / 0.949 |
| occ5 | 3317 | 0.964 / 0.915 | 0.988 / 0.941 | 0.988 / 0.935 |
| stack3 | 2038 | 0.922 / 0.859 | 0.984 / 0.914 | 0.968 / 0.896 |
| stack4 | 2682 | 0.914 / 0.851 | 0.977 / 0.911 | 0.969 / 0.897 |
| stack5 | 3370 | 0.938 / 0.876 | 0.987 / 0.919 | 0.979 / 0.906 |
| **總計** | 43050 | 0.955 / 0.893 | 0.992 / 0.925 | 0.990 / 0.921 |

## 3. 3D 實例層：instance vs GT（各組，場景平均）

recall / precision / 3D IoU（場景數）：

| 組 | 場景 | sam_only | MobileSAMv2 l2 | MobileSAMv2 tiny_vit |
|---|---|---|---|---|
| n1 | 59 | 0.966 / 0.949 / 0.841 | 1.000 / 0.975 / 0.873 | 1.000 / 0.975 / 0.870 |
| n3 | 61 | 0.880 / 0.927 / 0.850 | 0.929 / 0.970 / 0.856 | 0.934 / 0.980 / 0.857 |
| n4 | 61 | 0.807 / 0.940 / 0.853 | 0.848 / 0.983 / 0.857 | 0.848 / 0.983 / 0.856 |
| n5 | 61 | 0.810 / 0.945 / 0.847 | 0.852 / 0.978 / 0.856 | 0.856 / 0.975 / 0.853 |
| occ3 | 20 | 0.850 / 0.958 / 0.892 | 0.867 / 1.000 / 0.889 | 0.867 / 1.000 / 0.887 |
| occ4 | 20 | 0.850 / 0.958 / 0.871 | 0.900 / 1.000 / 0.874 | 0.900 / 1.000 / 0.874 |
| occ5 | 20 | 0.880 / 0.904 / 0.871 | 0.920 / 0.967 / 0.874 | 0.930 / 0.967 / 0.868 |
| stack3 | 20 | 0.634 / 0.975 / 0.786 | 0.667 / 1.000 / 0.823 | 0.667 / 1.000 / 0.824 |
| stack4 | 20 | 0.637 / 0.983 / 0.816 | 0.650 / 1.000 / 0.830 | 0.650 / 1.000 / 0.829 |
| stack5 | 20 | 0.700 / 0.941 / 0.848 | 0.680 / 0.978 / 0.854 | 0.680 / 0.978 / 0.854 |
| **總計** | 362 | 0.830 / 0.944 / 0.848 | 0.865 / 0.981 / 0.859 | 0.867 / 0.982 / 0.858 |

## 4. 總計對照

| 層級 | 指標 | sam_only | MobileSAMv2 l2 | MobileSAMv2 tiny_vit |
|---|---|---|---|---|
| 遮罩層 | recall@.5 | 0.955 | **0.992** | 0.990 |
| 遮罩層 | meanIoU | 0.893 | **0.925** | 0.921 |
| 3D 實例 | recall | 0.830 | 0.865 | **0.867** |
| 3D 實例 | precision | 0.944 | 0.981 | **0.982** |
| 3D 實例 | 3D IoU | 0.848 | **0.859** | 0.858 |

## 5. 結論

1. **MobileSAMv2（兩編碼器）全面勝過 sam_only**：遮罩層與 3D 實例層、每個指標皆較高。
   - 遮罩層 recall +0.037（0.955→0.992）、3D 實例 recall +0.035（0.830→0.865）、precision +0.037（0.944→0.981）。
2. **precision 提升最顯著**（0.944→0.981，occ/stack 多組達 1.000）：MobileSAMv2 物件級遮罩**幻影實例少**，反觀 sam_only 網格過分割會碎裂出假實例。
3. **難組（stack）提升最大**：堆疊相觸場景 sam_only recall 0.63-0.70，MobileSAMv2 的 YOLO 物件偵測更能整塊抓到（遮罩層 stack recall +0.05~0.06）。
4. **tiny_vit ≈ efficientvit_l2**：遮罩層 tiny_vit 微低（-0.002 recall），**3D 實例層 tiny_vit 反而微高**（recall 0.867 vs 0.865）。原因：物件框由 YOLO 決定（主導 recall/precision），編碼器只微幅影響遮罩邊界，下游 hull+IoU 對此不敏感。
   - **效率含意**：tiny_vit 編碼器僅 39MB（vs l2 235MB）、更快，卻拿到相同水準結果 → 更划算，且仍明顯優於 sam_only。

## 6. 各物體救回細分（遮罩層 recall，`sam_recall_perobj.py`）

MobileSAMv2 相對 sam_only 的 recall 提升**集中在困難幾何物體**。以下為提升最多者（Δ = mv2_l2 − sam_only，物視角 ≥ 12）：

| 物體 | 物視角 | sam_only | mv2_l2 | mv2_tiny | Δ | 幾何難點 |
|---|---|---|---|---|---|---|
| **059_chain** | 34 | **0.000** | **1.000** | 1.000 | **+1.000** | 鏤空細鏈（sam_only 全滅）|
| **028_skillet_lid** | 442 | 0.627 | 0.991 | 0.998 | +0.364 | 扁平中空環 |
| **062_dice** | 583 | 0.587 | 0.950 | 0.950 | +0.364 | 極小骰子 |
| **070-a_colored_wood_blocks** | 884 | 0.657 | 0.990 | 1.000 | +0.333 | 多塊積木 |
| **035_power_drill** | 442 | 0.778 | 0.998 | 0.995 | +0.219 | 不規則 |
| **037_scissors** | 476 | 0.803 | 0.998 | 0.998 | +0.195 | 薄 |
| 007_tuna_fish_can | 847 | 0.855 | 0.961 | 0.921 | +0.106 | 矮圓罐 |
| 008_pudding_box | 814 | 0.905 | 0.969 | 0.948 | +0.064 | — |
| 031_spoon | 501 | 0.934 | 0.990 | 0.990 | +0.056 | 薄 |
| 030_fork | 611 | 0.943 | 0.989 | 0.987 | +0.046 | 薄長 |

**觀察**：
- 救回的正是 sam_only 舊報告 §2.2 的**弱點清單**（chain / skillet_lid / dice / wood_blocks / power_drill / scissors，原 recall 0.0~0.8）→ MobileSAMv2 全部拉到 **0.95~1.0**。
- **059_chain 0.000 → 1.000** 最戲劇性：網格點對鏤空細鏈抓不到，YOLO 物件偵測則整塊框到。
- 原理：困難幾何（薄/小/鏤空/多塊/不規則）對「網格點自動切」不利，對「物件感知偵測」友善。
- **tiny_vit 救回同一批物體**（差異 <0.05），再證輕量編碼器夠用。
- sam_only 已做好的規則物體（球/罐/杯/水果，>0.95）兩者打平（Δ<0.03）。

全 64 物體明細見 `perobj_compare.txt`。

## 附錄：檔案與重現

**遮罩輸出**：`data/eval/{sam_only_fast, mobilesamv2_fast, mobilesamv2_tiny_vit_fast}/<場景>/<view>/masks/`
**3D 評估**：`data/eval/{srp_hull_fast, srp_hull_mobilesamv2, srp_hull_mv2_tiny_vit}/d1d2.csv`
**遮罩 recall**：`{sam_only_recall, mv2_recall, mv2_tinyvit_recall}.txt`

重現（MobileSAMv2 為例）：
```bash
# 分割(encoder 可換 tiny_vit):
MV2_ENCODER=efficientvit_l2 mobilesamv2/run_mobilesamv2_all.sh
# 下游 hull→instance→eval(共用 arm 剪影 + GT 快取):
MV2_ENCODER=efficientvit_l2 mobilesamv2/run_downstream_mv2.sh
# 遮罩層比較:
SAM_ROOT=$PWD/data/eval/mobilesamv2_fast srp/stage4_probe/sam_recall_fast.py
```
