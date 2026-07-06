# 免深度多視角管線實驗報告:SAM → Voxel Visual Hull → 實例分離 → 物件關係

> 範圍:從 SAM 遮罩 + voxel visual hull 雕殼起,經實例關聯、評估、關係生成、GNN 必要性裁決,
> 到免深度相觸/支撐物體分離。全程**免深度(RGB-only)**,不使用感測深度(plan C-DEP)。
> 計算用 torch GPU(RTX 4070 Ti)。日期:2026-06。工具/資料/重現指令見文末附錄。

---

## 0. 管線概觀

```
多視角 RGB ──[SAM 遮罩]──> 各視角遮罩
                              │
            [voxel visual hull 雕殼]      ← Stage 1(carve.py)
                              │  hull.npz(occupancy + observed)
            [跨視角 voxel 關聯]           ← Stage 2(associate.py)
                              │  instances.npz(每物體 voxel 群)
            [評估 / 關係生成 / 分離]      ← Stage 3–4
```

座標:機器人底座 `[-0.4,0,0]`;look-at 中心 x=0.35;拍攝半徑 0.65;相機水平 FOV `HFOV=1.4746`;
拍攝視角 12 個(方位 135–225° 單側、繞中心)。輸入只有 **SAM 預測遮罩 + 相機外參(pose.json,plan 允許的模擬 GT 外參)+ 內參**,**不用深度、不用物體 GT**。
(實驗場景幾何可視化:`worlds/experiment_viz.wbt`——手臂+桌面+12 相機(紅)+ look-at 中心(藍)+ 拍攝半月(紅半透明)+ 工作空間球(藍半透明 r0.35)+ voxel 工作空間(白半透明 0.7×0.7×0.35)。)

### 0.1 視角分佈與低仰角覆蓋限制
12 視角仰角分佈:**45/60/75° 各 3 台(方位 135/180/225,對稱)、90° 1 台(頂點)、20° 僅 2 台(方位 200/210,擠在 −y 側、不對稱)**。
成因(查 `controllers/ycb_supervisor_capture/candidate_viewpoint_config.py` 與候選/選取):
① 常規 sweep 只涵蓋 45/60/75/90°×8 方位,**20° 靠 `EXTRA_VIEWPOINTS_DEG` 額外加且只定義方位 150–210°**(az<150/>210 未當候選);
② IK 驗證:150/170/180/190/200/210 **可達**,僅 160 無解;
③ **貪婪 12-pose 選取只挑了 200/210**。
→ 低仰角覆蓋偏 −y 側、缺 +y 側,**並非可達性失敗**(150–190 可達卻未被選),而是候選定義範圍 + 貪婪選取所致;此覆蓋不均與 §2 的 hull 覆蓋不均/陰影過估計相關。

---

## 1. Stage 1:SAM 遮罩 + Voxel Visual Hull 雕殼(實驗 C1)

### 設計(`srp/stage1_hull/carve.py`)
- 體素網格:`BOX=[0,−0.35,0]…[0.7,0.35,0.35]`、voxel `0.005 m`(≈137 萬體素)、桌面 `TABLE_Z=0` 封底。
- 空間雕刻(space carving):每體素投影到各視角,落在前景遮罩內才保留;硬交集 = 需**全視角**前景。
- 輸出 `VisualHull{occupancy, observed, grid_min, voxel_size}`:
  - `occupancy`=被雕保留的體素;`observed`=視錐內被看過的體素。
  - 四象限語意(plan T10):空&已觀察=確定為空;空&未觀察=未觀測空間(場景 3a);佔據&未觀察=0(看過才保留)。
- **soft carving**(`allow_miss=k`):容忍 k 個視角漏檢(票數 ≥ V−k 即保留)→ 救「部分視角被遮/SAM 漏」的物體。
- GPU:voxel 0.005 約 0.65 s/場。

### 驗證
- 合成測試 T1–T10(`test_carve.py`)**12/12 通過**(含 T2 方向、T6 對齊 IoU 0.998、T10 observed 旗標)。
- 已確認 hull 生成**只用 SAM 預測遮罩 + 相機外/內參**,**無深度、無物體 GT/mesh**。

---

## 2. Stage 2:跨視角 voxel 實例關聯(實驗 D1/D2)

### 設計(`srp/stage2_instances/associate.py`)
- 每視角建 label 圖(每塊 SAM 遮罩一個 id,排除地板;重疊時 `cover` 控小/大遮罩勝)。
- 每個佔據 voxel 投影回各視角 → 取得「**跨視角 label 向量**」。
- **6-鄰接 + 「相鄰兩 voxel 在共同可見視角上 label 一致比例 ≥ `agree_frac`」才 union-find 合併** → 帶 instance 標籤的佔據網格。
- 輸出 `instances.npz`(labels)。

### 評估方法
- `eval.py`:GT 用 **amodal 遮罩**雕每物體 GT 視覺 hull → 3D IoU 匈牙利配對 → D1(found/recall/prec/mIoU)+ D2(vs 3D 連通元件)。
- `eval_mesh.py`:vs **強制實心 mesh**(sample+fill_holes)→ 找到=覆蓋率≥0.5(不懲罰膨脹),量冗餘/膨脹。
- `eval_reproj2d.py`:殼重投影 vs **modal 遮罩**(含遮擋)→ 2D recall(蓋住可見)/ prec。
- `viz_reproj.py`:疊圖可視化(綠=吻合/紅=殼超出/藍=殼漏)+ IoU 排名。

### 全場景基線(硬交集 am0,vs GT 視覺 hull;D1 = eval.py @IoU 0.25)

| 組 | 場景 | recall | precision | mIoU |
|---|---|---|---|---|
| n1 | 58 | 0.931 | 0.914 | 0.879 |
| n3 | 61 | 0.863 | 0.949 | 0.934 |
| n4 | 61 | 0.791 | 0.929 | 0.908 |
| n5 | 61 | 0.810 | 0.930 | 0.915 |
| stack3 | 20 | 0.634 | 0.963 | 0.842 |
| stack4 | 20 | 0.637 | 0.955 | 0.833 |
| stack5 | 20 | 0.670 | 0.909 | 0.879 |
| occ3 | 20 | 0.833 | 0.931 | 0.931 |
| occ4 | 20 | 0.875 | 0.958 | 0.924 |
| occ5 | 20 | 0.860 | 0.902 | 0.910 |
| **小計 multi_n** | 241 | 0.848 | 0.931 | 0.909 |
| **小計 stack** | 60 | **0.647** | 0.942 | 0.851 |
| **小計 occ** | 60 | 0.856 | 0.930 | 0.922 |
| **總計** | **361** | **0.816** | 0.933 | 0.902 |

- **stack recall 最低(0.647)**:堆疊物相觸**融成單一 instance、找不到** → 直接對應 §5 的 on=0;precision 仍高(0.94)、mIoU 0.85。
- **occ recall 高(0.856,甚至 > n4/n5)**:遮擋場景物體**並排相觸**(非上下堆疊),分得開 → 比隨機 n4/n5 還好。
- **precision/mIoU 各組全程穩定**(0.90–0.96 / 0.83–0.93)→ Stage 2 站得住。
- 結論:**堆疊(垂直相觸)是 recall 殺手,遮擋(水平並排)不是**;recall 缺口主要來自 touching(尤其垂直)欠分割。

### 參數掃描(18 組 = allow_miss × cover × agree_frac)
- `allow_miss` 是**真實取捨**(三套指標一致):

  | | 3D found | 3D mIoU | 3D 冗餘 | 3D 膨脹 | 2D recall | 2D(1−prec) |
  |---|---|---|---|---|---|---|
  | am0(硬) | 0.776 | **0.913** | 0.285 | 1.41 | 0.842 | 0.045 |
  | am1 | 0.814 | 0.854 | 0.330 | 1.58 | 0.877 | 0.069 |
  | am2 | 0.844 | 0.731 | 0.399 | 1.91 | 0.906 | 0.115 |

  am↑ → 找到/覆蓋↑、但過估計(冗餘/膨脹)↑、mIoU↓。
- `cover=large` 較穩、`agree_frac=0.5` 影響微小(已設為預設)。

### 兩個關鍵感知發現
1. **陰影過估計,且 2D 量不到**:am0 時 **3D 冗餘 28% vs 2D 超出可見僅 4.5%**——落差就是「陰影」(物體背向手臂側、方位 135–225 單側無相機覆蓋方向的剪影錐延伸),它**躲在物體背後,拍攝視角重投影看不到** → 只有 3D vs mesh 量得到。**參數調不掉,降它要靠視角覆蓋或深度(C-DEP 刻意不用)**(plan P2 固有過估計)。
2. **薄/扁物與被遮物被硬交集雕掉(藍色殼漏)**:低重投影 IoU 場景分兩型——薄物(skillet_lid/marker/wood_blocks)、擁擠遮擋。診斷確認是 **Stage 1 硬交集雕掉**(非 Stage 2 關聯):skillet_lid 雕刻覆蓋 am0=0.01→am2=0.59、mustard(遮擋)0.05→0.70。**soft carving 正是救回這些**(am0→am2 recall 0.78→0.84 的來源),代價是其他殼膨脹 → allow_miss 是「救薄物/遮物 vs 保真」取捨。

### 2.2 SAM 遮罩品質(每物-視角找到率,`sam_recall.py`)
指標:每 (場景, 視角, GT 物體),該物 GT modal 遮罩與該視角任一 class-agnostic SAM 遮罩 **best IoU ≥ 0.5 即「找到」**。recall@.5 = 找到 / 物-視角數;meanIoU = best IoU 平均。

| 組 | 物-視角數 | recall@.5 | meanIoU |
|---|---|---|---|
| n1 | 768 | 0.934 | 0.907 |
| n3 | 2194 | 0.975 | 0.909 |
| n4 | 2900 | 0.969 | 0.903 |
| n5 | 3625 | 0.964 | 0.895 |
| stack3 | 717 | 0.932 | 0.869 |
| stack4 | 945 | 0.908 | 0.847 |
| stack5 | 1187 | 0.928 | 0.870 |
| occ3 | 705 | 0.962 | 0.924 |
| occ4 | 950 | 0.967 | 0.927 |
| occ5 | 1160 | 0.966 | 0.916 |
| **小計 multi_n** | 9487 | 0.965 | 0.902 |
| **小計 stack** | 2849 | **0.922** | 0.862 |
| **小計 occ** | 2815 | 0.965 | 0.922 |
| **總計** | 15151 | **0.957** | 0.898 |

- **SAM 2D 找到率高(總計 0.957、meanIoU 0.898)** → SAM 大致都找得到物體、遮罩品質好。
- **stack 最低(0.922)**:堆疊遮擋/相觸使 SAM 略難。
- **意涵**:**SAM 不是主要瓶頸**;感知失敗(空 hull、融合)主要來自**硬交集雕刻 + 關聯**——即使 SAM 0.93+,硬交集需「全 12 視角」前景,薄物只要漏 1–2 張就空 hull(見 §2.3 A2)。

**SAM 最常漏的物體(per-object,物-視角數≥12,跨全部場景彙整)**
| 物體 | 物-視角 | recall@.5 | meanIoU | 難點型態 |
|---|---|---|---|---|
| 059_chain | 12 | **0.000** | 0.002 | 鏤空細鏈 |
| 028_skillet_lid | 156 | 0.603 | 0.573 | 扁平中空環 |
| 070-a_colored_wood_blocks | 312 | 0.606 | 0.645 | 多塊組合 |
| 062_dice | 197 | 0.614 | 0.525 | 極小 |
| 037_scissors | 168 | 0.792 | 0.701 | 薄 |
| 035_power_drill | 156 | 0.827 | 0.828 | 不規則 |
| 007_tuna_fish_can | 299 | 0.860 | 0.829 | 矮圓罐(易與桌/他物混) |
| 008_pudding_box | 288 | 0.903 | 0.882 | — |
| 030_fork | 216 | 0.917 | 0.758 | 薄長 |
| 032_knife | 191 | 0.932 | 0.816 | 薄長 |

→ SAM 只在**少數困難幾何**(鏤空 / 扁中空 / 多塊 / 極小 / 薄 / 不規則)recall 偏低,其餘物體多 > 0.94。chain 幾乎全滅、skillet_lid/wood_blocks-a/dice ~0.6 是 SAM 真正的弱點(對應 §2.3 A1)。

### 2.3 代表性失敗案例(附場景編號、推測原因、物體幾何)

**總覽:跨多物場景端到端失敗率 >50% 的物體(`per_obj_found.py`,僅 n3/n4/n5/stack/occ)**
> 找到 = 該物被某 instance 覆蓋 ≥0.5 的場景比例;失敗率 = 1−找到率。僅計多物場景(只在 n1 出現的物如 059_chain 已排除)。**62 種物體中 11 種 >50%**:

| 物體 | 出現 | 找到 | 失敗率 | 主因(對照 SAM recall) |
|---|---|---|---|---|
| 028_skillet_lid | 12 | 0 | **1.00** | SAM(0.60)+雕刻(扁中空環) |
| 070-a_colored_wood_blocks | 25 | 0 | **1.00** | SAM(0.61)+碎裂(多塊) |
| 062_dice | 17 | 1 | 0.94 | SAM(0.61,極小) |
| 031_spoon | 14 | 1 | 0.93 | 雕刻(SAM 0.96,薄) |
| 037_scissors | 13 | 1 | 0.92 | 雕刻(SAM 0.79,薄) |
| 035_power_drill | 12 | 2 | 0.83 | 雕刻/不規則(SAM 0.83) |
| 032_knife | 15 | 4 | 0.73 | 雕刻(SAM 0.93,薄) |
| 030_fork | 17 | 5 | 0.71 | 雕刻(SAM 0.92,薄) |
| 040_large_marker | 15 | 6 | 0.60 | 雕刻(細桿) |
| 043_phillips_screwdriver | 17 | 7 | 0.59 | 雕刻(薄) |
| 050_medium_clamp | 17 | 8 | 0.53 | 雕刻(薄夾) |

→ 全為**薄/扁/小/多塊/不規則**幾何。**多數是雕刻失敗**(SAM 找得到、硬交集仍雕掉/覆蓋不足:spoon/knife/fork/scissors/marker/screwdriver/clamp);**少數 SAM 也漏**(skillet_lid、wood_blocks-a、dice)。

**A. n1 單物失敗(統一基準:SAM 2D recall + 空hull,無融合干擾;依「SAM 有沒有找到」分因)**
> 同一基準下重新判定(原先把 power_drill/wood_blocks-a 誤歸 SAM、skillet_lid 誤歸雕刻,已修正)。

**A1. SAM 漏(SAM 2D recall < 0.6 → 根因是分割)**
| 場景 | 物體 | 尺寸(最小邊) | SAM2D recall | 空hull |
|---|---|---|---|---|
| n1_scene0048 | 059_chain | (0.31,0.307,0.025) | **0.00** | 是 | (鏤空細鏈,完全找不到) |
| n1_scene0050 | 062_dice | (0.017,0.017,0.018) | **0.00** | 是 | (極小 17mm) |
| n1_scene0024 | 028_skillet_lid | (0.268,0.269,0.076) | **0.33** | 是 | (扁中空環,多視角漏) |
| n1_scene0062 | 070-b_colored_wood_blocks | (0.035,0.035,0.026) | **0.58** | 是 | (小塊,半數漏) |

**A2. 雕刻失敗(SAM 2D recall ≥ 0.6,SAM 有找到但 hull 空/recall 低 → 根因是硬交集)**
| 場景 | 物體 | 尺寸(最小邊) | SAM2D recall | 空hull |
|---|---|---|---|---|
| n1_scene0037 | 044_flat_screwdriver | (0.162,0.157,0.035) | 0.92 | 是 |
| n1_scene0032 | 037_scissors | (0.096,0.202,0.016) | 0.92 | 否(有hull,recall 0) |
| n1_scene0030 | 035_power_drill | (0.184,0.188,0.057) | 0.92 | 否(有hull,recall 0) |
| n1_scene0061 | 070-a_colored_wood_blocks | (0.142,0.141,0.165) | 0.75 | 否(多塊→碎/低IoU) |
| n1_scene0026 | 030_fork | (0.198,0.027,0.016) | 0.67 | 是 |
| n1_scene0028 | 032_knife | (0.215,0.021,0.023) | 0.67 | 是 |

→ 讀法:**chain/dice/skillet_lid/wood_blocks-b 是 SAM 真的找不到**(鏤空/極小/扁環);**其餘薄長物(fork/knife/screwdriver/scissors/power_drill/wood_blocks-a)SAM 都找到了,是硬交集雕掉/碎裂**。

**B. 多物薄物(硬交集雕掉,於多物場景)**
| 場景 | 物體 | 尺寸 | 原因 |
|---|---|---|---|
| n3_scene0050 | 040_large_marker | (0.021,0.121,**0.019**) | 細桿筆 → 殼vs modal IoU 0.13 |
| n3_scene0003 | 030_fork(z0.016)+042_wrench(z0.015) | — | 兩薄物雕掉 → 3物→1 inst,recall 0.33 |

**C. hull 小物體失敗(體素太少)**
| 場景 | 物體 | 尺寸 | 原因 |
|---|---|---|---|
| n5_scene0016 | 057_racquetball / 058_golf_ball | (0.056) / (0.043) | 小球投影體素少、易被併/漏 → 5物→2,recall 0.4 |

**D. 欠分割/融合(相觸 → 併成單一 instance)**
| 場景 | 情況 | 原因 |
|---|---|---|
| stack4_scene0019 | 4物→pred 1,recall 0.25 | 垂直堆疊相觸融合 |
| stack5_scene0010 | 5物→pred 2,recall 0.2 | 垂直堆疊融合 |
| n5_scene0016 | 5物→pred 2,recall 0.4 | 水平靠太近相觸融合 |
| n3_scene0003 | 3物→pred 1,recall 0.33 | 水平融合 + 薄物雕掉雙重 |

**E. 幻影 phantom(多出假 instance,precision 低)**
| 場景 | phantom/n_pred/prec | 原因 |
|---|---|---|
| occ3_scene0016 | 4 / 5 / 0.20 | 070-a 多塊積木 → 過分割成多假 instance(+陰影碎塊) |
| n5_scene0037 | 4 / 8 / 0.50 | 細長物(drill、screwdriver)殼碎裂 + 陰影分裂假塊 |
| stack4_scene0017 | 4 / 7 / 0.43 | 堆疊 + 碎裂 |

**F. 其他重要失敗**
| 場景 | 物體 | 原因 |
|---|---|---|
| n5_scene0013 | 006_mustard_bottle (0.097,0.067,0.191) | 嚴重遮擋 → 殼vs modal IoU 0 |
| n3_scene0049 | 028_skillet_lid | 扁環+多物 → IoU 0.014(最差) |
| 任含 024_bowl 場景 | 024_bowl (0.161,0.161,0.055) | visual hull 把碗內凹填實 → 固有過估計 |

檢視:3D(B–F)用 `worlds/hull_viz.wbt`(`SRP_VIZ_ARGS="<scene> 1" webots worlds/hull_viz.wbt`);SAM(A)看 `data/eval/sam_only/<scene>/<view>/overlay.png`。

---

## 3. 關係半邊的動機與前置缺口

管線後半:instance → 物件關係(`on` 支撐 / `blocks_access` 視覺遮擋)→ 規劃。核心問題:
**現有結果是否影響「以 GNN 做關係推理」的必要性/可行性?**

前置發現:既有 captures(n3/n4/n5 隨機桌面)的 GT 關係 **`on`=0、`blocks`=112/183 場(稀疏)** → 關係半邊**無訊號**,GNN 無從評估。
→ 真正前置缺口 = **缺乏關係豐富場景**。

---

## 4. 實驗:生成關係豐富場景並重拍(路線甲)

### 設計
- `srp/scene_gen/gen_relation_scenes.py`:
  - **堆疊(產 on)**:真平頂白名單底物 + 上物置中、footprint 落底頂內、z=底頂+上半高(幾何穩定)。
  - **遮擋(產 blocks)**:緊密群聚(最小間距 ~0.095 m)。
  - stack3/4/5、occ3/4/5 各 20,共 **120 場**;注入 12 共用視角。
- **可重現性**(物理 settle):variant supervisor `ycb_supervisor_relation_capture`——spawn 尊重 plan 的 z(支援堆疊;原版強制 z=半高)、settle 後穩定性檢查、分開輸出 `multi_{stack,occ}{N}/`。
- 下游 `run_relation_downstream.sh`:SAM→GT modal→GT amodal→GT 關係→hull/instance,場景名驅動、自動分開。

### 結果(120 場)
| 組 | on | blocks |
|---|---|---|
| stack3/4/5(各20) | 各 20(共 **60**) | 23/43/45 |
| occ3/4/5(各20) | 0(正確) | 17/24/49 |
| **總計** | **60** | **201** |

對比舊隨機場景(on=0/blocks=112):**on 0→60、blocks 112→201、104/120 場有關係** → 關係半邊有訊號了。

### GT 關係生成(`gt_relations.py`,皆 GT、免深度)
- `on`:GT mesh 頂點 → footprint 重疊 + 垂直接觸 + X 在上。
- `blocks_access`:GT amodal − modal = 被遮區域 → 找蓋住它的物體 = 遮擋者。
- (plan「REGRAD 物理」的幾何+遮罩近似。)

### 4.1 各關係的幾何計算參數

| 關係 | 參數 | 值 | 意義 |
|---|---|---|---|
| **on**(支撐) | PEN | 0.015 m | 接觸穿透容差(`bot(X)−top(Y) ≥ −PEN`) |
| | GAP | 0.03 m | 接觸最大間隙(`bot(X)−top(Y) ≤ GAP`) |
| | ON_XY | 0.30 | footprint(xy-AABB)重疊 / X 面積 ≥ 此 |
| | above | — | 質心 `z(X) > z(Y)` |
| **blocks_access**(遮擋) | OCC_MIN | 0.10 | 物 i 被遮比例 ≥ 此才算被遮 |
| | OCCLUDER_MIN | 0.30 | 遮擋者 j 蓋住被遮區 ≥ 此(GT 用) |
| | MIN_VIEWS | 2 | 需在 ≥ 此視角數成立 |
| | DS | 4 | 預測 z-buffer 投影降採樣倍率 |
| **前後左右**(方向) | DIR_THR | 0.03 m | 質心差死區(< 此不產生該軸關係) |
| | 軸定義 | — | 前後 = 世界 x 軸、左右 = 世界 y 軸 |

幾何來源:on 的 footprint = xy-AABB、z 由 mesh 頂點(GT)或 voxel 佔據(hull);方向用質心(mesh 頂點均值 / voxel 中心均值);
GT blocks 用 amodal−modal 遮罩,預測 blocks 用 hull z-buffer。on/blocks 的 GT(`gt_relations.py`)與預測規則(`a1_rule.py`)參數一致。

---

## 5. 實驗:A1 probing — 裁決 GNN 必要性

### 設計(`a1_rule.py`)
幾何規則復現 GT 關係,**三元組 (type,x,y) 精確配對**評估(主受體+類型全對才 TP)。
- **mesh**(GT)= 上界/答案(on 為定義性、blocks 非定義性);**hull**(重建)= 真實。

### 結果(120 場)
| 幾何 | on F1 | blocks F1 |
|---|---|---|
| mesh(上界) | 1.00 | **0.93**(P0.92/R0.94) |
| **hull(真實)** | **0.00**(0對/10假陽/60漏) | **0.61**(P0.74/R0.52) |

### 推論
1. **關係在乾淨幾何上規則可解**(mesh-blocks 0.93 為非定義性證據)→ **關係推理層不需學習,GNN 無發揮空間**。
2. **牆在感知層**:`on` 兩物必然相觸 → hull 融成一個 instance(stack3 GT3→2inst、stack5 GT5→3inst)→ 無兩節點 → 規則與 GNN 都產不出 on。
3. `blocks` hull 上部分可恢復(0.61),融合/缺失壓低 recall。

→ **GNN 作為關係推理核心難成立**;學習若有價值,在「**實例分離**」。

### 5.1 各關係幾何判斷的找到率(recall,`rel_recall.py`)
範圍:**303 場景**(n3/n4/n5 + stack3/4/5 + occ3/4/5,即 sam_only 除 n1 外);全用 **hull**(免深度);
方向用質心(GT mesh 質心 vs hull instance 質心,死區 3 cm),on/block 用規則作用 hull;三元組 (type,主體,受體) 精確配對。
指標 = **recall = 找到數 / GT 存在數**;另加 **recall|雙方找到**(只算兩端物體都被 hull 配對到的關係 → 隔開「物體沒找到」拆出純關係判斷品質)。

| 關係 | GT | 預測 | 找到 | recall | precision | F1 | **recall \| 雙方找到** |
|---|---|---|---|---|---|---|---|
| 前後左右(方向) | 6696 | 4638 | 4598 | 0.687 | **0.991** | 0.811 | **0.992** |
| on(支撐) | 60 | 0 | 0 | 0.000 | — | — | **0.000** |
| blocks_access(遮擋) | 313 | 190 | 174 | 0.556 | **0.916** | 0.692 | **0.866** |

- **高 precision、低 recall**:方向 prec 0.991、遮擋 prec 0.916 → **管線敢輸出的關係幾乎都對**(關係層級幻影極少);瓶頸純在 recall(物體沒找到)。
- **方向**:雙方都找到時 recall **0.992(近乎完美)** → 全 recall 0.687 缺口**全來自物體沒找到**(實例缺失/融合),方向判斷本身幾乎不出錯。
- **遮擋**:雙方都找到時 **0.866**(vs 全 0.556)→ 遮擋者與被遮者都被重建時,幾何 z-buffer 約 87% 抓得到。
- **on**:**預測數=0**——配對到 GT 名的 instance 之間規則一個 on 都判不出(56/60 融成單一 instance 湊不齊兩物;少數分開的 4 個切面幾何不準也不過)→ recall/precision 皆 0/未定義。
- 拆解結論:**關係判斷高精度、低召回**(輸出可信、漏很多);方向/遮擋的限制在感知層(實例找到),on 則是分離+切面幾何雙重失敗。

---

## 6. 實驗:免深度相觸/支撐物體分離(決定 on)

> 一段反覆修正的調查:「不可分」→「可分」→「現實受遮罩品質上限」。

- **6.1 SAM 2D 是否分得開**(`diag_stack_sep.py`):60 場/717 視角 **分開 84%、併 1%、漏 15%**(多為底物被遮)→ 2D 物件層級訊號在。
- **6.2 為何 associate 仍融合**(`diag_gate.py`,唯讀):class-agnostic SAM(id 亂)+ union-find **傳遞合併**,接觸面一個橋接焊死整塊;可見性閘控**無效**(跨界一致度 0.9–1.0 不降、死角 0)——因接觸面相鄰上下 voxel(~5mm)投影到同一像素 → 同遮罩。
- **6.3 給乾淨成對遮罩**(`sep_probe.py --src gt`,上界):成對遮罩+可見性+**逐 voxel 絕對多數決** → **voxel 0.886、on 恢復 86%、無票 1.5%**。**修正 6.2**:接觸面模糊佔極少,bulk 多數決判對 → **免深度可分離 on**;associate 失敗是方法假象(class-agnostic + 傳遞合併)。
- **6.4 純幾何配對**(`geo_match.py`):現有 class-agnostic SAM → 遮罩共享-voxel 圖 → **normalized-cut(Fiedler)切 2 群**(從弱接縫切,勝 union-find)→ voxel 投票 → **voxel 0.80、on 恢復 41%**。診斷(`diag_geo_fail.py`):SAM 過分割(17.5 節點/2 物)、純度 0.81、上下不平衡。
- **6.5 換語意預測遮罩**(`sep_probe.py --src gsam`,grounded_sam):**voxel 0.58、on 恢復 5%、無票 31%**——GroundingDINO 文字框太鬆/抓錯,遮罩常**遠大於真實**(tuna 9×、wood_blocks 536× GT 面積)→ 投票亂。

### 三方對照(分離 on)
| 遮罩來源 | voxel | **on 恢復** | 性質 |
|---|---|---|---|
| GT per-object(上界) | 0.886 | **86%** | 完美遮罩 |
| **class-agnostic SAM + 幾何配對** | 0.80 | **41%** | **最佳真實做法** |
| grounded_sam 預測(語意) | 0.58 | **5%** | 文字框太鬆 → 最差 |

### 推論
- **瓶頸 = per-object 遮罩/分離品質**(GT 86%→真實 ≤41%)。
- **語意提示更糟 → 反證 class-agnostic SAM + 幾何配對才是對的方向**。
- 41%→86% 落差 = 把相觸分離做更好;**學習(MLP)可發揮處,價值在『分離』非『關係推理』**。

---

## 7. 總結論
1. **感知層**:免深度 SAM+voxel hull 站得住(基線 mIoU 0.91、prec 0.93);限制是**陰影過估計**(視角窄,參數調不掉)與 **touching 欠分割**(soft carving 部分救回,有取捨)。
2. **關係規則本身夠用,但前提是有正確 instance**:乾淨幾何(mesh)上 blocks F1=0.93(on 為定義性 1.0,循環不算證據)→ 關係**推理層**不需學習(GNN 無發揮空間)。**此為上界、需正確 instance,非真實管線能力。**
3. **真正瓶頸與貢獻 = 免深度實例分離(尤其相觸/支撐)**:真實 hull 上 on F1=**0**(融合)、blocks F1=**0.61**(噪声/缺失),皆因 instance 錯/缺,非規則錯。
4. **on 的分離只在「有乾淨成對遮罩」時可行**(上界 86%);真實 class-agnostic 幾何配對僅 41%、語意提示更差(5%);**真實管線 on 關係 recall≈0**。→ **規則好 ≠ 管線好**;免深度下 on/blocks 仍無法乾淨取得,落差全在實例分離。

## 8. 限制與未來工作
- **分離**:現實 41% vs 上界 86%,落差來自相觸物 per-object 分離;接觸面薄層、被遮底物為死角。未做:(a) 改進幾何配對;(b) **學習式分離**(小模型吃 hull+成對遮罩+相機射線,推 41%→86%,本路線真正技術貢獻);(c) blocks 上推;(d) 分離後接關係並量端到端。
- **感知**:陰影固有過估計;touching recall 天花板;soft carving 取捨;**低仰角(20°)視角覆蓋偏一側**(見 §0.1,候選範圍+貪婪選取所致,非可達性),加劇覆蓋不均。
- GT 關係為幾何+遮罩近似(非 REGRAD 物理)。

---

## 附錄 A:工具
| 階段 | 檔案 |
|---|---|
| Stage 1 雕殼 | `srp/stage1_hull/{carve,run_scene,test_carve,visualize_hull}.py` |
| Stage 2 關聯 | `srp/stage2_instances/associate.py` |
| 評估/可視化 | `srp/stage2_instances/{eval,eval_mesh,eval_reproj2d,viz_reproj,gen_viz_objs}.py`;`sweep_am_cover.sh` |
| 關係場景 | `srp/scene_gen/gen_relation_scenes.py`;`controllers/ycb_supervisor_relation_capture/`;`worlds/ycb_relation_capture.wbt`;`run_relation_downstream.sh` |
| GT 關係 | `srp/stage3_graph/gt_relations.py` |
| A1/分離 | `srp/stage4_probe/{a1_rule,diag_stack_sep,diag_gate,sep_probe,geo_match,diag_geo_fail}.py` |

## 附錄 B:資料
- 場景計畫 `data/scene_plans/{multi,stack,occ}_scene_plan.json`
- 拍攝 `data/captures/multi_{n,stack,occ}{N}/`
- 標註/關係 `data/labels/<scene>/{actual,amodal,relations.json}`
- hull/instance `data/eval/srp_hull/<scene>/`、掃描 `data/eval/srp_sweep/`、診斷 `data/eval/_diag/`

## 附錄 C:重現關鍵指令
```bash
# 感知層
./srp/stage1_hull/run_scene.py <scenes>     # SAM → hull
./srp/stage2_instances/associate.py <scenes> # hull → instance
./srp/sweep_am_cover.sh 3                     # 參數掃描
# 關係層
./srp/scene_gen/gen_relation_scenes.py --per 20
RELATION_ARGS="stack" webots worlds/ycb_relation_capture.wbt   # occ 同理
bash srp/scene_gen/run_relation_downstream.sh all
./srp/stage4_probe/a1_rule.py                # A1 規則 probing
./srp/stage4_probe/sep_probe.py              # 分離上界(86%)
./srp/stage4_probe/geo_match.py              # 純幾何分離(41%)
./srp/stage4_probe/sep_probe.py --src gsam   # 語意預測遮罩(5%)
```

## 附錄 D:執行成本(一組場景 = 12 視角)

| 階段 | 時間/場 | 依據 |
|---|---|---|
| 拍攝(Webots,12 視角) | **~55 s** | scene_manifest 時間戳實測(55–56s,穩定) |
| 分割(SAM class-agnostic,12 張) | **~50 s** | 實測 ~4 s/張 × 12(模型一次載入攤提) |

**SAM GPU vs CPU 實測(n5_scene0001,5 物,12 張,vit_b)**:GPU(RTX 4070 Ti)48.6 s(~4.0 s/張)、CPU 443.2 s=7m23s(~36.9 s/張)→ **GPU 快 ~9.1×**。倍數未更大,因 vit_b 編碼器較小 + 自動遮罩的 NMS/RLE 後處理在 CPU(兩邊共同地板)。換算 120 場:GPU ≈ 1.6 h、CPU ≈ 14.8 h。
| 建 hull(run_scene + associate,GPU) | **~3–5 s** | run_scene 1.6s + associate 1.6s |
| **合計** | **~110 s ≈ 1.8 分鐘/場** | |

- 瓶頸 = 拍攝(~55s)+ 分割(~50s);**建 hull 幾乎免費**(GPU)。
- 換算:120 場 ≈ **~3.7 小時**(序列;各階段可分批,SAM 權重一次載入)。
- 改用 grounded_sam(文字框)分割會更慢(GroundingDINO+SAM 每張)。
- 硬體:RTX 4070 Ti。可視化世界:`worlds/experiment_viz.wbt`(手臂停於圓球頂點 view_04,controller `arm_apex`)。

## 附錄 E:關係定義與計算(完整規格見 `srp/stage3_graph/GT_RELATIONS_SPEC.md`)

**幾何來源(每物/每 instance)**:GT 用 mesh 世界座標頂點 `V=(verts−ycb_center)·Rᵀ+pos`;hull 用佔據體素中心 `c=grid_min+(idx+0.5)·voxel`。由點集 P 得:footprint xy-AABB(min/max x,y)、`top=maxP.z`、`bot=minP.z`、`cenz=meanP.z`、`area=(Δx)(Δy)`、質心 xy。

**on(X,Y)** = 三條件全成立:① `−PEN≤bot(X)−top(Y)≤GAP`;② `xy_overlap(X,Y)/area(X)≥ON_XY`(xy-AABB 交集面積);③ `cenz(X)>cenz(Y)`。

**blocks_access(X,Y)**:
- GT(遮罩):每視角 `hidden_i=amodal_i∧¬modal_i`;`occ_frac=|hidden_i|/|amodal_i|≥OCC_MIN`;遮擋者 `j*=argmax_j|hidden_i∧modal_j|/|hidden_i|≥OCCLUDER_MIN`;`≥MIN_VIEWS` 視角成立。
- 預測(hull z-buffer):每視角投影(降採樣 DS)取每像素最小深度;`front=(Y)∧(X)∧depth(X)<depth(Y)`;`occ_frac=|front|/|Y|≥OCC_MIN`;`≥MIN_VIEWS`。

**前後左右**:用質心 (x,y),`right:A.y−B.y>DIR_THR`、`left:反`、`front:A.x−B.x>DIR_THR`、`back:反`;前後=世界 x、左右=世界 y。

(參數值見 §4.1;單位/出處/變更紀錄見規格文件。)

## 附錄 F:各 recall 的計算方式

**§2 D1 recall(實例 vs GT,`eval.py`)**
每場景把預測 instance ↔ GT 物體(GT 視覺 hull = amodal 遮罩雕成)以 **3D IoU 匈牙利配對**;`found` = 配對成功且 IoU ≥ 0.25 的 GT 物數;**`recall = found / n_gt`**(n_gt = GT 物數)。表中各組數值為**場景平均**。precision = found / n_pred、mIoU = 配對成功者的平均 3D IoU。

**§5 A1 關係 F1/recall(`a1_rule.py`)**
關係以**三元組 (type, x, y) 精確配對**(主體+受體+類型全對):`TP` = 預測關係 ∈ GT 關係集、`FN` = GT 關係 ∉ 預測、`FP` = 預測 ∉ GT。**`recall = TP/(TP+FN)`**、precision = TP/(TP+FP)、F1 = 兩者調和平均。mesh/hull 兩種幾何各自算。

**§5.1 關係 recall/precision/F1(`rel_recall.py`)**
- `找到`(=TP)= GT 關係三元組也被預測(規則作用在 **hull instance**);`GT存在` = 全部 GT 關係;`預測` = 規則在 hull 上產出的關係數。
- **`recall = 找到 / GT存在`**(存在的關係被找回多少)。
- **`precision = 找到 / 預測`**(預測出來的關係有多少是對的;預測=0 時未定義,如 on)。
- **`F1`** = recall 與 precision 的調和平均。
- **`recall | 雙方找到 = 找到 / GT(雙方找到)`**:`GT(雙方找到)` = **兩端物體都配對到 hull instance** 的 GT 關係(隔開「物體沒找到」的拖累,純看關係判斷)。
- 來源:方向 GT 用 GT 實心 mesh 質心、on/block GT 取 `relations.json`;預測一律用 hull instance(3D IoU 配對到 GT 名)。

> 註:§2 recall 是「**物體**找到率」(實例 vs GT);§5/§5.1 recall 是「**關係**找到率」(關係三元組);兩者層級不同。

## 附錄 G:各物體端到端失敗率(全 62 種,多物場景 n3/n4/n5/stack/occ,`per_obj_found.py`)

找到 = 該物被某 instance 覆蓋 ≥0.5 的場景比例;失敗率 = 1−找到率。只在 n1 出現的物體(如 059_chain)未列。

| 物體 | 出現 | 失敗率 | | 物體 | 出現 | 失敗率 |
|---|---|---|---|---|---|---|
| 028_skillet_lid | 12 | 1.000 | | 005_tomato_soup_can | 34 | 0.088 |
| 070-a_colored_wood_blocks | 25 | 1.000 | | 008_pudding_box | 23 | 0.087 |
| 062_dice | 17 | 0.941 | | 007_tuna_fish_can | 24 | 0.083 |
| 031_spoon | 14 | 0.929 | | 004_sugar_box | 26 | 0.077 |
| 037_scissors | 13 | 0.923 | | 036_wood_block | 27 | 0.074 |
| 035_power_drill | 12 | 0.833 | | 065-c_cups | 16 | 0.062 |
| 032_knife | 15 | 0.733 | | 065-f_cups | 18 | 0.056 |
| 030_fork | 17 | 0.706 | | 006_mustard_bottle | 19 | 0.053 |
| 040_large_marker | 15 | 0.600 | | 065-d_cups | 19 | 0.053 |
| 043_phillips_screwdriver | 17 | 0.588 | | 022_windex_bottle | 19 | 0.053 |
| 050_medium_clamp | 17 | 0.529 | | 065-h_cups | 20 | 0.050 |
| 044_flat_screwdriver | 16 | 0.500 | | 003_cracker_box | 22 | 0.045 |
| 058_golf_ball | 19 | 0.421 | | 065-e_cups | 22 | 0.045 |
| 012_strawberry | 23 | 0.391 | | 010_potted_meat_can | 24 | 0.042 |
| 070-b_colored_wood_blocks | 49 | 0.367 | | 015_peach | 25 | 0.040 |
| 038_padlock | 22 | 0.273 | | 061_foam_brick | 27 | 0.037 |
| 042_adjustable_wrench | 15 | 0.267 | | 009_gelatin_box | 29 | 0.034 |
| 057_racquetball | 17 | 0.235 | | 024_bowl / 053_mini_soccer_ball | 16/13 | 0.000 |
| 065-b_cups | 15 | 0.200 | | 026_sponge / 013_apple | 26/20 | 0.000 |
| 021_bleach_cleanser | 25 | 0.200 | | 056_tennis_ball / 002_master_chef_can | 22/28 | 0.000 |
| 014_lemon | 17 | 0.176 | | 011_banana / 055_baseball | 18/19 | 0.000 |
| 018_plum | 19 | 0.158 | | 017_orange / 016_pear | 21/24 | 0.000 |
| 065-a_cups | 19 | 0.158 | | 025_mug / 019_pitcher_base | 19/16 | 0.000 |
| 065-i_cups | 16 | 0.125 | | 052_extra_large_clamp / 065-j_cups | 13/17 | 0.000 |
| 077_rubiks_cube | 19 | 0.105 | | 071_nine_hole_peg_test / 051_large_clamp | 13/14 | 0.000 |
| | | | | 029_plate / 065-g_cups / 054_softball / 033_spatula* | 12/20/20/2 | 0.000 |

*033_spatula n=2,樣本不足。**11 種失敗率 >0.5(全薄/扁/小/多塊/不規則),~40 種 ≤0.1**(規則塊狀/球/罐/杯/水果重建良好)。

## 附錄 H:物體 × 場景組 失敗率矩陣(`per_obj_matrix.py`,CSV: `data/eval/_diag/per_obj_matrix.csv`)

列=物體(按總計失敗率排序),欄=各組失敗率(`-`=該組未出現),末欄=跨組總計、n=總出現場景數。各組每物樣本少(多為 3–9 場),單格較粗。

| 物體 | n3 | n4 | n5 | occ3 | occ4 | occ5 | stk3 | stk4 | stk5 | 總計 | n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 028_skillet_lid | 1.00 | 1.00 | 1.00 | - | - | - | - | - | - | **1.000** | 12 |
| 070-a_colored_wood_blocks | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.000** | 25 |
| 062_dice | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | - | - | - | **0.941** | 17 |
| 031_spoon | 0.67 | 1.00 | 1.00 | - | - | 1.00 | - | 1.00 | - | **0.929** | 14 |
| 037_scissors | 1.00 | 1.00 | 0.80 | 1.00 | - | - | - | - | - | **0.923** | 13 |
| 035_power_drill | 0.33 | 1.00 | 1.00 | - | - | - | - | - | - | **0.833** | 12 |
| 032_knife | 0.67 | 1.00 | 0.60 | - | - | - | - | 0.50 | 1.00 | **0.733** | 15 |
| 030_fork | 0.33 | 0.75 | 0.80 | - | - | 0.00 | - | 1.00 | 1.00 | **0.706** | 17 |
| 040_large_marker | 0.33 | 1.00 | 0.40 | 1.00 | 0.00 | 1.00 | - | - | - | **0.600** | 15 |
| 043_phillips_screwdriver | 1.00 | 0.25 | 0.60 | - | - | 0.50 | 1.00 | - | 0.50 | **0.588** | 17 |
| 050_medium_clamp | 0.33 | 0.25 | 0.80 | - | - | 0.67 | - | 1.00 | 0.00 | **0.529** | 17 |
| 044_flat_screwdriver | 1.00 | 0.50 | 0.40 | 0.00 | 0.00 | - | 0.00 | - | 1.00 | **0.500** | 16 |
| 058_golf_ball | 0.33 | 0.50 | 0.60 | 0.50 | 1.00 | - | - | 0.00 | 0.00 | **0.421** | 19 |
| 012_strawberry | 0.00 | 0.25 | 0.40 | 0.50 | 0.33 | 0.67 | - | 1.00 | 0.50 | **0.391** | 23 |
| 070-b_colored_wood_blocks | 0.00 | 1.00 | 0.60 | 1.00 | 0.75 | 1.00 | 0.00 | 0.12 | 0.30 | **0.367** | 49 |
| 038_padlock | 0.00 | 0.25 | 0.20 | 0.00 | 0.00 | - | - | 1.00 | 1.00 | **0.273** | 22 |
| 042_adjustable_wrench | 0.33 | 0.25 | 0.20 | - | - | 0.00 | 0.00 | 1.00 | - | **0.267** | 15 |
| 057_racquetball | 0.33 | 0.00 | 0.60 | - | 0.00 | 0.00 | 0.00 | 0.00 | - | **0.235** | 17 |
| 065-b_cups | 0.00 | 0.50 | 0.00 | 0.00 | 0.00 | - | - | 1.00 | - | **0.200** | 15 |
| 021_bleach_cleanser | 0.00 | 0.25 | 0.20 | 0.50 | 0.00 | 0.20 | 0.00 | 1.00 | 0.00 | **0.200** | 25 |
| 014_lemon | 0.00 | 0.75 | 0.00 | - | 0.00 | 0.00 | - | 0.00 | - | **0.176** | 17 |
| 018_plum | 0.00 | 0.25 | 0.20 | 0.00 | 0.00 | - | - | - | 0.50 | **0.158** | 19 |
| 065-a_cups | 0.00 | 0.25 | 0.20 | - | 0.00 | 0.33 | - | 0.00 | 0.00 | **0.158** | 19 |
| 065-i_cups | 0.33 | 0.00 | 0.20 | - | 0.00 | - | 0.00 | 0.00 | - | **0.125** | 16 |
| 077_rubiks_cube | 0.00 | 0.00 | 0.40 | - | 0.00 | 0.00 | - | 0.00 | 0.00 | **0.105** | 19 |
| 005_tomato_soup_can | 0.00 | 0.25 | 0.20 | 0.00 | 0.00 | 0.00 | 0.00 | 0.14 | 0.00 | **0.088** | 34 |
| 008_pudding_box | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.50 | **0.087** | 23 |
| 007_tuna_fish_can | 0.00 | 0.00 | 0.20 | 0.00 | - | - | 0.00 | 0.33 | 0.00 | **0.083** | 24 |
| 004_sugar_box | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | - | 0.00 | 0.00 | 0.29 | **0.077** | 26 |
| 036_wood_block | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.33 | 0.25 | **0.074** | 27 |
| 065-c_cups | 0.00 | 0.25 | 0.00 | - | 0.00 | 0.00 | - | - | 0.00 | **0.062** | 16 |
| 065-f_cups | 0.00 | 0.00 | 0.20 | 0.00 | 0.00 | 0.00 | 0.00 | - | 0.00 | **0.056** | 18 |
| 006_mustard_bottle | 0.00 | 0.00 | 0.20 | 0.00 | 0.00 | - | - | 0.00 | 0.00 | **0.053** | 19 |
| 065-d_cups | 0.00 | 0.00 | 0.20 | - | 0.00 | 0.00 | 0.00 | - | 0.00 | **0.053** | 19 |
| 022_windex_bottle | 0.00 | 0.25 | 0.00 | - | - | 0.00 | - | 0.00 | 0.00 | **0.053** | 19 |
| 065-h_cups | 0.00 | 0.25 | 0.00 | 0.00 | 0.00 | 0.00 | - | 0.00 | 0.00 | **0.050** | 20 |
| 003_cracker_box | 0.00 | 0.00 | 0.00 | 0.00 | - | 0.00 | 0.00 | 0.33 | 0.00 | **0.045** | 22 |
| 065-e_cups | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | **0.045** | 22 |
| 010_potted_meat_can | 0.00 | 0.00 | 0.00 | - | - | 0.00 | 0.00 | 0.00 | 0.33 | **0.042** | 24 |
| 015_peach | 0.33 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.040** | 25 |
| 061_foam_brick | 0.00 | 0.00 | 0.20 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.037** | 27 |
| 009_gelatin_box | 0.00 | 0.00 | 0.20 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.034** | 29 |
| 024_bowl | 0.00 | 0.00 | 0.00 | - | 0.00 | 0.00 | - | - | 0.00 | **0.000** | 16 |
| 053_mini_soccer_ball | 0.00 | 0.00 | 0.00 | 0.00 | - | - | - | - | - | **0.000** | 13 |
| 026_sponge | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.000** | 26 |
| 013_apple | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | - | - | 0.00 | 0.00 | **0.000** | 20 |
| 056_tennis_ball | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | - | 0.00 | **0.000** | 22 |
| 002_master_chef_can | 0.00 | 0.00 | 0.00 | - | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.000** | 28 |
| 011_banana | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | - | - | 0.00 | - | **0.000** | 18 |
| 055_baseball | 0.00 | 0.00 | 0.00 | 0.00 | - | 0.00 | 0.00 | - | 0.00 | **0.000** | 19 |
| 017_orange | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.000** | 21 |
| 016_pear | 0.00 | 0.00 | 0.00 | - | 0.00 | 0.00 | - | 0.00 | 0.00 | **0.000** | 24 |
| 025_mug | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | - | - | **0.000** | 19 |
| 019_pitcher_base | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | - | - | 0.00 | 0.00 | **0.000** | 16 |
| 052_extra_large_clamp | 0.00 | 0.00 | 0.00 | - | - | - | - | - | 0.00 | **0.000** | 13 |
| 065-j_cups | 0.00 | 0.00 | 0.00 | - | 0.00 | 0.00 | - | - | - | **0.000** | 17 |
| 071_nine_hole_peg_test | 0.00 | 0.00 | 0.00 | - | - | 0.00 | - | - | - | **0.000** | 13 |
| 051_large_clamp | 0.00 | 0.00 | 0.00 | 0.00 | - | - | - | - | 0.00 | **0.000** | 14 |
| 029_plate | 0.00 | 0.00 | 0.00 | - | - | - | - | - | - | **0.000** | 12 |
| 065-g_cups | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | - | 0.00 | - | **0.000** | 20 |
| 054_softball | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | - | - | 0.00 | **0.000** | 20 |
| 033_spatula | - | - | - | - | 0.00 | - | - | - | - | **0.000** | 2 |
