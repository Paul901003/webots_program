# 12 視角實例分離方法 — 程式 / 功能 / 邏輯報告

> 對象:在同一組 **12 視角**(A-3 `selected_view_names(12)`)、同一基礎 visual hull(`srp_hull_v12`)上，
> 把「一坨 hull」切成「各物體 instance」的方法。全部輸入只用 **SAM 遮罩 + 相機位姿**，不用深度、不用 GT。
> 五個可在 `hull_viz.wbt` 直接可視化的 root：
> `srp_hull_cg`、`srp_hull_cg_solid`、`srp_hull_semvote`、`srp_hull_semcluster`、`srp_hull_sempaper`。

---

## 0. 在管線中的位置

```
階段一(拍攝) ─► SAM 遮罩(sam_only_fast) + 相機位姿(captures_fast)
                          │
              ┌───────────┴─────────────┐
              ▼                          ▼
   Stage1: run_scene.py           手臂剪影 srp_arm_masks
   → srp_hull_v12/<scene>/hull.npz  (前景聯集雕殼，class-agnostic，不分物體)
              │
              ▼  ← 本報告的五個方法都吃這顆 hull 當基礎網格
   Stage2 實例分離(四種邏輯)
     ├─ cg_associate.py      → srp_hull_cg        (表面 voxel labels)
     │      └ fill_solid.py  → srp_hull_cg_solid  (填實心，供公平評估/可視化)
     ├─ voxel_sem_vote.py    → srp_hull_semvote   (命名法)
     ├─ voxel_sem_cluster.py → srp_hull_semcluster(分群法)
     └─ voxel_sem_paper.py   → srp_hull_sempaper  (論文法)
              │
              ▼
   評估 eval_surface.py(表面空間) / eval.py(填實後 3D-IoU)
   可視化 gen_viz_objs.py → hull_viz.wbt
```

輸出格式統一為 `data/eval/<root>/<scene>/instances.npz`，含三個 key：
`labels`(3D 整數網格，0=背景，k=第 k 個 instance)、`grid_min`、`voxel_size`。
`hull_viz` 對每個 label 做 marching cubes → 上色半透明 mesh。

---

## 1. 共同基礎：Stage1 visual hull（`srp/stage1_hull/run_scene.py` → `carve.py`）

所有方法都建在這顆 hull 上，先講它。

- **輸入**：每視角前景 = 該視角所有「非地板」SAM 遮罩的**聯集**，再**減掉手臂 FK 剪影**（`srp_arm_masks`）。
- **雕殼**（`carve_visual_hull`）：工作空間 AABB `[0,0.7]×[−0.35,0.35]×[0,0.35]`，體素 0.005 m，封底 `table_z=0`。
  每 voxel 投影到各視角，落在前景內就記一票；`occupancy = 票數 ≥ (V − allow_miss)`。
  **`allow_miss=0`（硬交集，實測最佳）**：所有視角都要看到才保留，最保守。
- **額外輸出** `observed`：voxel 是否曾進過任一相機視錐（供遮擋/場景分析用）。
- **`surface` key**（`add_surface_mask.py` 後補）：occupied 且至少一個 6-鄰居為空 = 表面 voxel。cg 用它當「點雲」。

> 重點：**Stage1 完全不分物體**。它只給一坨連通（或黏連）的佔據網格；「分成幾個物體」全交給 Stage2。

---

## 2. 共同投影機制（四個 Stage2 方法都用）

每個方法核心都是「voxel ↔ 影像遮罩」的投影，機制一致：

```python
X = P @ Rwc.T + t            # 世界 voxel 中心 P → 相機座標
u = fx*X0/zc + cx ; v = fy*X1/zc + cy   # 投影到像素
inb = (zc>0) & 在影像範圍內
落點像素若在某遮罩內 → 該 voxel 關聯到該遮罩(→ 其語意/群標籤)
排除手臂:落點在 arm mask 內 → 跳過
```

差別只在**「怎麼從遮罩關聯決定 voxel 的 instance」**。以下逐一說明。

---

## 3. 方法 A — ConceptGraphs 式關聯 `cg_associate.py`（`srp_hull_cg`）

**核心思想**：物體不預先切，而是從跨視角**增量關聯**中「湧現」。仿 ConceptGraphs，但用 hull **表面 voxel** 取代其 depth 點雲（免深度）。

**逐視角流程**（走 12 視角）：
1. **z-buffer 可見性**（`zbuffer_visible`）：所有表面 voxel 投影，每像素只留**最近**的 voxel → 模擬「這一視角實際看得到的表面」。
2. **掏合體遮罩**（`mask_subtract_contained`，th1=0.8/th2=0.7，同 ConceptGraphs）：若大遮罩幾乎包住某小遮罩，就把小的從大的挖掉，避免「合體 mask」污染關聯。
3. 每張 SAM mask → 框到的可見 voxel = 一個 **detection**；`largest_cc` 只留最大 3D 連通坨（濾掉合體 mask 跨物體的橋接雜點）；voxel 數 < `MIN_VOX_DET=15` 丟棄。
4. **純空間關聯**：detection 的 voxel 集合與既有物體算 `交集/detection`，最高者若 `> SPATIAL_MERGE=0.10` → 併入；否則新建物體。
   - **CLIP 特徵有累積但不參與合併判定**——因為同類物體（如兩個 cup）特徵幾乎一樣，用視覺會被錯併；只用空間重疊區分。

**收尾**：`merge_objects`（物體間 voxel 交集/較小者 `> MERGE_THR=0.5` 去重）→ 丟 `< MIN_OBJ_VOX=40` 小碎片 → 大物體給小 id → 輸出**只含表面 voxel** 的 labels。

**關鍵參數**：`SPATIAL_MERGE=0.10`、`MIN_VOX_DET=15`、`MERGE_THR=0.5`、`MIN_OBJ_VOX=40`。

**輸出型態**：**表面殼**（非實心）。→ 需 `fill_solid.py` 補實心版 `srp_hull_cg_solid` 才能與其他方法做同基準比較 / 直接可視化。

**特性**：唯一「不靠語意、純幾何關聯」的方法；對**同類相鄰**（兩個 cup）靠空間分離有效；缺點是同物體被拆成多片時會過切（precision 偏低）。

---

## 4. 方法 B — 命名投票法 `voxel_sem_vote.py`（`srp_hull_semvote`）

**核心思想**：每張遮罩先用 CLIP 對 64 個 YCB 名詞取 **top-1 phrase**（直接命名），voxel 依「投影落到的遮罩的 phrase」加權投票，**同 phrase 的 3D 連通分量 = 各 instance**。

**流程**：
1. 每視角每遮罩 → `cached_feats`（預存 CLIP 影像特徵，square-mean-crop→CLIP→去偏→L2）→ 對 `clip_text_feats.npz` 的 64 名詞算 cos → **top-1 phrase**。
2. 每 occupied voxel 投影各視角，落在哪張遮罩就投給該遮罩的 phrase，**權重 = 1/相機距離**（近相機票重）。
3. 加權多數決 → 每 voxel 一個 phrase。
4. 對每個 phrase 的 voxel 取 **3D 連通分量**（26-鄰）→ 每個 `≥ MIN_VOX=50` 的分量 = 一個 instance。

**關鍵參數**：`MIN_VOX=50`；語意來源 = 64 名詞 top-1。

**特性**：靠「命名」分物體，**不同類**物體天然分開；同 phrase 但空間分離者靠連通分量再切。缺點：CLIP 命名錯 → 該物體標籤錯；**同類**（都命名成同 phrase 且相鄰）分不開。

---

## 5. 方法 C — 特徵分群法 `voxel_sem_cluster.py`（`srp_hull_semcluster`）

**核心思想**：不靠固定 64 名詞，而是把**全場景所有遮罩的 CLIP 去偏特徵**做**凝聚式階層分群**，群 id 當語意標籤，再投票 + 連通。

**流程**：
1. 蒐集全 12 視角所有遮罩的 CLIP 特徵 → `debias`（扣平均）→ `F`。
2. `linkage(pdist(F,"cosine"),"average")` + `fcluster(t=sem_thr=0.3, criterion="distance")` → 每張遮罩一個群 id。
3. voxel 投影落到的遮罩 → 該群 id，**1/相機距離**加權投票。
4. 同群 id 的 3D 連通分量 → instance（`≥ MIN_VOX=50`）。

**關鍵參數**：`sem_thr=0.3`（群間 cosine 距離門檻，越小切越多群）、`MIN_VOX=50`。

**特性**：語意標籤是**資料驅動**（不受 64 名詞清單限制），對名詞表外的物體較穩；`sem_thr` 決定粒度。缺點同 B：同類外觀近者會被分到同群 → 相鄰同類分不開。

---

## 6. 方法 D — 論文貼合法 `voxel_sem_paper.py`（`srp_hull_sempaper`）

**核心思想**：最貼近論文的加權投票——加上 **z-buffer 可見性**與**三重權重**，且內部無票 voxel 用最近有票者填充。

**流程**：
1. 每遮罩 CLIP → top-1 phrase **及其信心 cos 值 S**。
2. **z-buffer**：每視角每像素只留最近 voxel（`owner`），只有「可見 voxel」才投票（遮擋的內部 voxel 不投）。
3. **三重權重投票**：`票 = S × (1/中心距離 cw) × (1/相機距離 dw)`
   - `S`：CLIP 命名信心；`cw`：像素離影像中心越近越重（鏡頭中心畸變小）；`dw`：相機越近越重。
4. 每 voxel 取加權最大 phrase；**無票 voxel**（被遮擋看不到的內部）用 `cKDTree` 找最近有票 voxel 借其 phrase 填充。
5. 同 phrase 3D 連通分量 → instance（`≥ MIN_VOX=50`）。

**關鍵參數**：三權重（信心×中心距離×相機距離）、`MIN_VOX=50`。

**特性**：對可見性與觀測品質建模最完整；但權重多、對 CLIP 命名依賴仍在，實測過切較多（見 §8，同場景切出 8 塊）。

---

## 7. 輔助程式

### 7.1 `fill_solid.py` — cg 表面 → 實心
cg 只輸出表面殼。此程式讀 `srp_hull_cg`(表面 labels) + `srp_hull_v12`(occupancy)：
對每個 occupied voxel 用 `distance_transform_edt` 找**最近的有標籤表面 voxel**繼承其 id，
再 `filled[~occ]=0` 只留 hull 實心內 → `srp_hull_cg_solid`。
用途：與其他實心方法在同基準（3D-IoU）比較 / 讓 hull_viz 不必加 `surface` 也能正常渲染。

### 7.2 `gen_viz_objs.py` — 產可視化 obj + manifest（`hull_viz` 用）
讀 `<root>/<scene>/instances.npz` → 每 label marching cubes → 世界座標 .obj（半透明上色）；
另擺真實 YCB mesh（半透明灰）做 GT 對照。`--surface` 旗標把每 instance 挖空只留表面 voxel（cube-per-voxel），供看 cg 原生殼。

### 7.3 `eval_surface.py` / `eval.py` — 評估
- `eval_surface.py`：在**表面 voxel 空間**評 recall / mixed(跨物體) / inst 數，讓「cg 表面」與「四方法實心」公平（都取 ∩surface）。門檻 `COVER_THR=0.3`、`PURITY=0.7`。
- `eval.py`（填實後）：per-object 3D-IoU>0.3、Hungarian 一對一配對 → recall / precision。

---

## 8. 五方法對照與實測

**方法邏輯對照：**

| 方法 | root | 分物體依據 | 語意來源 | 可見性建模 | 輸出型態 |
|------|------|-----------|---------|-----------|---------|
| cg 關聯 | `srp_hull_cg`(+`_solid`) | **純空間**跨視角關聯 | 無(CLIP 只累積不用) | z-buffer(表面) | 表面殼 |
| 命名投票 | `srp_hull_semvote` | 同 phrase + 3D 連通 | 64 名詞 top-1 | 無(全投) | 實心 |
| 特徵分群 | `srp_hull_semcluster` | 同群 id + 3D 連通 | 階層分群(資料驅動) | 無(全投) | 實心 |
| 論文法 | `srp_hull_sempaper` | 同 phrase + 3D 連通 | 64 名詞 top-1 | z-buffer + 三權重 | 實心 |

**先前實測**（填實後 `eval.py`，3D-IoU>0.3、Hungarian 一對一；n=散置 / occ=遮擋 / stack=堆疊）：

| 方法 | recall n | recall occ | recall stack | precision |
|------|:---:|:---:|:---:|:---:|
| cg 關聯 | 0.95 | 0.94 | **0.77** | 0.61 |
| 命名投票 | 0.95 | 0.94 | 0.74 | ~0.78 |
| 特徵分群 | 0.96 | 0.94 | 0.75 | 0.79 |
| 論文法 | 0.95 | 0.94 | 0.74 | 0.67 |
| （對照）associate 幾何法 | 0.89 | 0.88 | 0.67 | 0.96 |

> 讀法（皆為已量測結果，非推論）：
> - **堆疊分離**：cg 最高（0.77），因其純空間關聯不受「同類同命名」限制。
> - **散置準確度**：五方法 recall 都 ≈0.94–0.96，**cg 分堆疊沒有犧牲散置**。
> - **precision**：cg 最低（0.61），因同物體易被拆成多片過切；此問題（合併同物體碎片）**尚未解決**。
> - `associate` 幾何法 precision 最高但 recall（尤其 stack）最低，取向相反。

**同類相鄰的 SAM 上限**（已驗證，決定各法天花板）：平行雙 cup 在 32/34、27/34 視角有分開遮罩（79–94%）；
垂直木塊堆疊 0/34（SAM 完全分不開）→ 純靠 2D 遮罩的三個語意法對垂直堆疊無解，cg 靠 3D 空間關聯才有 0.77。

---

## 9. 執行與可視化指令

```bash
# ── 重算（單場景 / 組號 / 空=全部；預設 --n-views 12）──
./srp/stage2_instances/cg_associate.py     stack4_scene0012
./srp/stage2_instances/fill_solid.py --in-root srp_hull_cg --out-root srp_hull_cg_solid stack4_scene0012
./srp/stage2_instances/voxel_sem_vote.py    stack4_scene0012
./srp/stage2_instances/voxel_sem_cluster.py stack4_scene0012
./srp/stage2_instances/voxel_sem_paper.py   stack4_scene0012

# ── 評估 ──
./srp/stage2_instances/eval_surface.py --roots srp_hull_cg,srp_hull_semvote,srp_hull_semcluster,srp_hull_sempaper

# ── hull_viz 可視化（改 SRP_VIZ_ARGS 後 Ctrl+Shift+R 重載）──
#   語法: SRP_VIZ_ARGS="<scene> <show_gt:1/0> <root> [surface]"
SRP_VIZ_ARGS="stack4_scene0012 1 srp_hull_cg_solid"  webots worlds/hull_viz.wbt
SRP_VIZ_ARGS="stack4_scene0012 1 srp_hull_semvote"   webots worlds/hull_viz.wbt
SRP_VIZ_ARGS="stack4_scene0012 1 srp_hull_semcluster" webots worlds/hull_viz.wbt
SRP_VIZ_ARGS="stack4_scene0012 1 srp_hull_sempaper"  webots worlds/hull_viz.wbt
SRP_VIZ_ARGS="stack4_scene0012 1 srp_hull_cg surface" webots worlds/hull_viz.wbt  # cg 原生表面殼
```

**環境**：全部 stage2 方法走 `webots_visual_hull`(3.10)；env 預設
`SAM_ROOT=sam_only_fast HULL_ROOT=srp_hull_v12 CAPTURES_ROOT=captures_fast ARM_MASK_ROOT=srp_arm_masks`。
