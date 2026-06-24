# 關係定義與計算規格(on / blocks_access / 前後左右)

> **此文件與實作一致**(2026-06 對齊)。權威計算在程式碼:
> `srp/stage3_graph/gt_relations.py`(GT 的 on/blocks)、`srp/stage4_probe/a1_rule.py`(hull 預測規則 on/blocks)、
> `srp/stage4_probe/rel_recall.py`(方向)。座標皆世界座標(機器人底座 [-0.4,0,0]、look-at 中心 [0.35,0,0])。
> 全程免深度(C-DEP):幾何取自 GT mesh 位姿 或 重建 hull 體素;blocks 的 GT 另用渲染遮罩(非感測深度)。

---

## 0. 幾何來源(每物體 / 每 instance)

兩種來源,輸出同一組量:

| 來源 | 點集 | 出處 |
|---|---|---|
| **GT mesh** | `V = (mesh.vertices − ycb_center(name)) @ R(axis,angle)ᵀ + position_m`(世界座標頂點) | `gt_relations.obj_geom` |
| **hull(重建)** | `c = grid_min + (idx + 0.5)·voxel_size`(佔據體素中心),idx=該 instance 的佔據體素索引 | `a1_rule.entity_geom` |

由點集 `P`(=V 或 c)計算:
- **footprint(xy-AABB)**:`xmin/xmax = min/max P[:,0]`、`ymin/ymax = min/max P[:,1]`
- **頂/底 z**:`top = max P[:,2]`、`bot = min P[:,2]`
- **質心 z**:`cenz = mean P[:,2]`;**質心 xy**(方向用):`(mean P[:,0], mean P[:,1])`
- **footprint 面積**:`area = (xmax−xmin)·(ymax−ymin)`
- (方向關係的 GT 質心改用 GT 實心 mesh 佔據體素中心均值,見 `rel_recall.centroid`;hull 用 instance 體素均值。)

---

## 1. `on(X, Y)` — 支撐(X 在上、Y 在下)

三條件**全成立**才判 `on(X,Y)`(`gt_relations.compute_on` / `a1_rule.rule_on`):

1. **垂直接觸**:`−PEN ≤ bot(X) − top(Y) ≤ GAP`
   - X 底面落在 Y 頂面附近;`−PEN` 容許輕微穿插、`GAP` 容許輕微間隙。
2. **足跡重疊**:`xy_overlap(X,Y) / area(X) ≥ ON_XY`
   - `xy_overlap = max(0, min(xmaxX,xmaxY) − max(xminX,xminY)) · max(0, min(ymaxX,ymaxY) − max(yminX,yminY))`(兩 xy-AABB 交集面積)。
3. **在上**:`cenz(X) > cenz(Y)`。

| 參數 | 值 | 單位 | 定義 |
|---|---|---|---|
| PEN | 0.015 | m | 接觸穿透容差 |
| GAP | 0.03 | m | 接觸最大間隙 |
| ON_XY | 0.30 | — | X footprint 壓在 Y 上的最小比例 |

---

## 2. `blocks_access(X, Y)` — X 擋住接近 Y(此版以「視覺遮擋」為證據)

### 2a. GT(遮罩式,`gt_relations.compute_blocks`)
對每個拍攝視角 v(該視角 X、Y 皆有 amodal 與 modal 遮罩):
1. 對「被遮物」i:`hidden_i = amodal_i ∧ ¬modal_i`(完整輪廓減去含遮擋的可見輪廓 = 被遮區域)。
2. **被遮比例**:`occ_frac = |hidden_i| / |amodal_i|`;需 `≥ OCC_MIN` 才算 i 在此視角被遮。
3. **遮擋者**:`j* = argmax_j |hidden_i ∧ modal_j| / |hidden_i|`(蓋住被遮區最多者);需該比例 `≥ OCCLUDER_MIN`。
4. 累計 (j*, i) 出現的視角數;`blocks_access(j*, i)` 成立 ⟺ 出現於 `≥ MIN_VIEWS` 個視角。

### 2b. 預測(hull 幾何 z-buffer,`a1_rule.rule_blocks`)
每視角把各 instance 體素中心投影(降採樣 `DS`)→ 每像素取**最小深度**(z-buffer)。
對 (X 遮擋者, Y 被遮):`front = (Y有) ∧ (X有) ∧ (depth(X) < depth(Y))`;`occ_frac = |front| / |Y 投影像素|`;
取 `occ_frac` 最大的 X 為遮擋者,需 `≥ OCC_MIN`;累計 `≥ MIN_VIEWS` 視角成立。

| 參數 | 值 | 單位 | 定義 |
|---|---|---|---|
| OCC_MIN | 0.10 | — | 物 i 被遮比例下限(才算被遮) |
| OCCLUDER_MIN | 0.30 | — | 遮擋者蓋住被遮區的下限(僅 GT 用) |
| MIN_VIEWS | 2 | 視角 | 需成立的最少視角數 |
| DS | 4 | — | 預測 z-buffer 投影降採樣倍率(1280×720 → 320×180) |

> 註:GT 用渲染遮罩(amodal/modal),預測用 hull 自身 z-buffer;兩者皆免感測深度。

---

## 3. 前後左右 — 方向(`rel_recall.dir_triples`)

用兩物**世界座標質心** `(x,y)`(GT=實心 mesh 佔據體素均值;預測=instance 體素均值)。對有序對 (A,B):
- `right(A,B)` ⟺ `A.y − B.y > DIR_THR`;`left(A,B)` ⟺ `B.y − A.y > DIR_THR`
- `front(A,B)` ⟺ `A.x − B.x > DIR_THR`;`back(A,B)` ⟺ `B.x − A.x > DIR_THR`
- 質心差 `< DIR_THR` 的軸不產生關係(死區去雜訊)。每對最多 1 前後 + 1 左右。

| 參數 | 值 | 單位 | 定義 |
|---|---|---|---|
| DIR_THR | 0.03 | m | 質心差死區 |

**慣例**:前後 = 世界 x 軸、左右 = 世界 y 軸。命名為世界軸慣例,**未錨定機器人/相機視角**(對 recall 指標無影響,因 GT 與預測同慣例;若餵下游規劃需改錨定參考系)。

---

## 4. 輸出格式 `data/labels/<scene>/relations.json`
```json
{
  "scene": "...",
  "objects": ["..."],
  "relations": [
    {"type": "on", "x": "上物", "y": "底物", "gap": 0.01, "xy_overlap": 0.5},
    {"type": "blocks_access", "x": "遮擋者", "y": "被遮者", "n_views": 5, "max_occ_frac": 0.4}
  ],
  "params": {"PEN":0.015,"GAP":0.03,"ON_XY":0.30,"OCC_MIN":0.10,"OCCLUDER_MIN":0.30,"MIN_VIEWS":2}
}
```
(方向關係於評估時即時計算,不寫入 relations.json。)

## 5. 定位
- `on`:幾何(GT mesh / hull 體素);GT 與預測規則參數一致。
- `blocks_access`:GT=遮罩(amodal−modal)、預測=hull z-buffer;為 plan「REGRAD 物理」的幾何+遮罩近似(物理 drop-test 留 v2)。
- 方向:質心軸向比較。

## 6. 變更紀錄
- 早期草稿曾將 `blocks_access` 定為「頂向接近走廊(BLK_XY/H_MIN)」,**已棄用**;實作改為視覺遮擋(amodal−modal / z-buffer)。方向關係為後加。
