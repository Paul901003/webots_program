# Visual Hull Carving — 實作規格 (模擬環境 / 免深度)

> 本文件是「目的 → 契約 → 演算法 → 介面 → 不變式 → 驗收測試」的單一真相來源。
> 任何實作只要通過第 6 節的所有驗收測試，即視為與第 0 節的目的一致。
> 撰寫程式者只需讀本文件，不需其他上下文。

---

## 0. 目的 (PURPOSE — 唯一真相)

從機械手臂端末 (eye-in-hand) 相機在多個視角拍攝的 RGB 前景**剪影遮罩 (silhouette masks)**，
在**不使用深度**的前提下，重建桌面上物體的 3D 佔據體積 (visual hull / 佔據網格)。

對齊機制：所有視角的相機內外參都表達在**同一個世界座標系**中；不同視角靠這個共同座標系自動對齊，
**不靠任何影像對影像的配準 (no image-to-image registration / no SfM)**。

輸出的佔據網格是下游關係推理 (GNN) 的輸入，因此必須滿足：
- (P1) **度量正確**：輸出錨定在世界座標系，每個 voxel 有明確的世界座標與實體尺寸 (公尺)。
- (P2) **保守 (conservative)**：寧可多保留、不可誤刪真實物體所佔的 voxel。
- (P3) **可重現**：相同輸入必得相同輸出，無隨機性。
- (P4) **對齊正確**：hull 重投影回任一視角的剪影，必須與該視角輸入遮罩吻合。

---

## 1. 非目的與假設 (NON-GOALS / ASSUMPTIONS — 不成立則結果無效)

- (A1) **不做手眼校正**。外參為模擬器提供的 ground-truth。但介面須保留外參為可替換輸入，
  以便未來換成真機校正結果而不改動演算法 (sim-to-real 介面預留)。
- (A2) **無法重建凹面**。visual hull 的本質限制；下游不得期待物體內腔 (如碗內部)。
- (A3) **只觀測上半球**，物體底面 (貼桌面側) 不可見，以**已知支撐平面 (桌面)** 封底。
  假設所有物體靜置於桌面上。
- (A4) **Stop-and-shoot**：每個外參對應一張靜止影像，無運動模糊、無時間不同步。
- (A5) 遮罩為**二值**：前景 (物體) 與背景。多物體的實例區分不在本模組範圍 (本模組只算「物體 vs 非物體」的佔據；實例分離由下游或多次呼叫處理)。

---

## 2. 資料契約 (DATA CONTRACT — 型別 / 形狀 / 單位 / 座標系)

所有陣列為 numpy。長度為 V 的 list 代表 V 個視角，**順序必須三者一致** (masks[i] ↔ intrinsics[i] ↔ extrinsics[i])。

| 名稱 | 型別 / 形狀 | 單位 | 定義 (務必精確) |
|---|---|---|---|
| `masks[i]` | (H, W) bool | — | **True = 前景 (物體)**，False = 背景。polarity 不可搞反。 |
| `intrinsics[i]` (K) | (3, 3) float | 像素 | 針孔模型。`[[fx,0,cx],[0,fy,cy],[0,0,1]]`，主點 (cx,cy) 以像素計。 |
| `extrinsics[i]` (R, t) | R:(3,3), t:(3,) float | t 為公尺 | **world→camera**：`X_cam = R @ X_world + t`。見 (C1)。 |
| `grid_min`, `grid_max` | (3,) float | 公尺 | 世界座標系下，涵蓋物體的軸對齊包圍盒 (AABB) 的兩個對角。 |
| `voxel_size` | float | 公尺 | 單一 voxel 邊長。 |
| `table_z` | float \| None | 公尺 | 支撐平面在世界座標系的 z 值；None 表示不封底。 |

**影像座標慣例 (C，與多數 CV 函式庫一致)：**
- 影像原點在**左上角**，u 向右 (對應欄/column)，v 向下 (對應列/row)。
- 像素索引：`mask[round(v), round(u)]`（先 row 後 col）。

**輸出 `VisualHull`：**
```
occupancy : (Nx, Ny, Nz) bool      # True = 佔據
observed  : (Nx, Ny, Nz) bool      # True = 該 voxel 至少被一個視角「看進去」過 (in_front & in_bounds)，與佔據無關
grid_min  : (3,) float             # voxel[0,0,0] 中心對應 grid_min + 0.5*voxel_size
voxel_size: float
frame     : "world"                # 明示輸出座標系
```
voxel 索引 (i,j,k) → 世界座標：`X_world = grid_min + (array([i,j,k]) + 0.5) * voxel_size`  （對應 P1）

**`observed` 的用途 (場景 3a 必需)**：`occupancy=False 且 observed=False` 的 voxel 是**未觀測空間**
(沒被任一視角看進去，可能藏著完全遮擋的物體)；`occupancy=False 且 observed=True` 才是「確定為空」。
場景 3a 的揭露決策靠這個區分——見 `plan_check_schema.md` 的翻譯層。
計算：在第 4 節迴圈裡，對每個 voxel 額外記錄「是否曾在某視角 `in_front & in_bounds`」(不論前景背景)，OR 起來即得 `observed`。

---

## 3. 慣例檢查清單 (CONVENTION CHECKLIST — 靜默 bug 來源)

實作完成後、信任輸出前，逐項確認。任一項錯誤都會讓 hull 安靜地壞掉而不報錯：

- [ ] (C1) **外參方向**：是 world→camera (`X_cam = R@X_world + t`) 而非 camera→world。
  若模擬器給的是 camera→world (相機位姿)，須先反轉：`R_w2c = R_c2w.T`, `t_w2c = -R_c2w.T @ t_c2w`。
- [ ] (C2) **影像原點與軸向**：左上原點、u 右、v 下 (見第 2 節)。
- [ ] (C3) **相機後方的點**：投影前若 `X_cam.z <= 0` (在相機後方或在像平面上)，該 voxel 對此視角視為**背景 (雕掉)**。
- [ ] (C4) **投影到影像外的點**：u 或 v 落在 [0,W)×[0,H) 之外，依 `outside_is_background` 決定 (預設 True = 視為背景雕掉)。此選擇影響 hull 邊緣，須明示。
- [ ] (C5) **遮罩 polarity**：前景為 True。

---

## 4. 演算法 (ALGORITHM — carving 的本質)

**核心觀念**：不把遮罩投影到 3D，而是反過來——**把每個 3D voxel 投影到每張 2D 影像，檢查它落在前景還是背景**。
一個 voxel **唯有投影到「所有」視角都落在前景**，才保留 (intersection)。這直接實現 P2 的保守性，且使 (P4) 對齊成立。

```text
輸入: masks[V], intrinsics[V](K), extrinsics[V](R,t, world→camera),
      grid_min, grid_max, voxel_size, table_z, outside_is_background
輸出: occupancy (Nx,Ny,Nz) bool

1. 依 grid_min/grid_max/voxel_size 建立 voxel 中心點座標 (世界座標系)。
2. occupancy ← 全部 True
3. for each view i in [0..V):
       將所有 voxel 中心 X_world 一次轉到相機座標:  X_cam = (R_i @ X_world.T).T + t_i
       z = X_cam[:,2]
       u = K_i[0,0]*X_cam[:,0]/z + K_i[0,2]
       v = K_i[1,1]*X_cam[:,1]/z + K_i[1,2]
       in_front  = z > 0                                   # (C3)
       in_bounds = (0<=u<W) & (0<=v<H)                     # (C4)
       fg = False 陣列
       對 in_front & in_bounds 的 voxel:  fg = masks[i][round(v), round(u)]
       若 outside_is_background:  此視角的 keep = in_front & in_bounds & fg
       否則:                      此視角的 keep = (~in_bounds) | (in_front & fg)
       occupancy ← occupancy AND keep        # 任一視角是背景就雕掉 = 交集
4. (封底) 若 table_z 非 None:  將世界座標 z < table_z 的 voxel 設為 False     # (A3)
5. 回傳 occupancy
```

**保守性後果 (務必理解)**：因為是交集，**任一視角的遮罩少切了物體一塊，那塊就被永久雕掉**。
故遮罩**寧可過切 (含一點背景) 也不可少切 (漏掉物體)**；對應 P2，並與下游對「壞遮罩敏感度」的評估直接相關。

**效能**：第 3 步必須向量化 (一次投影全部 voxel)；V 個視角為外層迴圈即可，不需逐 voxel 迴圈。

---

## 5. 介面 (INTERFACE — 函式簽章須與契約一致)

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class VisualHull:
    occupancy: np.ndarray      # (Nx,Ny,Nz) bool
    observed: np.ndarray       # (Nx,Ny,Nz) bool, 曾被任一視角看進去
    grid_min: np.ndarray       # (3,) float, 公尺
    voxel_size: float          # 公尺
    frame: str = "world"

def carve_visual_hull(
    masks: list[np.ndarray],                 # [V] (H,W) bool, True=前景
    intrinsics: list[np.ndarray],            # [V] (3,3)
    extrinsics_w2c: list[tuple[np.ndarray, np.ndarray]],  # [V] (R(3,3), t(3,)); X_cam=R@X_world+t
    grid_min: np.ndarray,                    # (3,) 公尺
    grid_max: np.ndarray,                    # (3,) 公尺
    voxel_size: float,                       # 公尺
    table_z: float | None = None,            # 支撐平面世界 z；None=不封底
    outside_is_background: bool = True,       # (C4)
) -> VisualHull: ...

# 若模擬器提供 camera→world，先用此轉換 (對應 C1)，不要改 carve 內部
def c2w_to_w2c(R_c2w: np.ndarray, t_c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    R = R_c2w.T
    t = -R_c2w.T @ t_c2w
    return R, t
```

---

## 6. 驗收測試 (ACCEPTANCE TESTS — 把目的編碼成測試)

每個測試標註它驗證哪條目的。**全部通過 ⟺ 與第 0 節目的一致。**

- **T1 — 標準球 (驗 P1, P2, A2)**：合成一個半徑 r、中心已知的球，渲染數個視角的剪影。
  碰 hull 的世界包圍盒邊長 ≈ 2r (容差 ≤ 1 voxel)；中心位置誤差 ≤ 1 voxel。
  體積落在 [球體積, 球外接立方體體積] 之間 (hull 為過估計，符合 A2)。

- **T2 — 方向慣例 (驗 C1)**：故意把 camera→world 當成 world→camera 餵入。
  預期 hull **塌縮或全空**。此測試證明 (C1) 重要，且目前設定正確 (正確設定下 T1 通過、本測試異常)。

- **T3 — 單調性 (驗 P2 的交集性質)**：`hull(views[:k+1]).occupancy` 必為 `hull(views[:k]).occupancy` 的子集
  (加視角只會縮小或不變，不會變大)。對應第 4 節交集邏輯。

- **T4 — 多物體分離 (驗 P1)**：兩個分開的立方體 → 佔據網格出現兩個不相連的連通元件。

- **T5 — 封底 (驗 A3)**：設 table_z 後，世界 z < table_z 的 voxel 一律為 False。

- **T6 — 對齊自檢 (驗 P4，這是最關鍵的 smoke test)**：
  取少數視角、粗網格、單一凸物。將輸出 hull 重投影回每個視角得到 hull 剪影，
  與輸入遮罩比對 IoU。對齊正確時 IoU 應高 (例如 > 0.9)；若低，代表 C1–C4 某項慣例錯誤。
  **此測試須在調 voxel 解析度之前先通過**——對齊不對，解析度無意義。

- **T7 — 可重現 (驗 P3)**：同輸入呼叫兩次，`occupancy` 完全相等。

- **T10 — 未觀測空間 (驗場景 3a 所需)**：一個大物體後方放一個小物體，使小物體在「所有視角」都被大物體完全遮擋。
  預期：小物體所在 voxel `occupancy=False` (沒被重建出來) 但 `observed=False` (未觀測)；
  而場景中確定為空的區域 `occupancy=False 且 observed=True`。
  此測試確認「未觀測 ≠ 確定為空」可被區分——揭露決策的幾何基礎。

> 論文 ablation 掛載點 (非驗收，但用同一介面跑)：
> 體積 vs 視角數 (找收斂飽和點)；體積 vs voxel_size (找夠用解析度)；
> hull 誤差 vs 遮罩品質 (GT 遮罩 vs SAM 遮罩)。

---

## 7. 實作順序 (給撰寫者的執行順序)

1. 先實作 `carve_visual_hull` + `c2w_to_w2c`，嚴格照第 2、3 節慣例。
2. 先過 **T6 (對齊自檢)** 與 **T2 (方向慣例)**——這兩個確保幾何主幹對。
3. 再過 T1、T3、T5、T7。
4. 最後才接 ablation 掛載點 (解析度、視角數收斂)。

**禁止**：在 T6 通過前調整 voxel_size 或視角數;在 T2 釐清外參方向前信任任何 hull 輸出。
