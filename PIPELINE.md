# YCB 資料集拍攝 Pipeline

## 流程概覽

```
A 視角規劃：A-0（選 x_offset）→ A-1 → A-2 → A-3 → A-4 → A-5（拍攝）
B 場景擺位：B-1（生成多物體 plan,強制不重疊）
C 後處理  ：C-1 GT 標籤 → C-2 產遮罩(A grounded_sam / B sam_clip) → C-3 評估 → C-4 visual hull(A foreground / B instance,皆 depth-free+label-free) → C-5 驗證
```

| 步驟 | 腳本 / 世界檔                            | 說明                                      | Webots | Planning Bridge |
| ---- | ---------------------------------------- | ----------------------------------------- | :----: | :-------------: |
| A-0  | `scan_x_offset.py`                       | 掃描各 x 偏移可達率，選定拍攝中心 x 座標 |   —    |        ✓        |
| A-1  | `generate_candidate_viewpoints_multi.py` | 生成候選視角（IK + 幾何過濾）             |   —    |        —        |
| A-2  | `ycb_viewpoint_validator_multi.wbt`      | 實際移動手臂，驗證碰撞與相機精度          |   ✓    |        —        |
| A-3  | `select_validated_viewpoints.py`         | 從通過視角中選取分布最廣子集              |   —    |        —        |
| A-4  | `plan_viewpoint_paths.py`                | 規劃視角間完整路徑（含時間參數化）        |   —    |        ✓        |
| A-5  | `ycb_path_executor_multi.wbt`            | 執行路徑並拍攝                            |   ✓    |        —        |

**x_offset 一致性：** A-1、A-2、A-3、A-4 使用的 `--x-offset` 必須相同。

---

## 環境參數（已確認值）

| 參數                      | 值                          |
| ------------------------- | --------------------------- |
| 機器人底座（world frame） | `[-0.4, 0.0, 0.0]` m        |
| 物體中心（world frame）   | `[x_offset, 0.0, 0.0]` m   |
| 最佳 x_offset             | **0.35 m**                  |
| 拍攝半徑                  | `[0.55, 0.60, 0.65, 0.70]` m |
| 仰角                      | `[45°, 60°, 75°, 90°]`      |
| 方位角步數                | 8（每 45°）                  |
| 工作空間偏移量（ws-offset）| **0.30 m**（sphere_r = cam_r - ws_offset；cam_r=0.65 → 0.35 m） |
| 選取視角數                | 12（`NUM_OUTPUT_POSES`）     |

---

## A-0　掃描 x 偏移可達率

在跑主流程前執行，選定拍攝中心 x 座標。不需要任何前置步驟。

```bash
# 步驟一：啟動 Planning Bridge（保持開著）
source /opt/ros/jazzy/setup.bash
source ~/webots_program/ros2_ws/install/setup.bash
ros2 launch ur5e_2f140_planning planning_bridge_launch.py

# 步驟二（另開終端）
source /opt/ros/jazzy/setup.bash
source ~/webots_program/ros2_ws/install/setup.bash
cd ~/webots_program/controllers/ycb_viewpoint_validator

# 掃描預設範圍
/usr/bin/python3 scan_x_offset.py --multi --ws-offset 0.30

# 自訂範圍
/usr/bin/python3 scan_x_offset.py --multi --ws-offset 0.30 \
    --x-offsets 0.30 0.33 0.35 0.37 0.40
```

| 項目         | 路徑                                                |
| ------------ | --------------------------------------------------- |
| 輸出（具名） | `data/viewpoints/x_offset_scan_cyl45_x{X1}_{X2}_....json` |
| 輸出（最新） | `data/viewpoints/x_offset_scan_latest.json`         |

> 只產生統計結果，不產生後續視角檔。選定 x_offset 後從 A-1 開始。

---

## A-1　生成候選視角

```bash
cd ~/webots_program/controllers/ycb_supervisor_capture
/usr/bin/python3 generate_candidate_viewpoints_multi.py --x-offset 0.35
```

| 項目         | 路徑                                                               |
| ------------ | ------------------------------------------------------------------ |
| 設定         | `controllers/ycb_supervisor_capture/candidate_viewpoint_config.py` |
| 輸入         | —                                                                  |
| 輸出（具名） | `data/viewpoints/candidate_viewpoints_multi_x+035.json`（覆蓋）   |

---

## A-2　Webots 碰撞 + 相機精度驗證

```bash
cd ~/webots_program
VALIDATOR_ARGS="--multi --x-offset 0.35" webots worlds/ycb_viewpoint_validator_multi.wbt
```

`VALIDATOR_ARGS` 優先於 world 檔的 `controllerArgs`，不需修改 `.wbt`。  
Controller 內部自動啟動 ROS2 bridge，**不需要外部 Planning Bridge**。

| 項目         | 路徑                                                              |
| ------------ | ----------------------------------------------------------------- |
| 輸入         | `data/viewpoints/candidate_viewpoints_multi_x+035.json`          |
| 輸出（具名） | `data/viewpoints/validated_viewpoints_multi_{TAG}_x+035.json`    |
| 輸出（最新） | `data/viewpoints/validated_viewpoints_multi_latest.json`          |

`TAG` 格式：`el{仰角}_az{步數}_r{半徑s}`

---

## A-3　選取分布最廣視角子集

```bash
cd ~/webots_program/controllers/ycb_viewpoint_validator
/usr/bin/python3 select_validated_viewpoints.py --multi --x-offset 0.35
```

| 項目         | 路徑                                                              |
| ------------ | ----------------------------------------------------------------- |
| 輸入         | `data/viewpoints/validated_viewpoints_multi_latest.json`         |
| 輸出（具名） | `data/viewpoints/selected_viewpoints_multi_x+035.json`           |
| 輸出（最新） | `data/viewpoints/selected_viewpoints_multi_latest.json`          |

| 參數      | 說明                                        |
| --------- | ------------------------------------------- |
| `--count` | 選取數量（預設 12，來自 `NUM_OUTPUT_POSES`） |

---

## A-4　規劃視角間路徑

```bash
# 步驟一：啟動 Planning Bridge
source /opt/ros/jazzy/setup.bash
source ~/webots_program/ros2_ws/install/setup.bash
ros2 launch ur5e_2f140_planning planning_bridge_launch.py

# 步驟二（另開終端）
source /opt/ros/jazzy/setup.bash
source ~/webots_program/ros2_ws/install/setup.bash
cd ~/webots_program/controllers/ycb_viewpoint_validator

/usr/bin/python3 plan_viewpoint_paths.py \
    --multi --x-offset 0.35 \
    --ws-offset 0.30 \
    --vel-scale 0.2 --acc-scale 0.2
```

| 項目         | 路徑                                                         |
| ------------ | ------------------------------------------------------------ |
| 輸入         | `data/viewpoints/selected_viewpoints_multi_x+035.json`      |
| 輸出（具名） | `data/viewpoints/planned_paths_multi_ws_minus030_x+035.json`|
| 輸出（最新） | `data/viewpoints/planned_paths_multi_latest.json`           |

| 參數           | 說明                                                        |
| -------------- | ----------------------------------------------------------- |
| `--ws-offset`  | 工作空間偏移量（m），預設 0.30；sphere_r = cam_r - ws_offset |
| `--vel-scale`  | 速度 scaling，0.0 ~ 1.0                                     |
| `--acc-scale`  | 加速度 scaling，0.0 ~ 1.0                                   |

---

## A-5　Webots 執行路徑並拍攝

```bash
cd ~/webots_program
webots worlds/ycb_path_executor_multi.wbt
```

| 項目 | 路徑                                          |
| ---- | --------------------------------------------- |
| 輸入 | `data/viewpoints/planned_paths_multi_latest.json` |
| 輸出 | `data/captures/a5_multi/{YYYYMMDD_HHMMSS}/`   |

---

## B　場景 plan 與物體擺位

固定四視角資料集（n1 單物體、n3/n4/n5 多物體）由 `ycb_supervisor_four_view_single/multi` 讀取
`data/scene_plans/` 的 plan 後拍攝。Plan 記錄每個場景的物體組合與各物體 `position_m`；
supervisor spawn 時直接用 plan 的 x、y，z 自動覆寫為「物體半高 + 間隙」。

| 場景類型      | plan 檔                                    | 命名               |
| ------------- | ------------------------------------------ | ------------------ |
| n1 單物體     | `data/scene_plans/single_scene_plan.json`  | `n1_sceneXXXX`     |
| n3/n4/n5 多物體 | `data/scene_plans/multi_scene_plan.json`   | `n{N}_sceneXXXX`   |

- **單物體**：物體固定置於拍攝中心 `[x_offset, 0.0, 0.0]`（目前 0.35），無重疊問題。
- **多物體**：生成時即**強制不重疊、且不超出工作空間**（見 B-1）。

### B-1　生成多物體場景 plan（強制不重疊、不出界）

一步完成「抽物體組合 + 擺位」，直接輸出最終 `multi_scene_plan.json`。
不重疊與不出界皆於生成時強制保證，**沒有「先產生重疊、再修正」的中間步驟**。

```bash
cd ~/webots_program
/usr/bin/python3 controllers/ycb_supervisor_four_view_multi/generate_multi_object_scenes.py
```

| 項目         | 路徑 / 值                                                                       |
| ------------ | ------------------------------------------------------------------------------- |
| 輸入         | `data/viewpoints/selected_viewpoints_multi_latest.json`、`ycb_geometries.json`、`config.py` |
| 輸出（覆寫） | `data/scene_plans/multi_scene_plan.json`                                        |
| 備份         | `data/scene_plans/multi_scene_plan.json.bak`（僅首次覆寫時建立）                 |

**組合**：依物體池為每個 group size（`GROUP_SIZES=(3,4,5)`，即 n3/n4/n5）抽選。
每個物體**剛好出現「該組物體數 N」次**（n3→3 次、n4→4 次、n5→5 次），不超量；場景命名 `n{N}_scene{XXXX}`。
因出現次數 = N，`物體數 × N` 必被 N 整除 → 每組場景數 = 物體池大小，不會有湊不齊的殘餘。

**擺位（綁定工作空間球，3D，不寫死）**：工作空間是半徑 `= cam_r − WS_OFFSET`（cam_r 取自
selected viewpoints 的 `radius_m`，目前 0.65 − 0.30 = **0.35 m**）、球心在桌面 (z=0) 的**球**。
物體坐在桌上 (z:0→h)。每物體 rejection sampling（大物體優先），兩項硬條件：

> - **整個物體在球內（含高度）**：頂端外緣 `sqrt((r+footprint/2)² + h²) ≤ WS` → 中心水平可動半徑
>   `= sqrt(WS² − h²) − footprint/2`。高物體被推向中心、頂端不戳出球面
>   （低仰角視角時手臂沿球面外側移動才不會掃到高物體 → 避免手臂撞翻物體 / 跳過視角）。
> - **不重疊**：兩物體中心距離 ≥ `(footprintA + footprintB)/2 + MARGIN`
>
> `footprint = max(size.x, size.y)`，`h = size.z`

**自動排除放不下的物體**：某組合排不下時移除失敗場景中 footprint 最大的物體、重新生成，直到全部可擺入，
並回報被移除者（目前移除 `048_hammer`/`059_chain`/`033_spatula`，最終池 61 → 每組 61 場景）。

| 參數（檔頭常數） | 說明                                                              |
| ---------------- | ----------------------------------------------------------------- |
| `GROUP_SIZES`    | 每場景物體數，預設 (3, 4, 5)                                       |
| `WS_OFFSET`      | 工作空間偏移；工作空間半徑 = cam_r − WS_OFFSET，**須與 A-4 `--ws-offset` 一致** |
| `MARGIN`         | 物體最小邊距，= `config.SPACING_MARGIN`（0.02 m）                 |
| `SEED`           | 隨機種子，固定可重現                                               |
| `MAX_ATTEMPTS`   | 單物體取樣上限；池中若有組合排不下會觸發上述自動排除               |

> ⚠ 重新生成 plan 後，既有 `data/labels/*/planned/` 與 `data/eval/*/planned/` 仍是舊座標所產，
> 需重拍 → 重產 planned 標籤 → 重跑 evaluate 才會與新 plan 一致。

---

## C　後處理：標籤 / 產遮罩 / 評估 / Visual Hull

拍攝完成後的資料集處理。**產遮罩拆成兩條對稱 pipeline**(各自產 `view_XX_mask_<class>.png`),
之後共用同一套評估與建殼。各步驟 python 環境不同(見每節)：

| 簡稱 | python | 用途 |
| ---- | ------ | ---- |
| `$LBL_PY` | `…/3.10.10/bin/python3` | pyrender(C-1) |
| `$VH_PY`  | `…/webots_visual_hull/bin/python3` | DINO+SAM、評估、torchhull(C-2A/C-3/C-4) |
| `$GS_PY`  | `…/grounded_sam/bin/python3` | SAM+CLIP(C-2B) |

```
C-1 GT 標籤
C-2 產遮罩 ┬ A. grounded_sam (DINO→SAM)  ┐ view_XX_mask_<class>.png
           └ B. sam_clip      (SAM→CLIP) ┘ → C-3 評估 / C-4 建殼 → C-5 驗證
```

### C-1　生成 GT 標籤（pyrender）

`tools/generate_labels.py`：依**實際相機位姿 + 物體置中**渲染 GT mask(與拍攝圖像素對齊),輸出 COCO `annotations.json`。

```bash
MODE=actual KEEP=1 ./tools/run_generate_labels_multi.sh 1 3 4 5
```
| 輸入 | `data/captures/multi_n{N}/<scene>/scene_manifest.json` |
| ---- | ---- |
| 輸出 | `data/labels/<scene>/{actual,planned}/{annotations.json, images, masks}` |

> 對齊關鍵：相機用 manifest 的 `rotation_rpy`,物體套 `-center`。稽核：`python tools/audit_multi_labels.py 1 3 4 5`。

### C-2　產遮罩（兩條 pipeline,擇一或都跑比較）

兩者都輸出 `view_XX_mask_<class>.png`,候選類別取自 manifest 物體(經 `config.PROMPT_TABLE`)。

**A. Grounded-SAM**(文字 prompt → DINO 找框 → SAM 切;`$VH_PY`)
```bash
$VH_PY grounded_sam/run_grounded_sam.py 1 3 4 5                  # 預設門檻
$VH_PY grounded_sam/run_grounded_sam.py 3 --box-threshold 0.3 --text-threshold 0.3 --nms-threshold 0.7
$VH_PY grounded_sam/run_grounded_sam.py n3_scene0001            # 單場景;FORCE=1 重做
```
→ `data/eval/grounded_sam_<box>_<text>_<nms>/multi_n{N}/<scene>/`(門檻決定資料夾)

**B. SAM + CLIP**(SAM 全自動切 → CLIP 認;不需文字接地/位置;`$GS_PY`)
```bash
$GS_PY sam_clip/run_sam_clip.py 1 3 4 5                          # 預設 ViT-B/32, prob 0.3
$GS_PY sam_clip/run_sam_clip.py 3 --prob-threshold 0.5          # 或 --clip-model ViT-B/16
$GS_PY sam_clip/run_sam_clip.py n3_scene0001                    # 單場景;FORCE=1 重做
```
→ `data/eval/sam_clip_<clip模型>_<prob>/multi_n{N}/<scene>/`(參數決定資料夾,如 `sam_clip_vitb32_0.3`)

| | A grounded_sam | B sam_clip |
| -- | -- | -- |
| 方法 | DINO 文字找框 → SAM | SAM 全自動 → CLIP 分類 |
| 權重(資料夾) | 門檻 `grounded_sam_<box>_<text>_<nms>` | CLIP+信心 `sam_clip_<clip>_<prob>` |
| 易錯 | DINO 類別貼錯物體 | CLIP 對小物/同色杯認錯 |

### C-3　評估（純讀遮罩 vs GT,不載模型）

`tools/evaluate_masks.py`：讀某 pipeline 的遮罩 + C-1 GT,算 IoU/偵測率,結果寫回該遮罩資料夾。**可評任一 pipeline**。

```bash
# 單場景:--pred-dir 指向該 pipeline 的場景遮罩夾
$VH_PY tools/evaluate_masks.py --labels data/labels/n3_scene0001/actual/annotations.json \
    --pred-dir data/eval/grounded_sam_0.25_0.25_0.8/multi_n3/n3_scene0001
# 批次:--weight-dir 指定方法資料夾(名稱即 C-2 產出的資料夾)
$VH_PY tools/run_evaluate_all.py --weight-dir grounded_sam_0.25_0.25_0.8
$VH_PY tools/run_evaluate_all.py --weight-dir sam_clip_vitb32_0.3
# 彙總(自動掃所有方法夾,各出一份 eval_summary.json)
python tools/aggregate_eval.py
```
| 產出(寫回 pred-dir) | `results.csv`、`summary.json`、`visualizations/`(比較圖) |
| ---- | ---- |
| 彙總 | `data/eval/<方法>/eval_summary.json`(n1/n3/n4/n5 + overall) |

> ⚠ **產遮罩(C-2)與評估(C-3)是分開的兩步,每組權重都要各跑一次**。
> `aggregate_eval` 只收**有 summary.json** 的方法夾 → 只產遮罩沒評估的權重不會出現在彙總。

### 多權重 / 多方法比較(實際操作順序)

每個「權重」= 一組參數 = 一個資料夾。要比較就**逐組**跑「產遮罩 → 評估」,最後一次 aggregate：

```bash
# 例:比較 3 組 grounded_sam 門檻 + 1 組 sam_clip
for cfg in "0.25 0.25 0.8" "0.25 0.25 0.7" "0.2 0.2 0.8"; do
  read b t n <<< "$cfg"
  $VH_PY grounded_sam/run_grounded_sam.py 1 3 4 5 --box-threshold $b --text-threshold $t --nms-threshold $n
  $VH_PY tools/run_evaluate_all.py --weight-dir grounded_sam_${b}_${t}_${n}
done
$GS_PY sam_clip/run_sam_clip.py 1 3 4 5 --prob-threshold 0.3
$VH_PY tools/run_evaluate_all.py --weight-dir sam_clip_vitb32_0.3

python tools/aggregate_eval.py     # 一次列出所有方法夾的 n1/n3/n4/n5 + overall
```
> 評估很慢的主因是寫 `visualizations/` 比較圖;若只要指標,之後可加 `--no-vis`(尚未實作)跳過。

### C-4　Visual Hull（torchhull 雕殼）

雕殼一律靠 `torchhull.visual_hull`(octree 雕刻),**它強制要一個 cube**(`cube_corner_bfl`+`cube_length`,無預設)當八叉樹根——體素解析度 = `cube_length / 2^level`。**torchhull 需 CUDA 12.1+**,已指向 `/usr/local/cuda-12.6`(設 `CUDACXX`)。

目前採用兩種**完全 depth-free、label-free** 的多物體方法(cube 用已知工作空間幾何寫死,不靠深度;物體切分靠幾何,不靠辨識標籤):

#### 方法 A — foreground(前景合併 → 3D 連通元件分物體)
不分物體、不分類:每 view 取「所有非背景前景」union,12 view 合併雕一坨,再用 3D 連通元件切開(物體空間不重疊 → 自然分坨)。
```bash
$GS_PY foreground_hull/make_foreground.py n3_scene0001   # 出 view_XX_mask_foreground.png
$VH_PY foreground_hull/split_hull.py     n3_scene0001    # 固定 cube 雕殼 + 連通元件切
```
→ `data/eval/foreground/<scene>/components/obj_*.obj`、`report.txt`。
缺點:物體**重疊/接觸**時會幻影橋接,切不開(非保證)。

#### 方法 B — instance(多視角幾何關聯 → per-object 單獨雕)
SAM 出 class-agnostic 遮罩 → 純幾何把跨視角遮罩關聯成 instance(不看標籤、不用深度)→ 每 instance 用自己遮罩單獨雕。**解掉「同物體跨視角被辨識成不同東西 → 對應不上」**。所有關聯法輸入皆 = `sam_only` 遮罩 + 相機位姿,輸出統一 schema `instances.json`(中心 + 各視角遮罩),可互換餵 carve/評估。

**關聯方法族(輸出 `data/eval/<method>/<scene>/instances.json`)**
| 程式 | 方法 | env |
|---|---|---|
| **`instance_hull/associate_voxel.py --multi-label`** | **VOXEL 多標籤(目前最佳)**:工作空間切體素→投影回各 view→落遮罩內視角數≥keep_frac保留(visual hull)→「多標籤 bitmask 相鄰一致」連通分物體 | webots_visual_hull |
| `instance_hull/epipolar_match.py` | **Epipolar+SymNMF**(Doi et al. ACCV2020):對極帶相似度→SymNMF 圖聚類→3D 體素去重。漏檢極低但過檢偏高 | webots_visual_hull |
| `instance_hull/associate.py` | set-cover(質心射線交會,原始) | grounded_sam |
| `associate_hdbscan.py` / `associate_dbscan.py` | 點分群(實驗,較差) | webots_visual_hull |
| `clip_hull.py` / `voxel_candidates.py`+`filter_candidates_clip.py` | CLIP 外觀(實驗,判別力不足、棄用) | grounded_sam |

```bash
# 目前最佳:VOXEL 多標籤
$VH_PY sam_only/sam_only.py n3_scene0001          # (grounded_sam)各 view SAM 遮罩
$VH_PY instance_hull/associate_voxel.py 1 3 4 5 --multi-label   # → instance_hull_voxel_ml/
$VH_PY instance_hull/carve_instances.py n3_scene0001 --root=instance_hull_voxel_ml  # per-object 雕殼
```

**評估(用 GT,只打分;方法本身零 GT)**
| 程式 | 指標 |
|---|---|
| `instance_hull/eval_reproj.py --root=<method>` | 找到率 + 重投影遮罩 vs GT 遮罩 IoU |
| `instance_hull/eval_clip_match.py --root=<method>` | CLIP 特徵↔名詞配對 + 漏檢/過檢/3D IoU(vs **GT visual hull**)|
| `instance_hull/precompute_clip.py`(grounded_sam)| 預存每遮罩 CLIP 影像特徵 + 物體名詞文字特徵 |
- 全 17 方法比較結論:3D IoU 都 ~0.79(定位相近),**真正差別在過檢率** → **voxel 多標籤最均衡(過檢最低)**;set-cover/epipolar 漏檢最低但過檢高。詳見 `data/eval/eval_clip_detail.csv`。
- `eval_3diou.py`(vs mesh)棄用:YCB mesh 非封閉、hull 本質填空腔。

> 棄用:舊 **per-class**(`build_torchhull.py --class-name`)被跨視角辨識不一致打壞、原靠 depth;舊 set-cover 的 `run_all.sh`(sam_only→associate.py→carve)仍可用但已非最佳。

### C-5　驗證（兩種）

**3D(Webots)** — 手臂 Home + 物體(實際位姿) + 各物體 hull(不同色)。用 `VH_SOURCE` 選方法:
```bash
# 方法 B(instance)
VH_SCENE=n3_scene0001 VH_SOURCE=instance   webots worlds/ycb_visual_hull_view.wbt
# 方法 A(foreground)
VH_SCENE=n3_scene0001 VH_SOURCE=foreground webots worlds/ycb_visual_hull_view.wbt
```
**2D 重投影** — hull 投回各拍攝視角疊圖檢查貼合：
```bash
$VH_PY Grounded-Segment-Anything/webots_visual_hull/project_visual_hull.py \
    --scene-dir data/captures/multi_n3/n3_scene0001
# → .../<scene>/reprojected_<hull>/view_XX_{mask,overlay,outline}.png
```

### 單場景一鍵(pipeline A)
`./tools/run_one_scene.sh n3_scene0001 [box text nms]`：C-2A 產遮罩 → C-3 評估 → C-4 建殼(需先有 C-1 標籤)。

---

## 資料目錄結構

```
data/viewpoints/
├── x_offset_scan_cyl45_x+020_+022_....json      A-0 具名（不覆蓋）
├── x_offset_scan_latest.json                    A-0 最新
├── candidate_viewpoints_multi_x+035.json        A-1（覆蓋）
├── validated_viewpoints_multi_{TAG}_x+035.json  A-2 具名（不覆蓋）
├── validated_viewpoints_multi_latest.json       A-2 最新 ← A-3 讀取
├── selected_viewpoints_multi_x+035.json         A-3 具名（覆蓋）
├── selected_viewpoints_multi_latest.json        A-3 最新 ← A-4 讀取（fallback）
├── planned_paths_multi_ws_minus030_x+035.json   A-4 具名（覆蓋）
└── planned_paths_multi_latest.json              A-4 最新 ← A-5 讀取

data/scene_plans/
├── single_scene_plan.json       n1 單物體場景（物體置中）
├── multi_scene_plan.json        n3/n4/n5 多物體場景（B-1 不重疊擺位後）
└── multi_scene_plan.json.bak    B-1 首次覆寫前的備份

data/captures/multi_n{N}/<scene>/   A-5/拍攝原始輸出：view_XX.png、_depth.npy、_pose.json、scene_manifest.json
data/labels/<scene>/{actual,planned}/   C-1 GT：annotations.json、images/、masks/
data/eval/<方法>/                    方法 = grounded_sam_<box>_<text>_<nms>(A) 或 sam_clip_<clip>_<prob>(B)
├── eval_summary.json                       C-3 彙總（aggregate_eval）
└── multi_n{N}/<scene>/                      view_XX_mask_<class>.png(C-2 產)、
                                             results.csv、summary.json、visualizations/(C-3 評估)、
                                             visual_hull_<class>.obj、hull_build_info.json(C-4)、
                                             reprojected_<hull>/(C-5 2D 驗證)
```

---

## 設定檔索引

| 設定檔                                                              | 用途                                           |
| ------------------------------------------------------------------- | ---------------------------------------------- |
| `controllers/ycb_supervisor_capture/candidate_viewpoint_config.py` | `HEMISPHERE_RADII_M`、仰角、方位角步數         |
| `controllers/ycb_viewpoint_validator/ycb_viewpoint_validator.py`   | A-2 controller，`VALIDATOR_ARGS` 傳入參數      |
| `controllers/ycb_viewpoint_validator/select_validated_viewpoints.py`| A-3，`--x-offset`、`--count`                  |
| `controllers/ycb_viewpoint_validator/plan_viewpoint_paths.py`      | A-4，`--ws-offset`、vel/acc                    |
| `ros2_ws/src/ur5e_2f140_planning/config/ompl_planning.yaml`        | MoveIt 規劃器設定                              |
| `controllers/ycb_supervisor_four_view_multi/generate_multi_object_scenes.py` | B-1 生成多物體 plan（強制不重疊：GROUP_SIZES／BASE_RADIUS／MARGIN／SEED） |
| `tools/generate_labels.py`（`run_generate_labels_multi.sh`）        | C-1 GT 標籤（相機位姿/置中對齊） |
| `grounded_sam/`（`grounded_sam.py` + `run_grounded_sam.py`）        | C-2A 產遮罩：DINO→SAM，`--box/text/nms-threshold` 決定權重資料夾 |
| `sam_clip/`（`sam_clip.py` + `run_sam_clip.py`）                    | C-2B 產遮罩：SAM 全自動→CLIP 分類（grounded_sam env） |
| `tools/evaluate_masks.py`（`run_evaluate_all.py`）                 | C-3 評估：讀遮罩 vs GT 算 IoU，`--pred-dir`/`--weight-dir` |
| `controllers/ycb_supervisor/config.py` 的 `PROMPT_TABLE`           | 物體名 → SAM/CLIP prompt（與名稱解耦，可逐項調整） |
| `tools/run_visual_hull_multi.py` + `webots_visual_hull/build_torchhull.py` | C-4 visual hull（torchhull，需 CUDA 12.6；空遮罩跳過 + `--masks-partial`） |
| `worlds/ycb_visual_hull_view.wbt`（`visual_hull_viewer`）/ `project_visual_hull.py` | C-5 驗證：3D 檢視 / 2D 重投影 |
