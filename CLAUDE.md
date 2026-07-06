# CLAUDE.md — 專案說明

## ⚠ 工作準則（最高優先）

**永遠不在沒有數據／證據佐證的狀態下推論原因或下結論。**
任何「失敗成因、因果關係、為什麼會這樣、結論」都必須**先用實際數據／實驗／程式驗證證明**，
才能陳述為事實。未經驗證的想法一律明確標示為「**待驗證假設**」，並在動手做任何依賴它的修改前先驗證。
- 反例（禁止）：看到碰撞就斷定「夾爪撞桌」並據此改程式 → 實測發現夾爪沒碰桌、真正是相機撞手臂。
- 正確：先用 fcl/FK 或 log 把碰撞的實際 link 對印出來，確認後再說成因、再決定修法。

## 專案概述

在 Webots 模擬環境中，使用 UR5e + Robotiq 2F-140 夾爪對 YCB 物件進行多視角影像擷取，
再對擷取影像做分割（mask）與 3D visual hull 重建，產生用於訓練物件辨識模型的資料集並評估各種分割方法。

整個系統分兩階段：

| 階段 | 位置 | 產物 |
|------|------|------|
| **一、拍攝** | `controllers/`（Webots） | `data/captures/<group>/<scene>/`：RGB 影像 + 相機位姿 |
| **二、後處理** | 根目錄各資料夾（`grounded_sam`、`sam_clip`、`sam_only`、`foreground_hull`、`instance_hull`、`tools`） | `data/eval/<method>/...`：遮罩、IoU 評估、visual hull mesh |

- **完整拍攝流程：** `PIPELINE.md`（A-0 ~ A-5，詳細指令與參數）

---

## 目錄結構

```
webots_program/
├── PIPELINE.md                        ← 完整拍攝流程（A-0 ~ A-5）
│
├── controllers/                       ← 【階段一】Webots controller（拍攝端，每子目錄=一個 controller）
│   ├── ycb_supervisor/                ← 共用基礎模組（非獨立 controller）
│   │   ├── config.py                  ← 全域參數（手臂速度、spawn、PROMPT_TABLE 等）
│   │   ├── ycb_scanner.py             ← 掃描可用 YCB 物件
│   │   └── ycb_geometries.json        ← 各物件幾何尺寸資料
│   ├── ycb_viewpoint_validator/       ← A-0/A-2/A-3/A-4：視角前處理工具集
│   ├── ycb_supervisor_capture/        ← A-1：候選視角生成
│   ├── ycb_path_executor/             ← A-5：執行路徑並拍攝
│   ├── ycb_supervisor_ros2_test/      ← ROS2 bridge 共用工具
│   ├── ycb_supervisor_four_view_*/    ← 舊版固定四視角擷取（run_capture_all.sh 用）
│   ├── ur5e_*_controller/             ← 手臂動作 / 實機擷取測試
│   ├── workspace_supervisor/          ← 視覺化可達工作空間球體
│   ├── visual_hull_check_supervisor/  ← Visual hull 驗證
│   └── visual_hull_viewer/            ← Visual hull 檢視
│
├── ── 【階段二】產遮罩（class-specific，輸出 view_XX_mask_<class>.png）──
├── grounded_sam/                      ← 方法 A：GroundingDINO 文字找框 → SAM 分割
│   ├── grounded_sam.py                ← 核心（回傳 {class: mask}）  [webots_visual_hull 環境]
│   └── run_grounded_sam.py            ← 批次跑場景
├── sam_clip/                          ← 方法：SAM 全自動切 → CLIP 分類（不需文字框/位置）
│   ├── sam_clip.py                    ← 核心  [grounded_sam 環境]
│   └── run_sam_clip.py                ← 批次跑場景
│
├── ── 【階段二】class-agnostic 分割 + visual hull ──
├── sam_only/                          ← SAM 全自動切「所有」遮罩（不分類）
│   └── sam_only.py                    ← 輸出 data/eval/sam_only/<scene>/<view>/  [grounded_sam 環境]
├── foreground_hull/                   ← 方法：不分物體，前景聯集 → 固定 cube 雕殼 → 連通元件分物體
│   ├── make_foreground.py             ← ① 每 view 前景/背景二值遮罩  [grounded_sam 環境]
│   └── split_hull.py                  ← ② 固定 cube carve + trimesh 連通元件分物體  [webots_visual_hull 環境]
├── instance_hull/                     ← 方法 B：多視角幾何關聯 → per-object 雕殼（全部不用 GT/深度；輸入=SAM遮罩+位姿，輸出 instances.json）
│   │  ── 關聯方法（產 data/eval/<method>/<scene>/instances.json）──
│   ├── associate_voxel.py             ← ★最佳 VOXEL：體素投影+遮罩歸屬連通；--multi-label 多標籤(體素可屬多遮罩)  [webots_visual_hull]
│   ├── epipolar_match.py              ← Epipolar band + SymNMF 圖聚類(Doi et al. ACCV2020);漏檢低過檢高  [webots_visual_hull]
│   ├── associate.py                   ← set-cover：質心射線跨 view 交會(原始)  [grounded_sam]
│   ├── associate_hdbscan.py / associate_dbscan.py ← 點分群(實驗,較差)  [webots_visual_hull]
│   ├── clip_hull.py / voxel_candidates.py / filter_candidates_clip.py ← CLIP 外觀(實驗,判別力不足棄用)
│   │  ── 雕殼 + 評估 ──
│   ├── carve_instances.py             ← 每 instance 用自己遮罩 torchhull 雕 per-object hull（--root=<method>）  [webots_visual_hull]
│   ├── precompute_clip.py             ← 預存每遮罩 CLIP 影像特徵 + 物體名詞文字特徵  [grounded_sam]
│   ├── eval_reproj.py                 ← 評估：找到率 + 重投影遮罩 vs GT IoU
│   ├── eval_clip_match.py             ← 評估：CLIP特徵↔名詞配對 + 漏檢/過檢/3D IoU(vs GT visual hull)
│   ├── eval_instances.py / eval_3diou.py(棄) ← 早期評估(2D IoU / 3D mesh IoU)
│   └── run_all.sh                     ← 舊 set-cover 整批：sam_only → associate → carve_instances
│
├── tools/                            ← 評估與標註工具
│   ├── evaluate_masks.py              ← 純評估：預測遮罩 vs GT 算 IoU/像素正確率  [webots_visual_hull 環境]
│   ├── generate_labels.py             ← 產 GT annotations.json
│   ├── run_evaluate_all.{py,sh}       ← 批次評估
│   ├── run_visual_hull_multi.py       ← 批次建 visual hull
│   └── aggregate_eval.py / audit_multi_labels.py / migrate_sam_masks.sh …
│
├── Grounded-Segment-Anything/         ← GSA 子模組（GroundingDINO + segment_anything + sam_vit_b_01ec64.pth）
│
├── data/                              ← 所有輸入輸出（不進 git）
│   ├── captures/multi_n{1,3,4,5}/<scene>/  ← 階段一輸出：view_XX.png + view_XX_pose.json
│   ├── viewpoints/                    ← 各步驟視角 JSON（詳見 PIPELINE.md）
│   ├── scene_plans/                   ← 多物件場景配置 / manifest
│   ├── labels/<scene>/actual/annotations.json ← GT 標註
│   └── eval/<method>/                 ← 階段二輸出（遮罩、hull、instances.json、eval_reproj|eval_clip/summary.json）
│       ├── sam_only/<scene>/<view>/   ← SAM 遮罩 + clip_feats.npy（預存 CLIP 影像特徵）
│       ├── clip_text_feats.npz        ← 物體名詞 CLIP 文字特徵（precompute_clip 產）
│       └── eval_clip_detail.csv       ← 全方法×場景×物體 評估明細（found/cos/iou3d）
│
├── ros2_ws/                           ← ROS2 workspace（MoveIt planning bridge + 夾爪驅動）
├── protos/                            ← Webots 自訂 proto（相機、夾爪手臂）
├── worlds/                            ← Webots 場景檔（.wbt）
├── urdfs/ycb_assets/                  ← YCB 物件 URDF/mesh（ASSET_BASE 指向此處）
├── meshes/ libraries/ plugins/        ← Webots 資產（相機/手臂 mesh、plugin）
├── run_capture_all.sh                 ← 舊版四視角批次拍攝（n1/n3/n4/n5 全場景）
└── superupdate.sh                     ← 環境更新腳本
```

---

## Python 環境（重要）

階段二橫跨**三個** Python 直譯器，腳本 shebang 已寫死，請依此選環境：

| 環境 | 路徑 | 提供 | 用於 |
|------|------|------|------|
| Webots 內建 | `/usr/bin/python3`（3.8.10） | Webots controller API | **階段一**所有 controller |
| `grounded_sam` | `/home/cho/.pyenv/versions/grounded_sam/bin/python3` | torch + segment_anything + clip + GroundingDINO | `sam_clip`、`sam_only`、`grounded_sam`(核心)、`make_foreground`、`associate` |
| `webots_visual_hull` | `/home/cho/.pyenv/versions/webots_visual_hull/bin/python3` | torch + torchhull + trimesh + GroundingDINO | `split_hull`、`carve_instances`、`evaluate_masks`、`grounded_sam.py`(核心) |

- 建 hull 需 `export CUDACXX=/usr/local/cuda-12.6/bin/nvcc`（見 `instance_hull/run_all.sh`）。
- 多數腳本支援 `FORCE=1` 重做（忽略已存在輸出）。

---

## 模組依賴關係

```
controllers/ycb_supervisor/config.py              ← 階段一共用參數；PROMPT_TABLE 供階段二產遮罩用
controllers/ycb_supervisor_ros2_test/ros2_bridge_utils.py ← A-2/A-4 bridge 工具
controllers/ycb_supervisor_capture/candidate_viewpoint_config.py ← A-1/A-2/A-4 共用視角設定
Grounded-Segment-Anything/{segment_anything,GroundingDINO} ← 所有產遮罩腳本以 sys.path 引入
階段二產遮罩腳本輸出命名一致（view_XX_mask_<class>.png）→ 可互換餵 evaluate_masks / build hull
```

---

## 環境參數

| 項目 | 值 |
|------|----|
| Webots | 2023.x |
| ROS2 | Jazzy |
| 機器人底座（world frame） | `[-0.4, 0.0, 0.0]` m |
| 最佳拍攝中心 x_offset | **0.35 m** |
| 方位角步數 | 8（每 45°） |
| 選取視角數 | 12（`NUM_OUTPUT_POSES`） |
| 工作空間偏移量 | `ws_offset=0.30 m`；`sphere_r = cam_r - ws_offset`（cam_r=0.65 → 工作空間半徑 0.35 m） |
| 相機水平 FOV | `HFOV_RAD = 1.4746` |
| 場景組 | n1（64 單物體）、n3/n4/n5（各 61 多物體場景） |

---

## 常見操作

```bash
# 啟動 ROS2 Planning Bridge（A-0、A-4 需要）
source /opt/ros/jazzy/setup.bash && source ~/webots_program/ros2_ws/install/setup.bash
ros2 launch ur5e_2f140_planning planning_bridge_launch.py

# A-2：Webots 驗證（controller 自動啟動 bridge）
VALIDATOR_ARGS="--multi --x-offset 0.35" webots worlds/ycb_viewpoint_validator_multi.wbt

# A-5：執行拍攝
webots worlds/ycb_path_executor_multi.wbt

# 階段二 ── 產遮罩（單一場景 / 整組 / 多組）
./sam_clip/run_sam_clip.py n3_scene0001        # SAM+CLIP
./sam_only/sam_only.py 3                        # SAM 全部遮罩，整組 n3
FORCE=1 ./grounded_sam/run_grounded_sam.py 3    # Grounded-SAM 重做

# 階段二 ── instance hull（B 方法，三步整批）
./instance_hull/run_all.sh 3                    # sam_only → associate → carve_instances

# 階段二 ── foreground hull（不分物體，前景聯集 → 雕殼分物體）
./foreground_hull/make_foreground.py n3_scene0001
./foreground_hull/split_hull.py  n3_scene0001

# 階段二 ── 評估遮罩 vs GT
python tools/evaluate_masks.py \
  --labels   data/labels/n3_scene0001/actual/annotations.json \
  --pred-dir data/eval/grounded_sam_0.25_0.25_0.8/multi_n3/n3_scene0001
```

---

## 重要慣例

- **`_latest.json`** 是各步驟最新輸出，供下一步讀取；**具名 JSON** 保留歷史不覆蓋。
- `VALIDATOR_ARGS` 環境變數優先於 `.wbt` 的 `controllerArgs`，改參數無需動 world 檔。
- `controllers/ycb_supervisor/` 不是獨立 controller，是共用函式庫，被其他 controller 透過 `sys.path` 引入。
- **階段一** controller 一律用 `/usr/bin/python3`（Webots 相容）；**階段二**依上表選 pyenv 環境（shebang 已寫死）。
- 階段二產遮罩腳本輸出命名統一為 `view_XX_mask_<class>.png`，方法之間可互換評估、互餵建 hull。
- 後處理腳本參數慣例：傳 `n3_scene0001`=單場景、`3`=整組 n3、`1 3 4 5`=多組；`FORCE=1`=忽略已存在重做。
- `data/` 不進 git；`data/eval/<method>/` 子目錄名常含閾值（如 `grounded_sam_0.25_0.25_0.8`）以區分參數。
- **相機 mount 是「從 world 檔解析」不是讀 config**：A-1 視角生成的相機位移由 `load_wbt_mounts()` 從 `candidate_viewpoint_config.WORLD_FILE`（`worlds/ycb_supervisor_four_view_capture_multi.wbt`）的 `DEF UR5E_CAMERA translation` 讀取；`T_FLANGE_TO_D455_M` 只是 fallback。相機 mount **散在多個 world 檔**（A-1 來源／A-2 validator／armmove·multicam 拍攝 world／MoveIt URDF），改一定要**全部同步**再重跑，詳見 `PIPELINE.md` A-1 的「★★ 相機 mount 來源」。
