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
- **後處理現行主線：** `srp/`（免深度 visual-hull 實例分離：stage1 hull → stage2 關聯/評估 → stage3 關係GT → stage4 probing）。上表「二、後處理」列的根目錄方法為 **legacy**（保留供參考）。

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
├── srp/                               ← 【現行主線】免深度 visual-hull 實例分離管線（io+scene_gen+stage1~4；取代下方 legacy 根目錄方法）
│   ├── io/                            ← 共用 IO：camera(內外參)、masks(SAM遮罩+背景排除)、viewpoints(讀A-3挑選)、labels(場景名→分層目錄)
│   ├── scene_gen/                     ← 生成關係豐富場景（堆疊/密集遮擋）+ 實驗可視化世界
│   ├── stage1_hull/                   ← Stage1 visual hull：carve(GPU核心)、run_scene(真實場景)、arm_silhouette(FK手臂剪影減前景)、add_surface_mask、test_carve(T1–T10驗收)  [webots_visual_hull]
│   ├── stage2_instances/             ← Stage2 跨視角實例關聯 + 評估（不用深度，只用 SAM 遮罩+位姿）
│   │   ├─ 關聯：associate(voxel投影+遮罩歸屬agree連通,規格版)、cg_associate(ConceptGraphs逐視角)、cg_batch(全域批次)、voxel_sem_{vote,cluster,paper}(語意投票/分群/論文法)
│   │   ├─ 語意/切分：precompute_clip_mean(預存遮罩CLIP特徵)、mask_clip_cluster、refine_masks、semantic_split、split_instances
│   │   ├─ 填實/表面：fill_solid(表面labels填實心)、add_surface_mask(6鄰居表面)
│   │   ├─ 評估：eval(3D-IoU Hungarian)、eval_mesh、eval_surface、eval_reproj2d
│   │   └─ 報告：gen_hull_report(HTML)、gen_viz_objs(hull_viz上色)
│   ├── stage3_graph/                  ← 物體級關係 GT：on(支撐)/blocks_access(視覺遮擋)/前後/左右；皆模擬器真值，不用預測、不用深度；GT_RELATIONS_SPEC.md
│   └── stage4_probe/                  ← probing/診斷：a1_rule(規則基線復現關係)、geo_match(純幾何 ncut 分離堆疊)、SAM recall、per-object 找到率、堆疊分離診斷
│
├── ── 【legacy：早期單/多方法探索；現行主線已移至 srp/，以下整段保留供參考】──────────────
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
│   ├── captures/multi_<組>/<scene>/       ← 拍攝輸出（原版）：view_XX.png + view_XX_pose.json
│   ├── captures_fast/multi_<組>/<scene>/  ← noise=0 GT 版拍攝（srp 主線用；組=n1/n3/n4/n5/occ3-5/stack3-5）
│   ├── viewpoints/                    ← 各步驟視角 JSON（詳見 PIPELINE.md）
│   ├── scene_plans/                   ← 多物件場景配置 / manifest
│   ├── labels/<類別>/<數量>/<scene>/  ← GT 標註,依 n/occ/stack + 物體數分層(如 labels/n/3/n3_scene0001/)
│   │       {actual/annotations.json(modal), amodal/annotations.json, relations.json, scene_graph_gt/}
│   └── eval/                          ← 階段二 / srp 輸出
│       ├─ 【legacy】<method>/         ← 舊方法：sam_only/<scene>/<view>/(遮罩+clip_feats.npy)、clip_text_feats.npz、eval_clip_detail.csv、各 hull/instances.json
│       └─ 【srp】 sam_only_fast/、mobilesamv2_fast/  ← class-agnostic SAM / MobileSAMv2 遮罩（每遮罩 clip_mean_feats.npy 預存 CLIP 特徵）
│           srp_hull_mv2_v12_am{0,1,2,3}/  ← ★現行 12視角 MobileSAM 前景 hull（am=allow_miss;定案 am1）；hull.npz 含 occupancy/surface/build_meta
│           srp_hull_semcluster_clip_am{0,1,2,3}/ ← semcluster(CLIP-B32) instances（讀上面對應 am 的 hull；含 hull_gt 評估/reports/missed_* CSV）
│           srp_hull_v{6,8,10,12,14,16,18,20,34}/ ← viewcount 掃描 hull（sam_only,不同視角數,miss_frac=0.2;分析用）
│           （已棄用/刪除：srp_hull_mobilesamv2、_bf＝漏加 --num-views 的 34視角 hull）
│           srp_hull_{cg,semcluster,sempaper,semvote}/（sam_only 12視角,v12 為源）← 各關聯法 instances（_solid=填實心供3D-IoU）
│           gt_hull_cache/             ← eval.py 的 GT amodal hull 快取（依 scene+voxel_size）
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
| `webots_visual_hull` | `/home/cho/.pyenv/versions/webots_visual_hull/bin/python3`（3.10） | torch + torchhull + trimesh + GroundingDINO + **open_clip(CLIP語意)+DINOv2** | **srp/ 全管線 stage1–4**、`split_hull`、`carve_instances`、`evaluate_masks`、`grounded_sam.py`(核心) |

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
| 場景組 | n1（64 單物體）、n3/n4/n5（各 61 多物體）、occ3/4/5（各 20 遮擋）、stack3/4/5（各 20 堆疊）；**共 10 組 367 場** |

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

# 階段二 ── 評估遮罩 vs GT（legacy）
python tools/evaluate_masks.py \
  --labels   data/labels/n/3/n3_scene0001/actual/annotations.json \
  --pred-dir data/eval/grounded_sam_0.25_0.25_0.8/multi_n3/n3_scene0001

# ═══ srp 主線（免深度 visual hull 實例分離；env 詳見「重要慣例」）═══
# Stage1 hull（MobileSAM 前景,單場景）→ 之後補 surface
#   ★ 一定要 --num-views 12（A-3,否則預設吃全 34 視角）＋ --allow-miss 1（軟 hull,定案值,見下註）
SAM_ROOT=$PWD/data/eval/mobilesamv2_fast CAPTURES_ROOT=$PWD/data/captures_fast \
  srp/stage1_hull/run_scene.py stack3_scene0001 --num-views 12 --allow-miss 1 --root srp_hull_mv2_v12_am1
srp/stage2_instances/add_surface_mask.py stack3_scene0001 --root srp_hull_mv2_v12_am1
# Stage2 語意分群（12 視角，整組 n3）
HULL_ROOT=$PWD/data/eval/srp_hull_v12 SAM_ROOT=$PWD/data/eval/sam_only_fast \
  OUT_ROOT=srp_hull_semcluster srp/stage2_instances/voxel_sem_cluster.py 3
# Stage2 評估 3D-IoU@0.5（cg 表面法要先 fill_solid.py 填實心）
srp/stage2_instances/eval.py stack3_scene0001 --root srp_hull_semcluster --iou 0.5
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
- **`data/labels/` 依 `<類別>/<數量>/<scene>/` 分層**（n/occ/stack + 物體數，如 `labels/n/3/n3_scene0001/`）。程式一律透過 `srp/io/labels.py`：`from labels import LABELS`（`LABELS / scene / ...` 拼接與 `LABELS.glob(...)` 會自動分層）或 `label_dir(scene)`（直接取場景目錄）；**不要**自己寫 `data/labels/<scene>` 扁平路徑。
- **相機 mount 是「從 world 檔解析」不是讀 config**：A-1 視角生成的相機位移由 `load_wbt_mounts()` 從 `candidate_viewpoint_config.WORLD_FILE`（`worlds/ycb_supervisor_four_view_capture_multi.wbt`）的 `DEF UR5E_CAMERA translation` 讀取；`T_FLANGE_TO_D455_M` 只是 fallback。相機 mount **散在多個 world 檔**（A-1 來源／A-2 validator／armmove·multicam 拍攝 world／MoveIt URDF），改一定要**全部同步**再重跑，詳見 `PIPELINE.md` A-1 的「★★ 相機 mount 來源」。
- **srp 腳本 env 介面（易踩坑）**：`HULL_ROOT`／`SAM_ROOT`／`CAPTURES_ROOT` 吃**完整路徑**（`$PWD/data/eval/xxx`）；`OUT_ROOT` 與 `--root` 吃**根目錄名**（自動拼 `data/eval/`）；**例外 `fill_solid.py` 的 `HULL_ROOT` 吃名**。`run_scene.py`／`associate.py` 的 `CAPTURES_ROOT` 預設仍指舊 `data/captures`，跑 fast 資料要顯式設 `captures_fast`；且兩者 `scenes` 是 `nargs="+"`（要展開場景名，**不能空=全部**，其他 stage2 腳本 `nargs="*"` 才可空）。
- **srp 一律用 A-3 挑選的視角**：`srp/io/viewpoints.py` 的 `selected_view_names(N)` 是 Stage1/2 唯一視角來源；不要自寫 FPS。stage2 各方法預設 12 視角。
- **★ Stage1 hull 必須 `--num-views 12`（易踩坑,曾出大包）**：`run_scene.py` 的 `--num-views` **預設 `None`＝吃全部 34 拍攝視角**。過去 `srp_hull_mobilesamv2`／`_bf` 建置漏了 `--num-views 12`,結果 **hull 用 34 視角、stage2/評估卻用 12**,基準不一致(2026-08 修正)。建 hull 一律加 `--num-views 12`,讓 hull 與下游同視角。
- **★ allow_miss=1 為 semcluster 定案值**（2026-08 掃 0/1/2/3 定）：`--allow-miss 1`＝軟 hull,12 視角中容忍 1 個漏檢(≥11 落前景才保留)。救回硬交集(am0)砍掉的細長物(spoon/wood_blocks/windex);整體 @0.6 recall 0.904→**0.915**(分母=全放置物體,全遮擋計為漏)。⚠ **非全組最優**：occ5、stack3 硬交集(am0)較好、@0.7 亦然;am2/am3 過軟崩盤。標準管線用 am1,對相觸密集場(stack3/occ5)可考慮 am0。
- **build_meta（provenance,2026-08 起）**：`run_scene` 寫進 `hull.npz`、`voxel_sem_cluster` 寫進 `instances.npz/json`,記錄視角數/allow_miss/來源 root/thr 等。**用任何 hull/instances 前先讀 `build_meta` 確認怎麼來的**,不要再靠翻建置腳本推。
