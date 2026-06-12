# CLAUDE.md — 專案說明

## 專案概述

在 Webots 模擬環境中，使用 UR5e + Robotiq 2F-140 夾爪對 YCB 物件進行多視角影像擷取。
目標是產生用於訓練物件辨識模型的資料集（RGB 影像 + mask）。

**完整拍攝流程：** `PIPELINE.md`（詳細指令與參數）

---

## 目錄結構

```
webots_program/
├── PIPELINE.md                        ← 完整拍攝流程（A-0 ~ A-5）
├── controllers/                       ← Webots controller，每個子目錄即一個 controller
│   ├── ycb_supervisor/                ← 共用基礎模組（非獨立 controller）
│   │   ├── config.py                  ← 全域參數（手臂速度、spawn 設定等）
│   │   ├── ycb_scanner.py             ← 掃描可用 YCB 物件
│   │   └── ycb_geometries.json        ← 各物件幾何尺寸資料
│   │
│   ├── ycb_viewpoint_validator/       ← A-0/A-2/A-3/A-4：視角前處理工具集
│   │   ├── ycb_viewpoint_validator.py ← A-2 主 controller（Webots）
│   │   ├── scan_x_offset.py           ← A-0 掃描最佳拍攝中心
│   │   ├── select_validated_viewpoints.py ← A-3 選取分布最廣子集
│   │   ├── plan_viewpoint_paths.py    ← A-4 規劃路徑（需 ROS2 bridge）
│   │   └── collect_validated_viewpoints.py
│   │
│   ├── ycb_supervisor_capture/        ← 候選視角生成（A-1）
│   │   ├── generate_candidate_viewpoints_multi.py ← A-1 主腳本
│   │   ├── candidate_viewpoint_config.py ← 半徑、仰角、方位角參數
│   │   └── *.json                     ← 本機快取（data/ 才是正本）
│   │
│   ├── ycb_path_executor/             ← A-5：在 Webots 執行路徑並拍攝
│   │   └── ycb_path_executor.py
│   │
│   ├── ycb_supervisor_ros2_test/      ← ROS2 bridge 共用工具
│   │   ├── ros2_bridge_utils.py       ← launch/wait/plan/stop bridge
│   │   └── ros2_bridge_subprocess.py
│   │
│   ├── ycb_supervisor_four_view_multi/ ← 舊版固定四視角（多物件）
│   ├── ycb_supervisor_four_view_single/← 舊版固定四視角（單物件）
│   ├── ur5e_test_controller/          ← 手臂基本動作測試
│   ├── ur5e_auto_capture_controller/  ← 配合 realsense 實機擷取
│   ├── realsense_auto_capture_controller/
│   ├── workspace_supervisor/          ← 視覺化可達工作空間球體
│   └── visual_hull_check_supervisor/  ← Visual hull 驗證
│
├── worlds/                            ← Webots 場景檔（.wbt）
│   ├── ycb_viewpoint_validator_multi.wbt  ← A-2 用
│   ├── ycb_path_executor_multi.wbt        ← A-5 用
│   ├── ycb_supervisor_capture*.wbt        ← 舊版擷取場景
│   └── ur5e+gripper.wbt                   ← 手臂基本測試
│
├── data/                              ← 所有輸入輸出資料（不進 git）
│   ├── viewpoints/                    ← 各步驟視角 JSON（詳見 PIPELINE.md）
│   ├── scene_plans/                   ← 多物件場景配置
│   └── captures/                     ← A-5 輸出影像
│
├── ros2_ws/                           ← ROS2 workspace
│   └── src/
│       ├── ur5e_2f140_planning/       ← MoveIt 規劃 bridge（planning_bridge_launch.py）
│       └── ros2_robotiq_gripper/      ← 夾爪驅動
│
├── protos/                            ← Webots 自訂 proto
│   ├── IntelRealsenseD455.proto
│   └── ur5e_with_140gripper.proto
│
├── urdfs/ycb_assets/                  ← YCB 物件 URDF/mesh（ASSET_BASE 指向此處）
├── tools/                             ← 離線評估工具（evaluate_masks.py 等）
└── Grounded-Segment-Anything/         ← GSA 子模組（mask 生成）
```

---

## 模組依賴關係

```
ycb_supervisor/config.py          ← 被所有 controller 引用的共用參數
ycb_supervisor_ros2_test/ros2_bridge_utils.py ← A-2/A-4 使用的 bridge 工具
ycb_supervisor_capture/candidate_viewpoint_config.py ← A-1/A-2/A-4 共用視角設定
```

---

## 環境

| 項目 | 值 |
|------|----|
| Python | 3.8.10（Webots 內建），`/usr/bin/python3` |
| Webots | 2023.x |
| ROS2 | Jazzy |
| 機器人底座（world frame） | `[-0.4, 0.0, 0.0]` m |
| 最佳拍攝中心 x_offset | **0.35 m** |
| 方位角步數 | 8（每 45°） |
| 選取視角數 | 12（`NUM_OUTPUT_POSES`） |
| 工作空間偏移量 | `ws_offset=0.30 m`；`sphere_r = cam_r - ws_offset`（cam_r=0.65 → 工作空間半徑 0.35 m） |

---

## 常見操作

```bash
# 啟動 ROS2 Planning Bridge（A-0、A-4 需要）
source /opt/ros/jazzy/setup.bash && source ~/webots_program/ros2_ws/install/setup.bash
ros2 launch ur5e_2f140_planning planning_bridge_launch.py

# A-2：Webots 驗證（無需外部 bridge，controller 自動啟動）
VALIDATOR_ARGS="--multi --x-offset 0.35" webots worlds/ycb_viewpoint_validator_multi.wbt

# A-5：執行拍攝
webots worlds/ycb_path_executor_multi.wbt
```

---

## 重要慣例

- **`_latest.json`** 是各步驟的最新輸出，供下一步讀取；**具名 JSON** 保留歷史記錄不覆蓋。
- `VALIDATOR_ARGS` 環境變數優先於 `.wbt` 的 `controllerArgs`，修改參數時無需改 world 檔。
- `ycb_supervisor/` 不是獨立 controller，是共用函式庫，被其他 controller 透過 `sys.path` 引入。
- 所有 Python 腳本使用 `/usr/bin/python3`（非 venv）以確保 Webots 相容性。
