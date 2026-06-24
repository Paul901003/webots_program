# 視角／工作空間規劃工具說明

本目錄下用於**重新規劃 Stage 0 相機拍攝視角**的工具集。背景與已定方向見 `plan/`
(不可違背準則)與專案記憶 `viewpoint-redesign`。

## 共同基礎與環境

- 直譯器一律用 **Webots python**:`/usr/bin/python3`(需 numpy / scipy)。
- 全部 `import generate_candidate_viewpoints as G`,重用其**已驗證**的:
  真實 Webots UR5e + D455 數值 IK(`find_best_webots_ik`)、解析 IK(`IK`)、
  自碰撞 + 桌面淨空(`is_collision_free`)、連桿 capsule 幾何(`LINK_RADII_MM`)、
  FK / toolSlot / 相機鏈、座標常數(`ROBOT_BASE_M` 等)。
- 機械臂基座 `[-0.4, 0, 0]`;相機 HFOV 1.4746 rad(半視角 ≈42.2°,`sin≈0.672`)。
- 輸出統一寫到 `data/viewpoints/`。

## 關鍵定義

- **cam_r**:相機繞工作空間中心的半球半徑(拍攝距離)。視角只取**手臂側**方位 90°–270°(世界 +X=0°)。
- **工作空間半徑 ws_r**:拍攝時手臂可達且**過程不碰撞**的(半)球區域半徑。受兩項上限約束:
  - FOV:`ws_r ≤ 0.672 · cam_r`(框得住)。
  - 淨空:`ws_r ≤ min_over(可達視角, 手臂連桿/相機/夾爪)[ 距離→中心 − capsule半徑 ] − margin`。
- **margin(物體淨空裕度)**:手臂/EE 外表面與工作空間球面之間要求保留的最小空氣間隙
  (吸收擺放誤差、mesh 細節、控制誤差)。`0` = 可擦過物體;預設 `0.03 m`。

---

## 工具一覽

### 1. `generate_candidate_viewpoints.py`(舊有,基礎庫)
A-1 候選視角生成器。被其他工具當函式庫 import。可單獨跑產 `candidate_viewpoints.json`。
參數在 `candidate_viewpoint_config.py`。

### 2. `find_manipulation_workspace.py` — 桌面**抓取**可達半徑(探索用,非主線)
夾爪頂向/傾斜接近的 IK 掃描,量手臂能在桌面抓取多大範圍。
> 註:後來確認工作空間應以**拍攝不碰撞**定義,非抓取;此工具留作參考。

```bash
/usr/bin/python3 find_manipulation_workspace.py [--center 0.35 0 0] [--z-grasp 0.04]
```
輸出 `data/viewpoints/manipulation_workspace.json`:每方位可達半徑區間 + 圓心掃描 + 最佳對稱圓盤。

### 3. `find_capture_workspace.py` — ★ 主線:拍攝工作空間半徑分析
二維掃描 **(look-at 中心 x) × (cam_r)**,在「拍攝不碰撞」定義下找最佳 `(cx, cam_r, ws_r)`。
淨空計入**手臂 6 連桿 + 相機球 + 夾爪 capsule**(EE 幾何常數在檔頭可調,建議對照 Webots 微調)。

```bash
/usr/bin/python3 find_capture_workspace.py
/usr/bin/python3 find_capture_workspace.py --margin 0.05 --min-reach 0.7
/usr/bin/python3 find_capture_workspace.py --cx 0.20 0.40 0.05 --camr 0.55 0.70 0.05
```
輸出 `data/viewpoints/capture_workspace.json`:
- `grid`:每格 `reach / ws_fov / ws_clear / ws / bind`。
- `best`:reach 達門檻中 ws 最大的操作點。
- **`best_viewpoints`:該最佳操作點的可達視角明細(elevation/azimuth/`joint_deg`/clearance)。** ← 可達視角有被記錄。

### 3b. `find_capture_workspace.py` 附註：EE 幾何常數
淨空恆為綁定者,故 `ws_r` 幾乎完全由檔頭 EE 幾何決定(夾爪長 0.16/半徑 0.08/外伸 +Y、相機球 0.07)。
鎖定前務必對照 Webots 場景核對夾爪朝向與尺寸。

### 4. `generate_sweep_viewpoints.py` — 產可執行的有序視角軌跡
依掃描網格(仰角 × 方位 90–270 × 半徑)生成候選 → IK 過濾 → **farthest-point 有序軌跡**
(確定性、可重現;前 N 個前綴即合理子集,供實驗 B1 視角數收斂)。

```bash
/usr/bin/python3 generate_sweep_viewpoints.py              # 報告 + 排序 + 寫檔
/usr/bin/python3 generate_sweep_viewpoints.py --report-only  # 只印可達性報告
```
輸出 `data/viewpoints/sweep_viewpoints_latest.json`:有序 `viewpoints`,每筆含
`order / elevation / azimuth / radius / camera_position_m / joint_deg`。← 給機器人執行用的軌跡。

### 5. `sweep_to_validated.py` — 軌跡 → A-2/A-4 驗證器輸入
把 `sweep_viewpoints_latest.json` 轉成 `validated_viewpoints_latest.json`
(`{"validated":[{id, joint_deg, radius_m, order, elevation_deg, azimuth_deg, ...}]}`),
供 `ycb_viewpoint_validator/` 的 MoveIt 驗證器消費。

```bash
/usr/bin/python3 sweep_to_validated.py
```
接著(需 ROS2 + planning bridge):
```bash
# 終端機1:bridge
ros2 launch ur5e_2f140_planning planning_bridge_launch.py
# 終端機2:A-2b 逐視角可達(工作空間球當碰撞障礙)→ A-4 視角間繞球路徑
cd ~/webots_program/controllers/ycb_viewpoint_validator
/usr/bin/python3 validate_workspace_sphere.py --x-offset 0.15 --ws-offset 0.295
/usr/bin/python3 plan_viewpoint_paths.py   --x-offset 0.15 --ws-offset 0.295 --vel-scale 0.2 --acc-scale 0.2
```
> `--x-offset 0.15`=工作空間中心世界 x;`--ws-offset 0.295`=cam_r 0.65 − ws_r 0.355。

### 視覺化(`controllers/workspace_supervisor/`)
Webots 線框球已更新為現況:**拍攝球 cam_r=0.65**、**物體工作球 ws_r=0.355**,皆以 **[0.15,0,0]** 為心
(另保留 UR5e 手臂伸距 0.85 包絡)。參數在 `workspace_supervisor/config.py`。

---

## 建議流程

1. `find_capture_workspace.py` → 定 **(cx, cam_r, ws_r)** 操作點(看 ws 被 FOV 還是淨空綁定)。
2. 用選定的 cx / cam_r 對齊 `generate_sweep_viewpoints.py` 的中心與半徑 → 產**有序可執行軌跡**。
3. 軌跡交給拍攝端(`ycb_path_executor`)執行,每視角輸出 `view_XX.png` + `view_XX_pose.json`。

> 各檔的可達視角(joint_deg)記錄狀況:
> `find_capture_workspace`=記錄最佳操作點;`generate_sweep_viewpoints`=記錄整條有序軌跡;
> `find_manipulation_workspace`=只記區間/摘要,不記個別視角。
