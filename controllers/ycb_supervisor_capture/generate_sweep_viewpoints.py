#!/usr/bin/env python3
"""generate_sweep_viewpoints.py — 視角重新規劃(掃描網格 + 可達性報告 + farthest-point 有序軌跡)。

依使用者指定的掃描網格生成候選相機視角,沿用 generate_candidate_viewpoints 的
真實 Webots UR5e + D455 數值 IK / 關節極限 / 桌面淨空 / capsule 自碰撞 / roll≈0 過濾,
不動既有 kinematics。輸出:
  ① 可達性報告(仰角 × 方位角,每格標可達的半徑)→ 決定拍攝範圍用。
  ② farthest-point 有序軌跡(確定性、可重現):每個長度-N 前綴都是近最佳散布子集(供 B1)。

已鎖定參數(由 find_capture_workspace 分析定出):
  look-at 中心 : 世界 [0.15, 0, 0](工作空間中心;ws_r≈0.355 m 半球)
  仰角         : 20 30 45 60 75 90 deg(10° 全不可達已剔除)
  方位角       : 90~270 每 30 deg(世界 +X = 0 deg;朝機械臂 -X 側,可達)
  半徑 cam_r   : 0.65 m(單一)

用法(在 /usr/bin/python3 / Webots python):
  cd controllers/ycb_supervisor_capture
  python generate_sweep_viewpoints.py                 # 只報告 + 排序,輸出 JSON
  python generate_sweep_viewpoints.py --report-only   # 只印可達性報告,不寫檔
"""

import argparse
import json
import math
import os
from math import cos, sin, radians

import numpy as np

import generate_candidate_viewpoints as G  # 重用 IK / 碰撞 / Webots 鏈 / 幾何 helper

# ── 已鎖定參數(由 find_capture_workspace 分析定出)──────────────────────
#   工作空間 = 以 [0.15,0,0] 為心、半徑 ws_r≈0.355 m 的半球(拍攝不碰撞定義)。
#   cam_r=0.65 框得住且手臂側視角可達率高(reach≈0.86);ws_r≈cam_r−0.29。
LOOK_AT_CENTER_M = [0.15, 0.0, 0.0]
ELEVATIONS_DEG = [20, 30, 45, 60, 75, 90]        # 剔除 10°(全不可達)
AZIMUTHS_DEG = list(range(90, 271, 30))          # 手臂側 90 120 150 180 210 240 270
RADII_M = [0.65]                                 # 單一拍攝半徑
WS_RADIUS_M = 0.355                              # 對應工作空間半徑(記錄用)

OUT_DIR = os.path.join(G.DATA_VIEWPOINTS)
OUT_PATH = os.path.join(OUT_DIR, "sweep_viewpoints_latest.json")


def candidate_position(center, el_deg, az_deg, r):
    el, az = radians(el_deg), radians(az_deg)
    return center + np.array([r * cos(el) * cos(az),
                              r * cos(el) * sin(az),
                              r * sin(el)])


def enumerate_grid(center):
    """回傳候選 list[dict];仰角 90(天頂)方位退化 → 每半徑只取一個。"""
    cands = []
    for r in RADII_M:
        for el in ELEVATIONS_DEG:
            azs = [None] if abs(el - 90) < 1e-9 else AZIMUTHS_DEG
            for az in azs:
                az_use = 0.0 if az is None else az
                pos = candidate_position(center, el, az_use, r)
                cands.append({"el": el, "az": (None if az is None else az),
                              "r": r, "pos": pos})
    return cands


def solve_reachable(cands, center):
    """對每個候選解 IK;回傳 reachable list(附 joints_deg + 實測 el/az)。"""
    center = np.array(center, dtype=float)
    reachable = []
    for c in cands:
        j_deg = G.find_best_webots_ik(c["pos"], center)
        if j_deg is None:
            continue
        el_m, az_m = G._elevation_azimuth(c["pos"])
        reachable.append({**c, "joint_deg": [round(v, 4) for v in j_deg],
                          "el_meas": round(el_m, 2), "az_meas": round(az_m, 2)})
    return reachable


def reachability_report(reachable):
    """印 仰角 × 方位角 表格,格內列出可達的半徑(由大到小)。"""
    by_cell = {}  # (el, az) -> set(radii)
    zenith = set()
    for c in reachable:
        if c["az"] is None:
            zenith.add(c["r"])
        else:
            by_cell.setdefault((c["el"], c["az"]), set()).add(c["r"])

    print("\n===== 可達性報告(格內=可達半徑 m;'.'=全不可達)=====")
    print("仰角\\方位  " + "  ".join(f"{a:>21}" for a in AZIMUTHS_DEG))
    for el in ELEVATIONS_DEG:
        if abs(el - 90) < 1e-9:
            continue
        cells = []
        for az in AZIMUTHS_DEG:
            rs = sorted(by_cell.get((el, az), set()), reverse=True)
            cells.append((",".join(f"{r:.2f}" for r in rs) if rs else ".").rjust(21))
        print(f"{el:>4} deg   " + "  ".join(cells))
    if zenith:
        print(f" 90 deg(天頂)  可達半徑: {','.join(f'{r:.2f}' for r in sorted(zenith, reverse=True))}")

    total_grid = len(RADII_M) * (len(AZIMUTHS_DEG) * (len(ELEVATIONS_DEG) - 1) + 1)
    print(f"\n候選總數 {total_grid}  →  可達 {len(reachable)}")
    print("各半徑可達數: " + ", ".join(
        f"{r:.2f}m:{sum(1 for c in reachable if c['r']==r)}" for r in RADII_M))
    print("各仰角可達數: " + ", ".join(
        f"{el}:{sum(1 for c in reachable if c['el']==el)}" for el in ELEVATIONS_DEG))


def farthest_point_order(reachable, center):
    """確定性 farthest-point 排序(角度距離,以 center 為心的方向向量)。
    種子 = 最高仰角→最大半徑→最小方位(天頂優先),平手以 index 破 → 完全可重現。"""
    center = np.array(center, dtype=float)
    def unit(c):
        d = c["pos"] - center
        return d / max(np.linalg.norm(d), 1e-12)
    dirs = [unit(c) for c in reachable]
    n = len(reachable)
    seed = max(range(n), key=lambda i: (reachable[i]["el"], reachable[i]["r"],
                                        -(reachable[i]["az"] if reachable[i]["az"] is not None else -1)))
    order = [seed]
    remaining = [i for i in range(n) if i != seed]
    while remaining:
        def min_ang(i):
            return min(math.degrees(math.acos(float(np.clip(np.dot(dirs[i], dirs[s]), -1, 1))))
                       for s in order)
        best = max(remaining, key=lambda i: (round(min_ang(i), 6), -i))
        order.append(best)
        remaining.remove(best)
    return [reachable[i] for i in order]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report-only", action="store_true", help="只印可達性報告,不寫 JSON")
    ap.add_argument("--center", type=float, nargs=3, default=LOOK_AT_CENTER_M)
    args = ap.parse_args()

    center = np.array(args.center, dtype=float)
    G.OBJECT_CENTER_M = center  # helper(碰撞/角度)讀此 module global

    print("視角掃描生成器")
    print(f"  look-at 中心 : {center.tolist()} m")
    print(f"  仰角         : {ELEVATIONS_DEG} deg")
    print(f"  方位角       : {AZIMUTHS_DEG} deg (世界 +X=0)")
    print(f"  半徑         : {RADII_M} m")

    cands = enumerate_grid(center)
    print(f"\n解 IK 中(候選 {len(cands)})...")
    reachable = solve_reachable(cands, center)
    reachability_report(reachable)

    if args.report_only:
        return
    ordered = farthest_point_order(reachable, center)
    records = []
    for k, c in enumerate(ordered, 1):
        records.append({"order": k, "elevation_deg": c["el"],
                        "azimuth_deg": c["az"], "radius_m": c["r"],
                        "elevation_meas_deg": c["el_meas"], "azimuth_meas_deg": c["az_meas"],
                        "camera_position_m": [round(float(v), 5) for v in c["pos"]],
                        "joint_deg": c["joint_deg"]})
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"center_m": center.tolist(), "cam_r_m": RADII_M[0],
            "ws_radius_m": WS_RADIUS_M, "grid": {
            "elevations_deg": ELEVATIONS_DEG, "azimuths_deg": AZIMUTHS_DEG, "radii_m": RADII_M},
            "n_reachable": len(records), "viewpoints": records}, f, indent=2, ensure_ascii=False)
    print(f"\n→ 有序軌跡 {len(records)} 視角寫入 {OUT_PATH}")


if __name__ == "__main__":
    main()
