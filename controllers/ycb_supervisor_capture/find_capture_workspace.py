#!/usr/bin/env python3
"""find_capture_workspace.py — 「拍攝時不碰撞」定義下的工作空間半徑分析。

工作空間 = 拍攝時手臂可達、且過程中手臂不碰到工作空間內物體 的桌面(半)球區域。
不涉及抓取。沿用 generate_candidate_viewpoints 的真實相機鏈 IK / 自碰撞 / 連桿 capsule。

二維掃描 (look-at 中心 x) × (cam_r),每格輸出:
  ① 手臂側(方位 90~270°)視角可達率
  ② FOV 上限          ws_r ≤ sin(half_FOV)·cam_r
  ③ 淨空上限          ws_r ≤ min_over(可達視角,連桿)[ 距離(連桿軸→中心) − capsule半徑 ] − margin
  ④ 最終 ws_r = min(②,③),並標出全表最大者(在視角可達率達門檻的格子中)。

margin = 手臂連桿外表面與工作空間球面之間要求保留的最小間隙(見對話定義)。
淨空計入:手臂 6 連桿 capsule + 末端執行器(相機球 + 夾爪 capsule,掛在 toolSlot 上)。
末端執行器幾何為估計值(常數在檔頭可調),建議對照 Webots 場景微調。

用法(Webots python):
  cd controllers/ycb_supervisor_capture
  /usr/bin/python3 find_capture_workspace.py
  /usr/bin/python3 find_capture_workspace.py --margin 0.05 \
      --cx 0.15 0.40 0.05 --camr 0.50 0.70 0.05 --min-reach 0.8
"""

import argparse
import json
import math
import os
from math import cos, sin, radians

import numpy as np

import generate_candidate_viewpoints as G

SIN_HALF_FOV = math.sin(1.4746 / 2.0)              # ≈0.672
ELEVATIONS_DEG = [20, 30, 45, 60, 75, 90]
AZIMUTHS_DEG = list(range(90, 271, 30))            # 手臂側(世界 +X=0)
LINK_RADII_M = [r / 1000.0 for r in G.LINK_RADII_MM]

# ── 末端執行器幾何(掛在已驗證的 webots toolSlot 上;可依 Webots 實況微調)──
# 相機(IntelRealsenseD455)建為球;夾爪(Robotiq 2F-140)建為自 toolSlot 沿外伸軸的 capsule。
R_CAMERA_M = 0.07            # 相機球半徑(D455 體寬 ~0.124 → 半徑含裕度)
GRIPPER_LEN_M = 0.16         # 夾爪自 toolSlot 外伸長度(base→指尖)
GRIPPER_RADIUS_M = 0.08      # 夾爪 capsule 半徑(140 夾爪張開時較寬)
# 夾爪在 toolSlot 局部座標的外伸方向:世界檔掛載 Pose Ry(π/2)·Gripper Rx(−π/2) 換算後 ≈ toolSlot +Y。
GRIPPER_AXIS_LOCAL = np.array([0.0, 1.0, 0.0])

OUT_PATH = os.path.join(G.DATA_VIEWPOINTS, "capture_workspace.json")


def viewpoints(C, cam_r):
    """手臂側半球視角 (仰角, 方位, 相機世界座標);仰角 90 退化為單點(方位=None)。"""
    pts = []
    for el in ELEVATIONS_DEG:
        if abs(el - 90) < 1e-9:
            pts.append((el, None, C + np.array([0.0, 0.0, cam_r])))
            continue
        e = radians(el)
        for az in AZIMUTHS_DEG:
            a = radians(az)
            pts.append((el, az, C + np.array([cam_r * cos(e) * cos(a),
                                              cam_r * cos(e) * sin(a),
                                              cam_r * sin(e)])))
    return pts


def ee_clearance_to_center(joints_rad, C):
    """末端執行器(相機球 + 夾爪 capsule)到中心 C 的淨空(世界 m)。
    用已驗證的 webots toolSlot/相機鏈定位。"""
    T_ts = G.webots_tool_slot_transform_world(joints_rad)        # toolSlot 世界 4x4(m)
    cam = G.webots_camera_transform_world(joints_rad)[:3, 3]     # 相機感測器世界座標
    g0 = T_ts[:3, 3]                                             # 夾爪 base(toolSlot 原點)
    g1 = (T_ts @ np.array([*(GRIPPER_LEN_M * GRIPPER_AXIS_LOCAL), 1.0]))[:3]  # 夾爪指尖
    cam_clear = float(np.linalg.norm(C - cam)) - R_CAMERA_M
    grip_clear = G._seg_seg_dist(C, C, g0, g1) - GRIPPER_RADIUS_M
    return min(cam_clear, grip_clear)


def clearance_to_center(joints_rad, C):
    """手臂 6 連桿 + 末端執行器(相機+夾爪)到中心 C 的最小淨空(世界 m)。"""
    frames = G.fk_joint_frames(joints_rad)
    pts = [f[:3, 3] / 1000.0 + G.ROBOT_BASE_M for f in frames]   # 7 個關節原點(世界 m)
    best = float("inf")
    for i in range(6):
        d = G._seg_seg_dist(C, C, pts[i], pts[i + 1])           # 點→線段距離
        best = min(best, d - LINK_RADII_M[i])
    return min(best, ee_clearance_to_center(joints_rad, C))


def solve_viewpoints(cx, cam_r):
    """回傳該 (cx,cam_r) 的可達視角明細 list[{el,az,joint_deg,clearance}]。"""
    C = np.array([cx, 0.0, 0.0])
    G.OBJECT_CENTER_M = C
    out = []
    for el, az, p in viewpoints(C, cam_r):
        j_deg = G.find_best_webots_ik(p, C)
        if j_deg is None:
            continue
        clr = clearance_to_center([radians(v) for v in j_deg], C)
        out.append({"elevation_deg": el, "azimuth_deg": az,
                    "joint_deg": [round(v, 4) for v in j_deg],
                    "clearance_m": round(float(clr), 4)})
    return out, len(viewpoints(C, cam_r))


def evaluate(cx, cam_r, margin):
    detail, ntotal = solve_viewpoints(cx, cam_r)
    nreach = len(detail)
    clearance = min((d["clearance_m"] for d in detail), default=float("inf"))
    reach = nreach / ntotal
    ws_fov = SIN_HALF_FOV * cam_r
    ws_clear = (clearance - margin) if nreach > 0 else 0.0
    ws = max(0.0, min(ws_fov, ws_clear))
    return {"cx": round(cx, 3), "cam_r": round(cam_r, 3), "reach": round(reach, 3),
            "n_reach": nreach, "n_view": ntotal,
            "ws_fov": round(ws_fov, 4), "ws_clear": round(ws_clear, 4),
            "ws": round(ws, 4), "bind": "FOV" if ws_fov <= ws_clear else "淨空"}


def frange(lo, hi, step):
    n = int(round((hi - lo) / step))
    return [round(lo + i * step, 4) for i in range(n + 1)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--margin", type=float, default=0.03, help="物體淨空裕度 m(預設 0.03)")
    ap.add_argument("--cx", type=float, nargs=3, default=[0.15, 0.45, 0.05],
                    metavar=("LO", "HI", "STEP"), help="look-at 中心 x 範圍")
    ap.add_argument("--camr", type=float, nargs=3, default=[0.50, 0.75, 0.05],
                    metavar=("LO", "HI", "STEP"), help="cam_r 範圍")
    ap.add_argument("--min-reach", type=float, default=0.8, dest="min_reach",
                    help="視角可達率門檻(只在達標格子中挑最大 ws)")
    args = ap.parse_args()

    cxs = frange(*args.cx)
    camrs = frange(*args.camr)
    print("拍攝工作空間掃描(不碰撞定義)")
    print(f"  手臂基座 {G.ROBOT_BASE_M.tolist()}  margin {args.margin} m")
    print(f"  仰角 {ELEVATIONS_DEG}  手臂側方位 {AZIMUTHS_DEG}")
    print(f"  中心x {cxs}")
    print(f"  cam_r {camrs}")
    print(f"  視角總數/格 = {len(viewpoints(np.zeros(3), 0.6))}\n")

    rows = []
    print(f"{'cx':>5} {'cam_r':>6} {'reach':>6} {'ws_FOV':>7} {'ws_淨空':>8} {'ws':>7} {'綁定':>5}")
    best = None
    for cx in cxs:
        for cam_r in camrs:
            r = evaluate(cx, cam_r, args.margin)
            rows.append(r)
            print(f"{r['cx']:>5} {r['cam_r']:>6} {r['reach']:>6.2f} "
                  f"{r['ws_fov']:>7.3f} {r['ws_clear']:>8.3f} {r['ws']:>7.3f} {r['bind']:>5}"
                  f"  ({r['n_reach']}/{r['n_view']})")
            if r["reach"] >= args.min_reach and (best is None or r["ws"] > best["ws"]):
                best = r

    print()
    best_views = []
    if best:
        print(f"★ 最佳(reach≥{args.min_reach}): 中心x={best['cx']}  cam_r={best['cam_r']}  "
              f"工作空間半徑 ws_r={best['ws']} m  (綁定={best['bind']}, reach={best['reach']})")
        # 記錄最佳操作點的可達視角明細(仰角/方位/joint_deg/淨空)
        best_views, _ = solve_viewpoints(best["cx"], best["cam_r"])
        print(f"  已記錄該操作點 {len(best_views)} 個可達視角(含 joint_deg)到輸出 JSON。")
    else:
        print(f"沒有任何格子達到 reach≥{args.min_reach};放寬 --min-reach 或調整範圍。")

    os.makedirs(G.DATA_VIEWPOINTS, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"margin_m": args.margin, "elevations_deg": ELEVATIONS_DEG,
                   "azimuths_deg": AZIMUTHS_DEG, "best": best,
                   "best_viewpoints": best_views, "grid": rows},
                  f, indent=2, ensure_ascii=False)
    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
