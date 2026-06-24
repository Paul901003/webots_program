#!/usr/bin/env python3
"""find_manipulation_workspace.py — 量測 UR5e + 2F-140 夾爪在桌面上的「可達半徑」。

目的:回答「工作空間球半徑」的物理天花板 —— 手臂以頂向抓取(夾爪朝下、自由偏航)
能搆到桌面上多大一圈。輸出供決定拍攝半徑 cam_r 用(cam_r ≈ ws_r / sin(half_FOV))。

方法(完全沿用已驗證的解析 FK/IK,不用相機 toolSlot 那條鏈):
  ① 抓取目標 = 桌面點 p=[x,y,z_grasp],夾爪朝下 → flange +Z 指世界 -Z,
     flange 位置 = p + [0,0,TCP_LEN](TCP 沿 flange +Z 外伸 0.177 m,故 flange 在抓取點上方)。
  ② 偏航自由:掃多個 yaw,任一可解即可達。
  ③ 對每個 flange 目標(轉到 base 座標、mm)解析 IK(8 解)→ 取通過 關節極限 + is_collision_free 者。
  ④ 桌面極座標掃描(半徑環 × 方位)→ 報告每方位最大可達半徑、整圈對稱可達半徑。

用法(在 /usr/bin/python3 / Webots python):
  cd controllers/ycb_supervisor_capture
  python find_manipulation_workspace.py [--center 0.35 0 0] [--z-grasp 0.04]
"""

import argparse
import json
import math
import os
from math import cos, sin, radians

import numpy as np

import generate_candidate_viewpoints as G  # 重用 IK / is_collision_free / 常數

TCP_LEN_M = 0.176962          # FLANGE_TO_TCP_TRANSLATION_M[2](官方 2F-140 指墊中點)
R_RINGS_M = np.round(np.arange(0.0, 0.551, 0.025), 4)   # 桌面半徑環
AZIMUTHS_DEG = list(range(0, 360, 15))                  # 全 360 度(世界 +X=0)
YAWS_DEG = list(range(0, 180, 60))                      # 夾爪偏航(2 指對稱)
# 傾斜接近:接近方向在離垂直(朝下)TILT_MAX 內的圓錐;θ=傾角, ψ=傾斜方位
TILT_MAX_DEG = 30
TILT_STEP_DEG = 15
TILT_AZIMUTHS_DEG = list(range(0, 360, 45))

# 抓取用 UR5e 真實硬體關節極限(±360°),不可用 G.JOINT_LIMITS_DEG(那是相機拍攝調過的受限範圍)
GRASP_LIMITS_DEG = [(-360.0, 360.0)] * 6


def within_grasp_limits(joints_deg):
    return all(lo <= j <= hi for j, (lo, hi) in zip(joints_deg, GRASP_LIMITS_DEG))

OUT_PATH = os.path.join(G.DATA_VIEWPOINTS, "manipulation_workspace.json")


def approach_dirs():
    """接近單位向量集合:θ=0(正下)+ 各傾角×傾斜方位的圓錐。"""
    dirs = [np.array([0.0, 0.0, -1.0])]
    for th in range(TILT_STEP_DEG, TILT_MAX_DEG + 1, TILT_STEP_DEG):
        t = radians(th)
        for ps in TILT_AZIMUTHS_DEG:
            p = radians(ps)
            dirs.append(np.array([sin(t) * cos(p), sin(t) * sin(p), -cos(t)]))
    return dirs


_APPROACH = approach_dirs()


def flange_target_base_mm(p_world, z_grasp, approach, yaw_rad):
    """flange 目標 → robot-base 座標 4x4(平移 mm)。flange +Z=approach;
    flange = TCP點 - TCP_LEN*approach;偏航繞 approach 軸。base 與 world 同向(僅平移)。"""
    fz = approach / max(np.linalg.norm(approach), 1e-12)
    ref = np.array([cos(yaw_rad), sin(yaw_rad), 0.0])
    fx = ref - np.dot(ref, fz) * fz
    n = np.linalg.norm(fx)
    fx = (fx / n) if n > 1e-6 else np.array([1.0, 0.0, 0.0])
    fy = np.cross(fz, fx)
    R = np.column_stack([fx, fy, fz])
    tcp_world = np.array([p_world[0], p_world[1], z_grasp])
    flange_world = tcp_world - TCP_LEN_M * fz
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = (flange_world - G.ROBOT_BASE_M) * 1000.0
    return T


def reachable(p_world, z_grasp):
    """接近圓錐內任一方向 + 任一 yaw 能解析 IK 且通過 極限 + 自碰撞 → 可達。"""
    for approach in _APPROACH:
        for yaw in YAWS_DEG:
            T = flange_target_base_mm(p_world, z_grasp, approach, radians(yaw))
            try:
                sols = G.IK(T.tolist())
            except Exception:
                continue
            for sol in sols:
                if not all(math.isfinite(v) for v in sol):
                    continue
                if not within_grasp_limits([math.degrees(v) for v in sol]):
                    continue
                if not G.is_collision_free(sol):
                    continue
                return True
    return False


def reach_cart(x, y, z_grasp, _cache):
    key = (round(x, 4), round(y, 4))
    if key not in _cache:
        _cache[key] = reachable(np.array([x, y, 0.0]), z_grasp)
    return _cache[key]


def az_intervals(center, z_grasp, cache):
    """每方位的(內邊界, 外邊界)可達半徑;月牙有內外兩個邊界。回傳 {az:(in,out)};不可達=(None,None)。"""
    center = np.array(center, dtype=float)
    per_az = {}
    grid = {}
    for az in AZIMUTHS_DEG:
        a = radians(az)
        rs = [float(R) for R in R_RINGS_M
              if reach_cart(center[0] + R * cos(a), center[1] + R * sin(a), z_grasp, cache)]
        for R in R_RINGS_M:
            grid[(az, float(R))] = reach_cart(center[0] + float(R) * cos(a),
                                              center[1] + float(R) * sin(a), z_grasp, cache)
        per_az[az] = (min(rs), max(rs)) if rs else (None, None)
    return per_az, grid


def max_disk_radius(cx, z_grasp, cache):
    """以 (cx,0) 為心,最大『整圈內全可達』的對稱圓盤半徑(含圓心)。"""
    if not reach_cart(cx, 0.0, z_grasp, cache):
        return 0.0
    best = 0.0
    for R in R_RINGS_M:
        if R == 0.0:
            continue
        a_ok = all(reach_cart(cx + float(R) * cos(radians(az)),
                              float(R) * sin(radians(az)), z_grasp, cache)
                   for az in AZIMUTHS_DEG)
        if a_ok:
            best = float(R)
        else:
            break
    return best


def report(center, z_grasp, per_az, grid, cache):
    print(f"\n===== 桌面抓取可達性(center={list(center)} z_grasp={z_grasp}m TCP={TCP_LEN_M}m)=====")
    print("方位定義:世界 +X=0°(0°=遠離手臂,180°=朝向手臂基座)\n")
    header = "方位\\R  " + "".join(f"{R:>5.2f}" for R in R_RINGS_M)
    print(header)
    for az in AZIMUTHS_DEG:
        row = "".join("    ●" if grid[(az, float(R))] else "    ·" for R in R_RINGS_M)
        print(f"{az:>4}°  {row}")

    print("\n--- 每方位可達半徑區間(內邊界~外邊界, m;'—'=該方位全不可達)---")
    for az in AZIMUTHS_DEG:
        lo, hi = per_az[az]
        s = f"{lo:.3f}~{hi:.3f}" if lo is not None else "—"
        print(f"  {az:>4}°: {s}")

    # 以 0.35 為心的對稱圓盤(預期 0,因圓心不可達)
    disk_035 = max_disk_radius(center[0], z_grasp, cache)
    # 掃描圓心找最大對稱圓盤
    print("\n--- 圓心沿 X 軸掃描:最大『整圈全可達』對稱圓盤半徑 ---")
    best_cx, best_r = None, -1.0
    for cx in np.round(np.arange(0.0, 0.601, 0.025), 4):
        r = max_disk_radius(float(cx), z_grasp, cache)
        mark = ""
        if r > best_r:
            best_r, best_cx = r, float(cx); mark = "  ←最佳"
        print(f"  center x={cx:.3f}: 對稱可達半徑 {r:.3f} m{mark}")

    sin_half_fov = math.sin(1.4746 / 2.0)
    print(f"\n以 [{center[0]},0,0] 為心的對稱圓盤半徑 : {disk_035:.3f} m(圓心不可達→受限)")
    print(f"最佳對稱工作空間: 圓心 x={best_cx:.3f} m, 半徑 {best_r:.3f} m")
    if best_r > 0:
        print(f"  → 框得住此工作空間的最小 cam_r = {best_r/sin_half_fov:.3f} m"
              f"(從工作空間中心量)")
    return {"disk_center035": disk_035, "best_center_x": best_cx, "best_radius": best_r}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--center", type=float, nargs=3, default=[0.35, 0.0, 0.0])
    ap.add_argument("--z-grasp", type=float, default=0.04, dest="z_grasp",
                    help="抓取點高度(m,桌面=0);越低越難搆")
    args = ap.parse_args()
    center = np.array(args.center, dtype=float)
    G.OBJECT_CENTER_M = center  # is_collision_free 不用,但保持一致

    print("桌面抓取可達半徑量測")
    print(f"  手臂基座     : {G.ROBOT_BASE_M.tolist()} m")
    print(f"  工作空間中心 : {center.tolist()} m")
    print(f"  抓取高度     : {args.z_grasp} m")
    print(f"  半徑環       : {R_RINGS_M[0]}~{R_RINGS_M[-1]} step 0.025 m")
    print(f"  方位 / 偏航  : {len(AZIMUTHS_DEG)} 方位 × {len(YAWS_DEG)} yaw")
    print(f"  傾斜接近     : 離垂直 ≤{TILT_MAX_DEG}° (step {TILT_STEP_DEG}°), {len(_APPROACH)} 個接近方向")

    cache = {}
    per_az, grid = az_intervals(center, args.z_grasp, cache)
    summary = report(center, args.z_grasp, per_az, grid, cache)

    os.makedirs(G.DATA_VIEWPOINTS, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"center_m": center.tolist(), "z_grasp_m": args.z_grasp,
                   "tcp_len_m": TCP_LEN_M,
                   "per_azimuth_interval_m": {str(k): v for k, v in per_az.items()},
                   "summary_m": summary}, f, indent=2, ensure_ascii=False)
    print(f"\n→ {OUT_PATH}")


if __name__ == "__main__":
    main()
