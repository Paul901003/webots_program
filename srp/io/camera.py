#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""camera.py — 相機內外參工具(Stage 0/1 共用)。

把 Webots 拍攝端的 pose.json(相機位置 + rpy,body 慣例)轉成 carve_visual_hull 要的
world→camera 外參(OpenCV 慣例)。換算沿用既有驗證過的關係:
    R_w2c = BODY_TO_OPENCV @ R_body.T ,  t = -R_w2c @ C
(與 instance_hull/carve_instances.make_transform、hull_common.project 完全一致)。
內參 fx = W / (2 tan(HFOV/2)),主點置中。介面預留外參為可替換輸入(sim-to-real)。
"""

import json
import math
from pathlib import Path

import numpy as np

HFOV_RAD = 1.4746
BODY_TO_OPENCV = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float64)


def rpy_to_R(roll, pitch, yaw):
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=np.float64)


def load_pose(pose_path):
    """讀 pose.json → (C(3,), R_body(3,3))。相容 {camera:{...}} 與扁平兩種。"""
    meta = json.loads(Path(pose_path).read_text(encoding="utf-8"))
    if "position_m" not in meta and isinstance(meta.get("camera"), dict):
        meta = meta["camera"]
    p = meta["position_m"]
    C = np.array([p["x"], p["y"], p["z"]], dtype=np.float64)
    r = meta["rotation_rpy_rad"]
    return C, rpy_to_R(r["roll"], r["pitch"], r["yaw"])


def intrinsics(W, H, hfov_rad=HFOV_RAD):
    """針孔 K。fx=fy=W/(2 tan(hfov/2)),主點置中。"""
    fx = W / (2.0 * math.tan(hfov_rad / 2.0))
    return np.array([[fx, 0, W / 2.0], [0, fx, H / 2.0], [0, 0, 1.0]], dtype=np.float64)


def pose_to_w2c(C, R_body):
    """相機位姿(body 慣例)→ world→camera(OpenCV)。回傳 (R_w2c, t)。"""
    R_w2c = BODY_TO_OPENCV @ R_body.T
    t = -R_w2c @ np.asarray(C, dtype=np.float64)
    return R_w2c, t
