#!/usr/bin/env python3
"""
generate_candidate_viewpoints.py — candidate viewpoint generator for UR5e + YCB capture.

Pipeline
--------
1. Sample camera positions on a hemisphere above the target object.
2. For each sample, aim the configured camera local axis at target.
3. Numerically solve joints against the actual Webots UR5e toolSlot + D455
   Camera transform.
4. Filter by joint limits, table clearance, and capsule-based self-collision
   between all non-adjacent link pairs.
5. Export all valid candidate poses to candidate_viewpoints.json for Webots
   collision validation.

Usage
-----
    cd controllers/ycb_supervisor_four_view
    python generate_candidate_viewpoints.py
"""

import argparse
import json
import math
import sys
import os
import re
from math import cos, sin, pi, isfinite

import numpy as np
from scipy.optimize import least_squares

# ── path setup ──────────────────────────────────────────────────────────────
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
CONTROLLERS_DIR = os.path.dirname(SCRIPT_DIR)
KINEMATICS_DIR  = os.path.join(CONTROLLERS_DIR, "ur5e_controller", "my_ur_kinematics")
sys.path.insert(0, KINEMATICS_DIR)

from Inverse_Kinematics import IK  # noqa: E402  analytical closed-form, 8 solutions
import ur_config                   # noqa: E402
import candidate_viewpoint_config as config  # noqa: E402

# ── DH parameters (from ur_config) ──────────────────────────────────────────
_a     = ur_config.UR5_DH_param.a      # link lengths (mm)
_alpha = ur_config.UR5_DH_param.alpha  # link twists  (rad)
_d     = ur_config.UR5_DH_param.d      # link offsets (mm)


def _axis_angle_rotation(axis: list, angle: float) -> np.ndarray:
    axis = np.array(axis, dtype=float)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    one_minus_c = 1.0 - c
    return np.array([
        [c + x * x * one_minus_c,
         x * y * one_minus_c - z * s,
         x * z * one_minus_c + y * s],
        [y * x * one_minus_c + z * s,
         c + y * y * one_minus_c,
         y * z * one_minus_c - x * s],
        [z * x * one_minus_c - y * s,
         z * y * one_minus_c + x * s,
         c + z * z * one_minus_c],
    ], dtype=float)


def _fixed_transform(translation: list | np.ndarray,
                     rotation: np.ndarray | None = None) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, 3] = np.array(translation, dtype=float)
    if rotation is not None:
        T[:3, :3] = rotation
    return T


def _extract_node_block(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text)
    if match is None:
        return None
    start = text.find("{", match.end())
    if start < 0:
        return None

    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:index]
    return None


def _parse_vec_field(block: str, field_name: str,
                     fallback: list[float]) -> np.ndarray:
    match = re.search(
        rf"^\s*{re.escape(field_name)}\s+"
        r"([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)",
        block,
        flags=re.MULTILINE,
    )
    if match is None:
        return np.array(fallback, dtype=float)
    return np.array([float(match.group(i)) for i in range(1, 4)], dtype=float)


def _parse_rotation_field(block: str, fallback: list[float]) -> np.ndarray:
    match = re.search(
        r"^\s*rotation\s+"
        r"([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)",
        block,
        flags=re.MULTILINE,
    )
    if match is None:
        return _axis_angle_rotation(fallback[:3], fallback[3])
    values = [float(match.group(i)) for i in range(1, 5)]
    return _axis_angle_rotation(values[:3], values[3])


def load_wbt_mounts() -> dict:
    world_path = os.path.normpath(os.path.join(SCRIPT_DIR, config.WORLD_FILE))
    fallback_camera_rotation = [0.0, 0.0, 1.0, 1.5708]
    mounts = {
        "world_path": world_path,
        "robot_base_m": np.array(config.ROBOT_BASE_M, dtype=float),
        "tool_to_d455_m": np.array(config.T_FLANGE_TO_D455_M, dtype=float),
        "R_tool_to_d455": np.array(config.R_FLANGE_TO_CAM, dtype=float),
        "source": "config fallback",
    }

    try:
        with open(world_path, "r", encoding="utf-8") as file:
            text = file.read()
    except OSError:
        return mounts

    robot_block = _extract_node_block(
        text,
        rf"DEF\s+{re.escape(config.UR5E_DEF)}\s+UR5e\b",
    )
    camera_block = _extract_node_block(
        text,
        rf"DEF\s+{re.escape(config.CAMERA_DEF)}\s+IntelRealsenseD455\b",
    )

    if robot_block is not None:
        mounts["robot_base_m"] = _parse_vec_field(
            robot_block,
            "translation",
            config.ROBOT_BASE_M,
        )
    if camera_block is not None:
        mounts["tool_to_d455_m"] = _parse_vec_field(
            camera_block,
            "translation",
            config.T_FLANGE_TO_D455_M,
        )
        mounts["R_tool_to_d455"] = _parse_rotation_field(
            camera_block,
            fallback_camera_rotation,
        )
    if robot_block is not None or camera_block is not None:
        mounts["source"] = "world file"
    return mounts


WBT_MOUNTS = load_wbt_mounts()

# ── editable parameters from candidate_viewpoint_config.py ───────────────────
ROBOT_BASE_M     = WBT_MOUNTS["robot_base_m"]
OBJECT_CENTER_M  = np.array(config.OBJECT_CENTER_M, dtype=float)
TABLE_Z_M        = config.TABLE_Z_M
LINK_CLEARANCE_M = config.LINK_CLEARANCE_M

WEBOTS_TOOL_SLOT_TRANSLATION_M = np.array(config.WEBOTS_TOOL_SLOT_TRANSLATION_M, dtype=float)
_R_flange_to_cam = WBT_MOUNTS["R_tool_to_d455"]
_t_flange_to_d455 = WBT_MOUNTS["tool_to_d455_m"]
_t_d455_to_sensor = np.array(config.T_D455_TO_SENSOR_M, dtype=float)
_t_flange_to_cam = _t_flange_to_d455 + _R_flange_to_cam @ _t_d455_to_sensor
CAMERA_AIM_AXIS_LOCAL = np.array(config.CAMERA_AIM_AXIS_LOCAL, dtype=float)
CAMERA_UP_AXIS_LOCAL = np.array(config.CAMERA_UP_AXIS_LOCAL, dtype=float)
WORLD_UP_AXIS = np.array(config.WORLD_UP_AXIS, dtype=float)
WORLD_ROLL_FALLBACK_AXIS = np.array(config.WORLD_ROLL_FALLBACK_AXIS, dtype=float)
CAMERA_ROLL_WEIGHT = config.CAMERA_ROLL_WEIGHT
MAX_CAMERA_ROLL_ERROR_DEG = config.MAX_CAMERA_ROLL_ERROR_DEG

T_FLANGE_TO_CAM        = np.eye(4, dtype=float)
T_FLANGE_TO_CAM[:3,:3] = _R_flange_to_cam
T_FLANGE_TO_CAM[:3, 3] = _t_flange_to_cam
T_CAM_TO_FLANGE        = np.linalg.inv(T_FLANGE_TO_CAM)
MAX_WEBOTS_POSITION_ERROR_M = 0.02
MAX_WEBOTS_TARGET_ERROR_DEG = 2.0
MAX_CAMERA_RAY_MISS_M = 0.005

HEMISPHERE_RADIUS_M  = config.HEMISPHERE_RADIUS_M
ELEVATION_ANGLES_DEG = config.ELEVATION_ANGLES_DEG
AZIMUTH_STEPS        = config.AZIMUTH_STEPS

JOINT_LIMITS_DEG = config.JOINT_LIMITS_DEG
REFERENCE_DEG    = config.REFERENCE_DEG
NUM_OUTPUT_POSES = config.NUM_OUTPUT_POSES
LINK_RADII_MM    = config.LINK_RADII_MM
EXISTING_POSES_DEG = config.EXISTING_POSES_DEG


# =============================================================================
# FK — standard DH, pure numpy ndarray (no deprecated np.matrix)
# =============================================================================

def _dh_matrix(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """4x4 homogeneous DH step: Rot_z(theta) * Trans_z(d) * Trans_x(a) * Rot_x(alpha)."""
    ct, st = cos(theta), sin(theta)
    ca, sa = cos(alpha), sin(alpha)
    return np.array([
        [ct,  -st * ca,   st * sa,  a * ct],
        [st,   ct * ca,  -ct * sa,  a * st],
        [ 0,        sa,       ca,   d     ],
        [ 0,         0,        0,   1     ],
    ], dtype=float)


def fk_joint_frames(joints_rad: list) -> list:
    """Return 7 cumulative DH transforms in robot-base frame, translation in mm.

    frames[0] = identity (base origin), frames[i] = T_{0,i} for i = 1..6.
    """
    T = np.eye(4, dtype=float)
    frames = [T.copy()]
    for i in range(6):
        T = T @ _dh_matrix(_a[i], _alpha[i], _d[i], joints_rad[i])
        frames.append(T.copy())
    return frames


def fk_camera_transform_world(joints_rad: list) -> np.ndarray:
    """Actual Webots Camera device transform in world frame (metres)."""
    return webots_camera_transform_world(joints_rad)


def fk_camera_world(joints_rad: list) -> np.ndarray:
    """Camera position in world frame (metres) for given joint angles (radians)."""
    return fk_camera_transform_world(joints_rad)[:3, 3]


def webots_tool_slot_transform_world(joints_rad: list) -> np.ndarray:
    """UR5e toolSlot parent transform from the official Webots UR5e.proto.

    This is intentionally separate from the analytical DH FK.  The Webots
    toolSlot is mounted under wrist_3_link with an extra 0.1 m local +Y
    translation, so using the DH flange frame directly points the real camera
    at the wrong place.
    """
    q = joints_rad
    T = _fixed_transform(ROBOT_BASE_M)
    T = T @ _fixed_transform([0.0, 0.0, 0.163],
                             _axis_angle_rotation([0, 0, 1], q[0]))
    T = T @ _fixed_transform([0.0, 0.138, 0.0],
                             _axis_angle_rotation([0, 1, 0], q[1]))
    T = T @ _fixed_transform([0.0, 0.0, 0.0],
                             _axis_angle_rotation([0, 1, 0], pi / 2))
    T = T @ _fixed_transform([0.0, -0.131, 0.425],
                             _axis_angle_rotation([0, 1, 0], q[2]))
    T = T @ _fixed_transform([0.0, 0.0, 0.392],
                             _axis_angle_rotation([0, 1, 0], q[3]))
    T = T @ _fixed_transform([0.0, 0.0, 0.0],
                             _axis_angle_rotation([0, 1, 0], pi / 2))
    T = T @ _fixed_transform([0.0, 0.127, 0.0],
                             _axis_angle_rotation([0, 0, 1], q[4]))
    T = T @ _fixed_transform([0.0, 0.0, 0.100],
                             _axis_angle_rotation([0, 1, 0], q[5]))
    return T @ _fixed_transform(WEBOTS_TOOL_SLOT_TRANSLATION_M)


def webots_camera_transform_world(joints_rad: list) -> np.ndarray:
    """Actual Camera/RangeFinder/GPS transform in world frame."""
    T_tool = webots_tool_slot_transform_world(joints_rad)
    return T_tool @ T_FLANGE_TO_CAM


# =============================================================================
# Collision checking
# =============================================================================

def _seg_seg_dist(p1: np.ndarray, p2: np.ndarray,
                  q1: np.ndarray, q2: np.ndarray) -> float:
    """Minimum Euclidean distance between line segments p1->p2 and q1->q2.

    Parametric closest-point algorithm (Ericson, Real-Time Collision Detection
    section 5.1.9).  Handles degenerate (point) segments gracefully.
    """
    d1, d2, r = p2 - p1, q2 - q1, p1 - q1
    a = float(np.dot(d1, d1))
    e = float(np.dot(d2, d2))
    f = float(np.dot(d2, r))
    EPS = 1e-10

    if a <= EPS and e <= EPS:
        return float(np.linalg.norm(r))
    if a <= EPS:
        s, t = 0.0, float(np.clip(f / e, 0.0, 1.0))
    else:
        c = float(np.dot(d1, r))
        if e <= EPS:
            t, s = 0.0, float(np.clip(-c / a, 0.0, 1.0))
        else:
            b     = float(np.dot(d1, d2))
            denom = a * e - b * b
            s = float(np.clip((b * f - c * e) / denom, 0.0, 1.0)) \
                if abs(denom) > EPS else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, float(np.clip(-c / a, 0.0, 1.0))
            elif t > 1.0:
                t, s = 1.0, float(np.clip((b - c) / a, 0.0, 1.0))

    return float(np.linalg.norm(p1 + s * d1 - (q1 + t * d2)))


def is_collision_free(joints_rad: list) -> bool:
    """Return True if the posture is free of table penetration and self-collision.

    Table check
    -----------
    Every joint origin (except the fixed base) must be at least
    LINK_CLEARANCE_M above TABLE_Z_M in world frame.

    Self-collision (capsule model)
    ------------------------------
    Each UR5e link is approximated as a capsule:
      capsule_i = segment(joint_origin[i], joint_origin[i+1]) with radius LINK_RADII_MM[i]

    Adjacent capsules (i, i+1) share a joint and are always touching — they are
    skipped.  All non-adjacent pairs are tested; a collision is detected when:
      segment_distance(capsule_i, capsule_j) < LINK_RADII_MM[i] + LINK_RADII_MM[j]
    """
    frames = fk_joint_frames(joints_rad)

    # Joint origins in world frame (metres), shape (7, 3)
    pts_m = np.array([f[:3, 3] / 1000.0 + ROBOT_BASE_M for f in frames])

    # Table clearance — skip index 0 (fixed base at table level by design)
    if np.any(pts_m[1:, 2] < TABLE_Z_M + LINK_CLEARANCE_M):
        return False

    # Capsule self-collision — non-adjacent pairs (gap >= 2)
    pts_mm = pts_m * 1000.0
    for i in range(6):
        for j in range(i + 2, 6):
            dist = _seg_seg_dist(pts_mm[i], pts_mm[i + 1],
                                 pts_mm[j], pts_mm[j + 1])
            if dist < LINK_RADII_MM[i] + LINK_RADII_MM[j]:
                return False
    return True


# =============================================================================
# Geometry helpers
# =============================================================================

def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def look_at_rotation(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Legacy helper: 3x3 camera-to-world rotation with local -Z aimed at target.

    Near-zenith/nadir: world-up is swapped to X to avoid gimbal lock.
    """
    z  = _normalize(eye - target)
    up = np.array([0., 0., 1.])
    if abs(float(np.dot(z, up))) > 0.95:
        up = np.array([1., 0., 0.])
    x = _normalize(np.cross(up, z))
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def camera_target_angle_deg(joints_rad: list, target: np.ndarray) -> float:
    """Angle between the configured camera aim axis and camera-to-target vector."""
    T_cam = fk_camera_transform_world(joints_rad)
    p_cam = T_cam[:3, 3]
    aim_axis = _normalize(T_cam[:3, :3] @ CAMERA_AIM_AXIS_LOCAL)
    target_axis = _normalize(target - p_cam)
    return math.degrees(float(math.acos(
        float(np.clip(np.dot(aim_axis, target_axis), -1.0, 1.0))
    )))


def camera_ray_miss_distance_m(joints_rad: list, target: np.ndarray) -> float:
    """Shortest distance from target point to the configured camera center ray."""
    T_cam = fk_camera_transform_world(joints_rad)
    p_cam = T_cam[:3, 3]
    ray_axis = _normalize(T_cam[:3, :3] @ CAMERA_AIM_AXIS_LOCAL)
    target_delta = target - p_cam
    projection = float(np.dot(target_delta, ray_axis))
    if projection < 0.0:
        return float(np.linalg.norm(target_delta))
    return float(np.linalg.norm(target_delta - projection * ray_axis))


def _project_onto_view_plane(axis: np.ndarray,
                             aim_axis: np.ndarray,
                             fallback_axis: np.ndarray | None = None) -> np.ndarray:
    projected = axis - float(np.dot(axis, aim_axis)) * aim_axis
    if float(np.linalg.norm(projected)) > 1e-9:
        return _normalize(projected)
    if fallback_axis is None:
        return np.zeros(3, dtype=float)
    fallback = fallback_axis - float(np.dot(fallback_axis, aim_axis)) * aim_axis
    return _normalize(fallback)


def camera_roll_error_deg(joints_rad: list, target: np.ndarray) -> float:
    """Roll error around the camera ray.

    0 deg means the configured camera up axis, after projection onto the image
    plane, is aligned with world up projected onto the same plane.  When the ray
    is nearly parallel to world up, WORLD_ROLL_FALLBACK_AXIS defines the upright
    direction so zenith views still have a stable roll preference.
    """
    T_cam = fk_camera_transform_world(joints_rad)
    p_cam = T_cam[:3, 3]
    aim_axis = _normalize(T_cam[:3, :3] @ CAMERA_AIM_AXIS_LOCAL)
    camera_up = _normalize(T_cam[:3, :3] @ CAMERA_UP_AXIS_LOCAL)
    desired_up = _roll_reference_axis(aim_axis)
    actual_up = _project_onto_view_plane(camera_up, aim_axis, desired_up)
    return math.degrees(float(math.acos(
        float(np.clip(np.dot(actual_up, desired_up), -1.0, 1.0))
    )))


def camera_axis_errors_deg(joints_rad: list, target: np.ndarray) -> dict:
    T_cam = fk_camera_transform_world(joints_rad)
    p_cam = T_cam[:3, 3]
    target_axis = _normalize(target - p_cam)
    axes = {
        "+X": np.array([1.0, 0.0, 0.0]),
        "-X": np.array([-1.0, 0.0, 0.0]),
        "+Y": np.array([0.0, 1.0, 0.0]),
        "-Y": np.array([0.0, -1.0, 0.0]),
        "+Z": np.array([0.0, 0.0, 1.0]),
        "-Z": np.array([0.0, 0.0, -1.0]),
    }
    return {
        name: math.degrees(float(math.acos(float(np.clip(
            np.dot(_normalize(T_cam[:3, :3] @ axis), target_axis),
            -1.0,
            1.0,
        )))))
        for name, axis in axes.items()
    }


def cam_to_flange_base_mm(p_cam_m: np.ndarray,
                           R_cam_world: np.ndarray) -> np.ndarray:
    """4x4 flange transform in robot-base frame with translation in mm."""
    T_cw = np.eye(4, dtype=float)
    T_cw[:3, :3] = R_cam_world
    T_cw[:3,  3] = p_cam_m
    T_fw = T_cw @ T_CAM_TO_FLANGE
    T_fw[:3, 3] = (T_fw[:3, 3] - ROBOT_BASE_M) * 1000.0
    return T_fw


# =============================================================================
# IK filtering — all 8 solutions tried, both limits and collision checked
# =============================================================================

def _within_limits(joints_deg: list) -> bool:
    return all(lo <= j <= hi for j, (lo, hi) in zip(joints_deg, JOINT_LIMITS_DEG))


def find_best_ik(T_mm: np.ndarray) -> list | None:
    """Best valid IK solution (degrees), or None if none passes all checks.

    All 8 closed-form solutions are evaluated:
      1. Reject NaN / Inf (unreachable target).
      2. Reject joint-limit violations.
      3. Reject configurations that fail is_collision_free (table + self-collision).
      4. Among survivors, choose minimum weighted distance to REFERENCE_DEG;
         wrist-2 (J5) carries 5x weight to prefer camera-facing-down postures.
    """
    try:
        solutions = IK(T_mm.tolist())
    except Exception:
        return None

    ref_rad = [math.radians(d) for d in REFERENCE_DEG]
    best, best_score = None, float("inf")

    for sol in solutions:
        if not all(isfinite(v) for v in sol):
            continue
        sol_deg = [math.degrees(v) for v in sol]
        if not _within_limits(sol_deg):
            continue
        if not is_collision_free(sol):
            continue

        score  = sum(abs(sol[i] - ref_rad[i]) for i in range(6))
        score += 5.0 * abs(sol[4] - ref_rad[4])   # extra weight on wrist-2

        if score < best_score:
            best_score = score
            best = sol_deg

    return best


def _joint_bounds_rad() -> tuple:
    lower = np.array([math.radians(lo) for lo, _ in JOINT_LIMITS_DEG], dtype=float)
    upper = np.array([math.radians(hi) for _, hi in JOINT_LIMITS_DEG], dtype=float)
    return lower, upper


def _webots_ik_seeds() -> list:
    lower, upper = _joint_bounds_rad()
    seeds = [np.array([math.radians(v) for v in REFERENCE_DEG], dtype=float)]
    for joints_deg in EXISTING_POSES_DEG.values():
        seeds.append(np.array([math.radians(v) for v in joints_deg], dtype=float))

    rng = np.random.default_rng(7)
    for _ in range(14):
        seeds.append(lower + rng.random(6) * (upper - lower))
    return [np.clip(seed, lower, upper) for seed in seeds]


def _webots_pose_errors(joints_rad: np.ndarray,
                        p_cam_target: np.ndarray,
                        look_target: np.ndarray) -> tuple:
    T_cam = webots_camera_transform_world(joints_rad)
    p_cam = T_cam[:3, 3]
    aim_axis = _normalize(T_cam[:3, :3] @ CAMERA_AIM_AXIS_LOCAL)
    target_axis = _normalize(look_target - p_cam)
    pos_err = float(np.linalg.norm(p_cam - p_cam_target))
    angle_err = math.degrees(float(math.acos(
        float(np.clip(np.dot(aim_axis, target_axis), -1.0, 1.0))
    )))
    target_delta = look_target - p_cam
    projection = float(np.dot(target_delta, aim_axis))
    if projection < 0.0:
        ray_miss = float(np.linalg.norm(target_delta))
    else:
        ray_miss = float(np.linalg.norm(target_delta - projection * aim_axis))
    roll_err = camera_roll_error_deg(joints_rad.tolist(), look_target)
    return pos_err, angle_err, ray_miss, roll_err


def _roll_reference_axis(aim_axis: np.ndarray) -> np.ndarray:
    desired_up = _project_onto_view_plane(
        _normalize(WORLD_UP_AXIS),
        aim_axis,
        _normalize(WORLD_ROLL_FALLBACK_AXIS),
    )
    if float(np.linalg.norm(desired_up)) > 1e-9:
        return desired_up
    return np.array([0.0, 1.0, 0.0], dtype=float)


def _roll_residual(T_cam: np.ndarray, aim_axis: np.ndarray) -> np.ndarray:
    camera_up = _normalize(T_cam[:3, :3] @ CAMERA_UP_AXIS_LOCAL)
    desired_up = _roll_reference_axis(aim_axis)
    actual_up = _project_onto_view_plane(camera_up, aim_axis, desired_up)
    return (actual_up - desired_up) * CAMERA_ROLL_WEIGHT


def find_best_webots_ik(p_cam_target: np.ndarray,
                        look_target: np.ndarray) -> list | None:
    """Numerically solve joints against the actual Webots UR5e + Camera chain."""
    lower, upper = _joint_bounds_rad()
    ref = np.array([math.radians(v) for v in REFERENCE_DEG], dtype=float)
    def residual(q: np.ndarray) -> np.ndarray:
        T_cam = webots_camera_transform_world(q)
        p_cam = T_cam[:3, 3]
        aim_axis = _normalize(T_cam[:3, :3] @ CAMERA_AIM_AXIS_LOCAL)
        target_axis = _normalize(look_target - p_cam)
        ray_miss = np.cross(aim_axis, look_target - p_cam)
        return np.concatenate([
            (p_cam - p_cam_target) / 0.015,
            (aim_axis - target_axis) * 3.0,
            ray_miss / 0.005,
            _roll_residual(T_cam, target_axis),
            (q - ref) * 0.02,
        ])

    best = None
    for seed in _webots_ik_seeds():
        result = least_squares(
            residual,
            seed,
            bounds=(lower, upper),
            max_nfev=350,
            xtol=1e-8,
            ftol=1e-8,
            gtol=1e-8,
        )
        pos_err, angle_err, ray_miss, _roll_err = _webots_pose_errors(
            result.x,
            p_cam_target,
            look_target,
        )
        if pos_err > MAX_WEBOTS_POSITION_ERROR_M:
            continue
        if angle_err > MAX_WEBOTS_TARGET_ERROR_DEG:
            continue
        if ray_miss > MAX_CAMERA_RAY_MISS_M:
            continue
        if _roll_err > MAX_CAMERA_ROLL_ERROR_DEG:
            continue
        if not is_collision_free(result.x.tolist()):
            continue

        score = float(np.linalg.norm(residual(result.x)))
        if best is None or score < best[0]:
            best = (score, result.x)

    if best is None:
        return None
    return [math.degrees(v) for v in best[1]]


# =============================================================================
# Hemisphere sampling + greedy angular-spread selection
# =============================================================================

def sample_hemisphere() -> list:
    positions = []
    for el_deg in ELEVATION_ANGLES_DEG:
        el_rad = math.radians(el_deg)
        if abs(el_deg - 90.0) < 1e-9:
            positions.append(OBJECT_CENTER_M + np.array([0., 0., HEMISPHERE_RADIUS_M]))
            continue
        for k in range(AZIMUTH_STEPS):
            az_rad = 2 * pi * k / AZIMUTH_STEPS
            dx = HEMISPHERE_RADIUS_M * cos(el_rad) * cos(az_rad)
            dy = HEMISPHERE_RADIUS_M * cos(el_rad) * sin(az_rad)
            dz = HEMISPHERE_RADIUS_M * sin(el_rad)
            positions.append(OBJECT_CENTER_M + np.array([dx, dy, dz]))
    if not any(abs(el_deg - 90.0) < 1e-9 for el_deg in ELEVATION_ANGLES_DEG):
        positions.append(OBJECT_CENTER_M + np.array([0., 0., HEMISPHERE_RADIUS_M]))
    return positions


def deduplicate_viewpoints(valid: list,
                           pos_decimals: int = 4,
                           joint_decimals: int = 2) -> list:
    unique = []
    seen = set()
    for p_cam, j_deg in valid:
        key = (
            tuple(round(float(v), pos_decimals) for v in p_cam),
            tuple(round(float(v), joint_decimals) for v in j_deg),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append((p_cam, j_deg))
    return unique


def _min_angle_to_selected(selected: list, candidate: np.ndarray) -> float:
    if not selected:
        return 180.0
    v_c = _normalize(candidate - OBJECT_CENTER_M)
    return min(
        math.degrees(float(math.acos(
            float(np.clip(np.dot(v_c, _normalize(s - OBJECT_CENTER_M)), -1.0, 1.0))
        )))
        for s in selected
    )


def find_valid_viewpoints() -> list:
    """Return all valid (cam_pos_world, joints_deg) pairs."""
    candidates = sample_hemisphere()
    print(f"  Candidates sampled : {len(candidates)}")

    valid = []
    for p_cam in candidates:
        j_deg = find_best_webots_ik(p_cam, OBJECT_CENTER_M)
        if j_deg is not None:
            valid.append((p_cam, j_deg))

    print(f"  Valid after Webots IK + collision filter : {len(valid)}")
    unique = deduplicate_viewpoints(valid)
    if len(unique) != len(valid):
        print(f"  Unique after duplicate filter : {len(unique)}")
    return unique


def select_viewpoints(valid: list, num_poses: int) -> list:
    if not valid:
        return []
    selected_pos: list = []
    selected: list     = []
    pool = valid[:]
    for _ in range(min(num_poses, len(pool))):
        best_idx = max(
            range(len(pool)),
            key=lambda i: _min_angle_to_selected(selected_pos, pool[i][0]),
        )
        p, j = pool[best_idx]
        selected_pos.append(p)
        selected.append((p, j))
        pool.pop(best_idx)

    return selected


def plan_viewpoints() -> list:
    """Return up to NUM_OUTPUT_POSES (cam_pos_world, joints_deg) pairs."""
    return select_viewpoints(find_valid_viewpoints(), NUM_OUTPUT_POSES)


# =============================================================================
# Output helpers
# =============================================================================

def _elevation_azimuth(p_cam: np.ndarray) -> tuple:
    delta = p_cam - OBJECT_CENTER_M
    dist  = float(np.linalg.norm(delta))
    el    = math.degrees(math.asin(float(np.clip(delta[2] / max(dist, 1e-9), -1.0, 1.0))))
    az    = math.degrees(math.atan2(delta[1], delta[0]))
    return el, az


def print_camera_poses(selected: list) -> None:
    sep = "=" * 64
    print()
    print(sep)
    print("  Paste into ur5e_auto_capture_controller.py -> CAMERA_POSES")
    print(sep)
    print()
    print("CAMERA_POSES = {")
    for idx, (p_cam, j_deg) in enumerate(selected, start=1):
        rounded = [round(v, 2) for v in j_deg]
        el, az  = _elevation_azimuth(p_cam)
        target_err = camera_target_angle_deg(
            [math.radians(v) for v in j_deg],
            OBJECT_CENTER_M,
        )
        ray_miss_mm = 1000.0 * camera_ray_miss_distance_m(
            [math.radians(v) for v in j_deg],
            OBJECT_CENTER_M,
        )
        print(f"    {idx}: {{\"joint_deg\": {rounded}}},  "
              f"# el={el:.0f} az={az:.0f} "
              f"target_err={target_err:.1f}deg ray_miss={ray_miss_mm:.1f}mm")
    print("}")
    print()
    print("Also update VIEW_SEQUENCE in ycb_supervisor_four_view.py:")
    print(f"VIEW_SEQUENCE = {tuple(range(1, len(selected) + 1))}")


def export_candidates(valid: list, output_path: str) -> None:
    records = []
    for idx, (p_cam, j_deg) in enumerate(valid, start=1):
        el, az = _elevation_azimuth(p_cam)
        joints_rad = [math.radians(v) for v in j_deg]
        records.append({
            "id": idx,
            "joint_deg": [round(v, 4) for v in j_deg],
            "camera_position_m": [float(v) for v in p_cam],
            "elevation_deg": float(el),
            "azimuth_deg": float(az),
            "target_err_deg": camera_target_angle_deg(joints_rad, OBJECT_CENTER_M),
            "ray_miss_m": camera_ray_miss_distance_m(joints_rad, OBJECT_CENTER_M),
            "roll_err_deg": camera_roll_error_deg(joints_rad, OBJECT_CENTER_M),
        })

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(records, file, indent=2)
    print(f"Wrote {len(records)} candidate poses to {output_path}")


def verify_existing_poses() -> None:
    """Print FK camera position, distance, elevation/azimuth, and collision status."""
    print()
    print("Existing CAMERA_POSES — camera positions in world frame:")
    print(f"  {'Pose':>4}  {'x_m':>7}  {'y_m':>7}  {'z_m':>7}  "
          f"{'el':>6}  {'az':>7}  {'dist_m':>7}  {'target_err':>10}  {'ray_mm':>7}  "
          f"{'collision_free':>14}")
    print("  " + "-" * 94)
    for pose_id, joints_deg in EXISTING_POSES_DEG.items():
        joints_rad = [math.radians(d) for d in joints_deg]
        p_cam      = fk_camera_world(joints_rad)
        el, az     = _elevation_azimuth(p_cam)
        dist       = float(np.linalg.norm(p_cam - OBJECT_CENTER_M))
        target_err = camera_target_angle_deg(joints_rad, OBJECT_CENTER_M)
        ray_miss_mm = 1000.0 * camera_ray_miss_distance_m(joints_rad, OBJECT_CENTER_M)
        roll_err = camera_roll_error_deg(joints_rad, OBJECT_CENTER_M)
        ok         = "yes" if is_collision_free(joints_rad) else "NO (collision!)"
        print(f"  {pose_id:>4}  {p_cam[0]:>7.3f}  {p_cam[1]:>7.3f}  {p_cam[2]:>7.3f}  "
              f"{el:>6.1f}  {az:>7.1f}  {dist:>7.3f}  {target_err:>9.1f}°  "
              f"{ray_miss_mm:>7.1f}  roll={roll_err:>5.1f}°  "
              f"{ok:>14}")
        axis_errors = camera_axis_errors_deg(joints_rad, OBJECT_CENTER_M)
        best_axis = min(axis_errors, key=axis_errors.get)
        print("        axis_err: "
              f"+X={axis_errors['+X']:.1f} "
              f"-X={axis_errors['-X']:.1f} "
              f"+Y={axis_errors['+Y']:.1f} "
              f"-Y={axis_errors['-Y']:.1f} "
              f"+Z={axis_errors['+Z']:.1f} "
              f"-Z={axis_errors['-Z']:.1f} "
              f"best={best_axis}")


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    global OBJECT_CENTER_M, HEMISPHERE_RADIUS_M

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--center", type=float, nargs=3, metavar=("X", "Y", "Z"),
                        default=None,
                        help=f"Object centre in world (metres). "
                             f"Default: {OBJECT_CENTER_M.tolist()}")
    parser.add_argument("--radius", type=float, default=HEMISPHERE_RADIUS_M,
                        help=f"Hemisphere radius in metres "
                             f"(default: {HEMISPHERE_RADIUS_M})")
    parser.add_argument("--output",
                        default=os.path.join(SCRIPT_DIR, "candidate_viewpoints.json"),
                        help="Candidate JSON output path "
                             "(default: candidate_viewpoints.json next to this script)")
    args = parser.parse_args()

    if args.center is not None:
        OBJECT_CENTER_M = np.array(args.center)
    HEMISPHERE_RADIUS_M = args.radius

    print("Candidate Viewpoint Generator")
    print(f"  Mount source        : {WBT_MOUNTS['source']} ({WBT_MOUNTS['world_path']})")
    print(f"  Robot base (world)  : {ROBOT_BASE_M.tolist()} m")
    print(f"  Object centre       : {OBJECT_CENTER_M.tolist()} m")
    print(f"  Camera mount xyz    : {_t_flange_to_d455.tolist()} m")
    print(f"  Camera sensor xyz   : {_t_flange_to_cam.tolist()} m")
    print(f"  Hemisphere radius   : {HEMISPHERE_RADIUS_M} m")
    print(f"  Elevation rings     : {ELEVATION_ANGLES_DEG} deg")
    print(f"  Azimuth steps       : {AZIMUTH_STEPS}")
    print(f"  Table clearance     : {LINK_CLEARANCE_M} m above Z={TABLE_Z_M}")
    print(f"  Link capsule radii  : {LINK_RADII_MM} mm")
    print(f"  Roll up axis        : camera {CAMERA_UP_AXIS_LOCAL.tolist()} -> world {WORLD_UP_AXIS.tolist()}")
    print(f"  Max roll error      : {MAX_CAMERA_ROLL_ERROR_DEG} deg")
    print(f"  Output              : {args.output}")

    print()
    print("Generating candidate viewpoints ...")
    valid = find_valid_viewpoints()

    if not valid:
        print("No candidate poses found. Suggestions:")
        print("  * Increase --radius (try 0.6 or 0.65)")
        print("  * Relax JOINT_LIMITS_DEG in this script")
        print("  * Reduce LINK_RADII_MM or LINK_CLEARANCE_M")
        sys.exit(1)

    export_candidates(valid, args.output)


if __name__ == "__main__":
    main()
