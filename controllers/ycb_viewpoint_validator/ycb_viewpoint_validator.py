from controller import Supervisor
import importlib.util
import json
import math
import shutil
import sys
import time
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
VIEWPOINTS_DIR      = REPO_ROOT / "data" / "viewpoints"
CAPTURE_CONFIG_DIR  = REPO_ROOT / "controllers" / "ycb_supervisor_capture"
UR5E_TEST_DIR       = REPO_ROOT / "controllers" / "ur5e_test_controller"
SUPERVISOR_TEST_DIR = REPO_ROOT / "controllers" / "ycb_supervisor_ros2_test"

for path in (CAPTURE_CONFIG_DIR, UR5E_TEST_DIR, SUPERVISOR_TEST_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import candidate_viewpoint_config as planner_config  # noqa: E402

# bridge 工具從共用模組引入，不重複維護
from ros2_bridge_utils import (  # noqa: E402
    launch_ros2_bridge,
    wait_for_bridge_ready,
    request_plan,
    stop_ros2_bridge,
)


ARM_COMMAND_EMITTER  = "arm_command_emitter"
ARM_STATUS_RECEIVER  = "arm_status_receiver"

# 支援 --multi / --x-offset（優先順序：VALIDATOR_ARGS 環境變數 > controllerArgs）
import os as _os
_ARGS_STR = _os.environ.get("VALIDATOR_ARGS", "") or " ".join(sys.argv[1:])
_ARGS_LIST = _ARGS_STR.split()
_MULTI_MODE = "--multi" in _ARGS_STR

def _parse_float_arg(name, default=None):
    for i, arg in enumerate(_ARGS_LIST):
        if arg == name and i + 1 < len(_ARGS_LIST):
            try:
                return float(_ARGS_LIST[i + 1])
            except ValueError:
                pass
    return default

_X_OFFSET  = _parse_float_arg("--x-offset", 0.0)
_RADIUS    = _parse_float_arg("--radius")    # None = 不過濾，驗證所有半徑
_WS_OFFSET = _parse_float_arg("--ws-offset") # None = 不帶工作球碰撞物件

def _build_param_tag():
    """從設定檔參數自動產生標籤，相同設定永遠對應同一個檔案。
    格式：el<角度s>_az<步數>_r<半徑s>[_x<X>]
    例：el45_60_75_90_az4_r050_055_065_070_075_080_x+010
    """
    import importlib.util as _ilu
    _cfg_path = Path(CURRENT_DIR).parents[1] / "controllers" / "ycb_supervisor_capture" / "candidate_viewpoint_config.py"
    _spec = _ilu.spec_from_file_location("_cfg", _cfg_path)
    _cfg  = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_cfg)

    el_str = "_".join(str(int(e)) for e in _cfg.ELEVATION_ANGLES_DEG)
    az_str = str(int(_cfg.AZIMUTH_STEPS))
    if _RADIUS is not None:
        r_str = f"{int(_RADIUS * 100):03d}"
    elif _MULTI_MODE:
        r_str = "_".join(f"{int(r*100):03d}" for r in _cfg.HEMISPHERE_RADII_M)
    else:
        r_str = f"{int(_cfg.HEMISPHERE_RADIUS_M * 100):03d}"
    x_str  = f"_x{int(_X_OFFSET * 100):+04d}" if _X_OFFSET != 0.0 else ""
    ws_str = f"_ws{int(_WS_OFFSET * 100):03d}" if _WS_OFFSET is not None else ""
    return f"el{el_str}_az{az_str}_r{r_str}{x_str}{ws_str}"

_TAG = _build_param_tag()

_base = "validated_viewpoints_multi" if _MULTI_MODE else "validated_viewpoints"

# 候選視角：x≠0 時讀對應的 x_offset 檔，不存在則 fallback 到預設檔
def _resolve_candidate_path():
    if _MULTI_MODE:
        if _X_OFFSET != 0.0:
            x_tag = f"x{int(_X_OFFSET * 100):+04d}"
            p = VIEWPOINTS_DIR / f"candidate_viewpoints_multi_{x_tag}.json"
            if p.exists():
                return p
        return VIEWPOINTS_DIR / "candidate_viewpoints_multi.json"
    else:
        return VIEWPOINTS_DIR / "candidate_viewpoints.json"

CANDIDATE_PATH     = _resolve_candidate_path()
OUTPUT_PATH        = VIEWPOINTS_DIR / f"{_base}_{_TAG}.json"
OUTPUT_PATH_LATEST = VIEWPOINTS_DIR / f"{_base}_latest.json"

if _MULTI_MODE:
    print(f"[Validator] 多半徑模式：讀取 {CANDIDATE_PATH.name}")
print(f"[Validator] x_offset  = {_X_OFFSET:+.3f} m")
print(f"[Validator] radius    = {_RADIUS} m" + ("  (全部半徑)" if _RADIUS is None else ""))
print(f"[Validator] ws_offset = {_WS_OFFSET} m" + ("  (不帶工作球)" if _WS_OFFSET is None else f"  → 工作球半徑 = {_RADIUS - _WS_OFFSET:.3f} m" if _RADIUS is not None else ""))
print(f"[Validator] 輸出標籤: {_TAG}  →  {OUTPUT_PATH.name}")

_oc = planner_config.OBJECT_CENTER_M
TARGET_M = [_oc[0] + _X_OFFSET, _oc[1], _oc[2]]
ROBOT_BASE_M         = planner_config.ROBOT_BASE_M
YCB_OBJECT_NAME      = getattr(planner_config, "YCB_OBJECT_NAME",     "")
YCB_OBJECT_ROTATION  = getattr(planner_config, "YCB_OBJECT_ROTATION", [0, 1, 0, 0])

def _build_workspace_collision_object():
    if _RADIUS is None or _WS_OFFSET is None:
        return []
    sphere_r = _RADIUS - _WS_OFFSET
    if sphere_r <= 0:
        return []
    oc = planner_config.OBJECT_CENTER_M
    rb = planner_config.ROBOT_BASE_M
    pos = [oc[0] + _X_OFFSET - rb[0],
           oc[1] - rb[1],
           oc[2] - rb[2]]
    return [{
        "id": "ycb_workspace_sphere",
        "shape": "sphere",
        "size": [sphere_r * 2],
        "position": pos,
    }]

_WORKSPACE_COLLISION_OBJECT = _build_workspace_collision_object()
# 半球體視覺化：半徑 → DEF 名稱對應表
HEMISPHERE_DEF_MAP = {
    0.50: "HEMISPHERE_R05",
    0.55: "HEMISPHERE_R055",
    0.60: "HEMISPHERE_R060",
    0.65: "HEMISPHERE_R065",
    0.70: "HEMISPHERE_R07",
    0.75: "HEMISPHERE_R075",
    0.80: "HEMISPHERE_R08",
}


def set_hemisphere_visibility(supervisor, active_radius_m):
    """只顯示 active_radius_m 對應的半球體，其餘隱藏。"""
    active_r = round(float(active_radius_m), 2) if active_radius_m is not None else None
    for r, def_name in HEMISPHERE_DEF_MAP.items():
        node = supervisor.getFromDef(def_name)
        if node is None:
            continue
        try:
            shape = node.getField("children").getMFNode(0)
            material = shape.getField("appearance").getSFNode().getField("material").getSFNode()
            transparency = 0.85 if (r == active_r) else 1.0
            material.getField("transparency").setSFFloat(transparency)
        except Exception:
            pass


CAMERA_AIM_AXIS_LOCAL    = planner_config.CAMERA_AIM_AXIS_LOCAL
CAMERA_UP_AXIS_LOCAL     = planner_config.CAMERA_UP_AXIS_LOCAL
CAMERA_SENSOR_OFFSET_LOCAL = planner_config.T_D455_TO_SENSOR_M
WORLD_UP_AXIS        = planner_config.WORLD_UP_AXIS
WORLD_ROLL_FALLBACK_AXIS = planner_config.WORLD_ROLL_FALLBACK_AXIS
PHYSICS_SETTLE_SEC           = 2.0
ARM_MOTOR_VELOCITY_RAD_PER_SEC = 1.5
ARM_SETTLE_TIME_BUFFER_SEC   = 8.0
POST_ARRIVAL_PAUSE_SEC       = 0.75
MAX_RAY_MISS_M               = 0.005
MAX_CAMERA_ROLL_ERROR_DEG    = planner_config.MAX_CAMERA_ROLL_ERROR_DEG
MIN_CAMERA_Z_M               = 0.08
HOME_POSE_RAD = [0.0, -math.pi / 2, math.pi / 2, -math.pi / 2, -math.pi / 2, 0.0]

BRIDGE_STARTUP_TIMEOUT_SEC   = 30.0
PHYSICS_SETTLE_SEC           = 2.0


# ── 候選點載入 ────────────────────────────────────────────────────────────────

def load_ur5e_camera_poses():
    module_path = UR5E_TEST_DIR / "ur5e_test_controller.py"
    spec = importlib.util.spec_from_file_location("ur5e_test_controller", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return [
        {"id": pose_id, "joint_deg": data["joint_deg"], "source": "ur5e_test_controller"}
        for pose_id, data in sorted(module.CAMERA_POSES.items())
    ]


def load_candidates():
    if CANDIDATE_PATH.exists():
        with CANDIDATE_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        candidates = []
        for i, item in enumerate(data, start=1):
            if isinstance(item, dict) and "joint_deg" in item:
                candidates.append({
                    "id": item.get("id", i),
                    "joint_deg": item["joint_deg"],
                    "source": str(CANDIDATE_PATH),
                    "meta": item,
                })
        if candidates:
            return candidates
    return load_ur5e_camera_poses()


# ── Webots 工具 ───────────────────────────────────────────────────────────────

def interpolate_trajectory(waypoints: list, timestep_ms: int, travel_time_sec: float) -> list:
    """MoveIt 稀疏 waypoints → 每個 simulation step 一個點的密集軌跡（線性內插）。"""
    n_steps = max(len(waypoints), int(travel_time_sec * 1000 / max(1, timestep_ms)))
    if len(waypoints) == 1:
        return [waypoints[0][:] for _ in range(n_steps)]
    n_seg = len(waypoints) - 1
    result = []
    for i in range(n_steps):
        t = i / (n_steps - 1)
        seg_f = t * n_seg
        seg = min(int(seg_f), n_seg - 1)
        lt = seg_f - seg
        a, b = waypoints[seg], waypoints[seg + 1]
        result.append([a[j] + lt * (b[j] - a[j]) for j in range(len(a))])
    return result


def wait_seconds(supervisor, timestep, seconds):
    steps = max(1, int(seconds * 1000 / max(1, timestep)))
    for _ in range(steps):
        if supervisor.step(timestep) == -1:
            return False
    return True


def clear_receiver(receiver):
    if receiver is None:
        return
    while receiver.getQueueLength() > 0:
        receiver.nextPacket()


def wait_for_arm_arrival(supervisor, timestep, receiver, command_id, timeout_sec):
    if receiver is None:
        return wait_seconds(supervisor, timestep, timeout_sec)
    start_time = supervisor.getTime()
    while supervisor.getTime() - start_time <= timeout_sec:
        if supervisor.step(timestep) == -1:
            return False
        while receiver.getQueueLength() > 0:
            msg = receiver.getString()
            receiver.nextPacket()
            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                continue
            if data.get("status") == "arrived" and str(data.get("command_id")) == str(command_id):
                print(f"[Validator] Arm arrived (max_error={float(data.get('max_error_rad', 0.0)):.4f} rad)")
                return True
    print(f"[Validator] Arm arrival timeout for command {command_id}")
    return False


def send_waypoint(emitter, joints_rad: list, command_id: str) -> bool:
    if emitter is None:
        return False
    payload = {"type": "waypoint", "joints": [float(v) for v in joints_rad], "id": str(command_id)}
    emitter.send(json.dumps(payload).encode("utf-8"))
    return True


def estimate_travel_time(current: list, target: list) -> float:
    max_delta = max(abs(t - c) for t, c in zip(target, current))
    motion_time = max_delta / max(ARM_MOTOR_VELOCITY_RAD_PER_SEC, 1e-6)
    return max(5.0, motion_time + ARM_SETTLE_TIME_BUFFER_SEC)


# ── 幾何工具 ──────────────────────────────────────────────────────────────────

def mat_vec_mul(m, v):
    return [m[0]*v[0]+m[1]*v[1]+m[2]*v[2],
            m[3]*v[0]+m[4]*v[1]+m[5]*v[2],
            m[6]*v[0]+m[7]*v[1]+m[8]*v[2]]

def norm(v):
    l = math.sqrt(sum(x*x for x in v))
    return [x/l for x in v] if l > 1e-12 else [0.0]*3

def subtract(a, b): return [a[i]-b[i] for i in range(3)]
def add(a, b):      return [a[i]+b[i] for i in range(3)]
def scale(v, s):    return [x*s for x in v]
def dot(a, b):      return sum(a[i]*b[i] for i in range(3))
def distance(a, b): return math.sqrt(sum((a[i]-b[i])**2 for i in range(3)))

def project_onto_view_plane(axis, ray_axis, fallback_axis=None):
    p = subtract(axis, scale(ray_axis, dot(axis, ray_axis)))
    if math.sqrt(dot(p, p)) > 1e-9:
        return norm(p)
    if fallback_axis is None:
        return [0.0]*3
    fb = subtract(fallback_axis, scale(ray_axis, dot(fallback_axis, ray_axis)))
    return norm(fb)

def roll_reference_axis(ray_axis):
    desired = project_onto_view_plane(norm(WORLD_UP_AXIS), ray_axis, norm(WORLD_ROLL_FALLBACK_AXIS))
    return desired if math.sqrt(dot(desired, desired)) > 1e-9 else [0.0, 1.0, 0.0]

def roll_error_deg(orientation, ray_axis):
    camera_up  = norm(mat_vec_mul(orientation, CAMERA_UP_AXIS_LOCAL))
    desired_up = roll_reference_axis(ray_axis)
    actual_up  = project_onto_view_plane(camera_up, ray_axis, desired_up)
    return math.degrees(math.acos(max(-1.0, min(1.0, dot(actual_up, desired_up)))))

def camera_ray_report(camera_node):
    body_pos   = camera_node.getPosition()
    orient     = camera_node.getOrientation()
    offset     = mat_vec_mul(orient, CAMERA_SENSOR_OFFSET_LOCAL)
    ray_origin = add(body_pos, offset)
    ray_axis   = norm(mat_vec_mul(orient, CAMERA_AIM_AXIS_LOCAL))
    delta      = subtract(TARGET_M, ray_origin)
    proj       = dot(delta, ray_axis)
    closest    = add(ray_origin, scale(ray_axis, max(0.0, proj)))
    miss       = distance(TARGET_M, closest)
    angle      = math.degrees(math.acos(max(-1.0, min(1.0, dot(norm(delta), ray_axis)))))
    return {
        "body_position_m":  body_pos,
        "ray_origin_m":     ray_origin,
        "ray_axis_world":   ray_axis,
        "ray_projection_m": proj,
        "ray_miss_m":       miss,
        "target_angle_deg": angle,
        "roll_err_deg":     roll_error_deg(orient, ray_axis),
    }

def contact_count(node, include_descendants=False):
    if node is None:
        return 0
    try:
        return len(node.getContactPoints(include_descendants))
    except TypeError:
        try:
            return len(node.getContactPoints())
        except Exception:
            return 0
    except Exception:
        return 0


# ── 主程式 ────────────────────────────────────────────────────────────────────

def main():
    supervisor     = Supervisor()
    timestep       = int(supervisor.getBasicTimeStep())
    emitter        = supervisor.getDevice(ARM_COMMAND_EMITTER)
    status_receiver = supervisor.getDevice(ARM_STATUS_RECEIVER)
    if status_receiver is not None:
        status_receiver.enable(timestep)
    else:
        print(f"[Validator] 找不到 {ARM_STATUS_RECEIVER}，改用時間等待。")

    ur5e_node   = supervisor.getFromDef(planner_config.UR5E_DEF)
    camera_node = supervisor.getFromDef(planner_config.CAMERA_DEF)

    if emitter is None:
        print("[Validator] 找不到 arm_command_emitter"); return
    if ur5e_node is None:
        print(f"[Validator] 找不到 DEF {planner_config.UR5E_DEF}"); return
    if camera_node is None:
        print(f"[Validator] 找不到 DEF {planner_config.CAMERA_DEF}"); return

    # x_offset：移動半球視覺節點，使其中心對齊物體實際位置
    if _X_OFFSET != 0.0:
        for def_name in HEMISPHERE_DEF_MAP.values():
            node = supervisor.getFromDef(def_name)
            if node is not None:
                node.getField("translation").setSFVec3f([_X_OFFSET, 0.0, 0.0])
        print(f"[Validator] 半球視覺節點已平移 x = {_X_OFFSET:+.3f} m")

    candidates = load_candidates()
    print(f"[Validator] 候選點數: {len(candidates)}")
    print(f"[Validator] 目標: {TARGET_M}")
    print(f"[Validator] 輸出: {OUTPUT_PATH}")

    # 先推進一步讓模擬開始，再啟動 bridge（與 supervisor_ros2_test 一致）
    supervisor.step(timestep)

    # 啟動 bridge（非阻塞），在物理穩定期間同時等就緒
    bridge_proc_raw, bridge_line_queue = launch_ros2_bridge()
    print(f"[Validator] 等待物理穩定 + bridge 就緒 ({PHYSICS_SETTLE_SEC:.1f}s)...")
    bridge_proc = wait_for_bridge_ready(
        supervisor, timestep, bridge_proc_raw, bridge_line_queue, PHYSICS_SETTLE_SEC
    )
    if bridge_proc_raw is not None and bridge_proc is None:
        print(f"[Validator] Bridge 尚未就緒，繼續等候（最多 {BRIDGE_STARTUP_TIMEOUT_SEC:.0f}s）...")
        bridge_proc = wait_for_bridge_ready(
            supervisor, timestep, bridge_proc_raw, bridge_line_queue, BRIDGE_STARTUP_TIMEOUT_SEC
        )

    if bridge_proc is not None:
        print("[Validator] MoveIt 規劃已啟用")
    else:
        print("[Validator] 無 MoveIt bridge，僅做 Webots 碰撞檢測")

    # 初始化：明確送手臂到 HOME，確保起始狀態正確（supervisor 世界預設已在 HOME）
    print("[Validator] 初始化手臂到 HOME ...")
    clear_receiver(status_receiver)
    send_waypoint(emitter, HOME_POSE_RAD, "init_home")
    if not wait_for_arm_arrival(supervisor, timestep, status_receiver, "init_home", 30.0):
        print("[Validator] 警告：手臂未能到達 HOME（逾時），繼續執行但結果可能不準確")
    wait_seconds(supervisor, timestep, 2.0)

    baseline_ur5e_contacts   = contact_count(ur5e_node, include_descendants=True)
    baseline_camera_contacts = contact_count(camera_node, include_descendants=True)

    current_joints = HOME_POSE_RAD[:]

    results   = []
    validated = []

    try:
        for idx, candidate in enumerate(candidates, start=1):
            joint_deg  = [float(v) for v in candidate["joint_deg"]]
            target_rad = [math.radians(d) for d in joint_deg]
            cid        = candidate["id"]
            radius_m   = (candidate.get("radius_m")
                          or candidate.get("meta", {}).get("radius_m")
                          or planner_config.HEMISPHERE_RADIUS_M)
            if _RADIUS is not None and abs(radius_m - _RADIUS) > 1e-6:
                continue  # 跳過不符合 --radius 的視角
            set_hemisphere_visibility(supervisor, radius_m)
            print(f"\n[Validator] ── 候選點 {idx}/{len(candidates)}  id={cid} ──")

            travel_time = estimate_travel_time(current_joints, target_rad)

            # ── MoveIt 規劃 ──────────────────────────────────────────────────
            plan_success  = None
            plan_error    = ""
            n_waypoints   = 1

            if bridge_proc is not None:
                print("[Validator] 請求 MoveIt 規劃...")
                # 工作球依當前視角半徑動態計算
                if _WS_OFFSET is not None:
                    sphere_r = radius_m - _WS_OFFSET
                    oc = planner_config.OBJECT_CENTER_M
                    rb = planner_config.ROBOT_BASE_M
                    ws_col = [{
                        "id": "ycb_workspace_sphere",
                        "shape": "sphere",
                        "size": [sphere_r * 2],
                        "position": [oc[0] + _X_OFFSET - rb[0],
                                     oc[1] - rb[1],
                                     oc[2] - rb[2]],
                    }]
                else:
                    ws_col = []
                result = request_plan(bridge_proc, bridge_line_queue, current_joints, target_rad,
                                     ws_col,
                                     supervisor=supervisor, timestep=timestep)
                if result is None or not result.get("success"):
                    plan_error   = result.get("error", "unknown") if result else "no response"
                    plan_success = False
                    print(f"  SKIP  規劃失敗: {plan_error}")
                    results.append(_make_record(candidate, joint_deg, False, plan_error,
                                                0, f"plan_failed: {plan_error}"))
                    continue  # 手臂不動，直接跳下一個
                plan_success = True
                raw_wps = result.get("waypoints") or [target_rad]
                # 相容新格式（dict with positions）與舊格式（plain list）
                waypoints = [
                    wp["positions"] if isinstance(wp, dict) else wp
                    for wp in raw_wps
                ]
                n_waypoints  = len(waypoints)
                print(f"[Validator] 規劃成功，{n_waypoints} 個 waypoints")
            else:
                # 無 bridge：直接移動（僅做幾何 & 碰撞驗證）
                waypoints = [target_rad]

            # ── 平滑軌跡執行 ──────────────────────────────────────────────
            # 將稀疏 waypoints 內插成密集軌跡，每個 simulation timestep 送一個點
            dense = interpolate_trajectory(waypoints, timestep, travel_time)
            final_cid = f"{cid}_final"
            clear_receiver(status_receiver)
            arm_ok = True

            for step_i, wp in enumerate(dense):
                is_last = (step_i == len(dense) - 1)
                wp_cid  = final_cid if is_last else f"{cid}_s{step_i}"

                if not send_waypoint(emitter, wp, wp_cid):
                    print("[Validator] 找不到 arm emitter，停止")
                    stop_ros2_bridge(bridge_proc)
                    return

                if supervisor.step(timestep) == -1:
                    arm_ok = False
                    break

            # 等手臂在終點穩定
            if arm_ok:
                if not wait_for_arm_arrival(supervisor, timestep, status_receiver, final_cid, travel_time):
                    print(f"[Validator] 最終位置 timeout，回 home ...")
                    arm_ok = False

            current_joints = list(dense[-1]) if dense else list(target_rad)

            if not arm_ok:
                # 手臂在中途失敗，位置不確定 → 必須回 HOME
                _reset_home(supervisor, timestep, emitter, status_receiver)
                current_joints           = HOME_POSE_RAD[:]
                baseline_ur5e_contacts   = contact_count(ur5e_node, include_descendants=True)
                baseline_camera_contacts = contact_count(camera_node, include_descendants=True)
                results.append(_make_record(candidate, joint_deg, plan_success, plan_error,
                                            n_waypoints, "waypoint_timeout"))
                continue

            if POST_ARRIVAL_PAUSE_SEC > 0.0:
                if not wait_seconds(supervisor, timestep, POST_ARRIVAL_PAUSE_SEC):
                    break

            # ── 幾何 & 碰撞檢測 ──────────────────────────────────────────────
            ray = camera_ray_report(camera_node)
            ur5e_contacts   = contact_count(ur5e_node, include_descendants=True)
            camera_contacts = contact_count(camera_node, include_descendants=True)
            contact_delta        = max(0, ur5e_contacts   - baseline_ur5e_contacts)
            camera_contact_delta = max(0, camera_contacts - baseline_camera_contacts)

            geo_ok  = True
            reasons = []
            if ray["ray_projection_m"] <= 0.0:
                geo_ok = False; reasons.append("target behind camera")
            if ray["ray_miss_m"] > MAX_RAY_MISS_M:
                geo_ok = False; reasons.append(f"ray miss {ray['ray_miss_m']*1000:.1f} mm")
            if ray["roll_err_deg"] > MAX_CAMERA_ROLL_ERROR_DEG:
                geo_ok = False; reasons.append(f"roll {ray['roll_err_deg']:.1f} deg")
            if ray["ray_origin_m"][2] < MIN_CAMERA_Z_M:
                geo_ok = False; reasons.append(f"camera z {ray['ray_origin_m'][2]:.3f} m")
            if contact_delta > 0 or camera_contact_delta > 0:
                geo_ok = False
                reasons.append(f"collision ur5e+{contact_delta} camera+{camera_contact_delta}")

            overall_ok = geo_ok and (plan_success is not False)

            record = {
                "id":        candidate["id"],
                "ok":        overall_ok,
                "reasons":   reasons,
                "joint_deg": joint_deg,
                "planning": {
                    "enabled":   bridge_proc is not None,
                    "success":   plan_success,
                    "waypoints": n_waypoints,
                    "error":     plan_error,
                },
                "ray":      ray,
                "contacts": {
                    "ur5e": ur5e_contacts, "camera": camera_contacts,
                    "ur5e_delta": contact_delta, "camera_delta": camera_contact_delta,
                },
                "source": candidate.get("source", ""),
                "meta":   candidate.get("meta", {}),
            }
            results.append(record)
            if overall_ok:
                validated.append(record)
                print(f"  PASS  ray={ray['ray_miss_m']*1000:.2f}mm  roll={ray['roll_err_deg']:.1f}deg  "
                      f"plan={'ok' if plan_success else '-'}  contacts={contact_delta}")
            else:
                print(f"  FAIL  {', '.join(reasons) or ('plan failed' if plan_success is False else 'unknown')}")
                if contact_delta > 0 or camera_contact_delta > 0:
                    print("[Validator] 碰撞 → 回 home 重置物理 ...")
                    _reset_home(supervisor, timestep, emitter, status_receiver)
                    current_joints           = HOME_POSE_RAD[:]
                    baseline_ur5e_contacts   = contact_count(ur5e_node, include_descendants=True)
                    baseline_camera_contacts = contact_count(camera_node, include_descendants=True)

    finally:
        stop_ros2_bridge(bridge_proc)

    # ── 寫入結果 ──────────────────────────────────────────────────────────────
    OUTPUT_PATH.write_text(
        json.dumps({
            "target_m": TARGET_M,
            "x_offset_m": _X_OFFSET,
            "radius_m": _RADIUS,
            "ws_offset_m": _WS_OFFSET,
            "ycb_object_name": YCB_OBJECT_NAME,
            "ycb_object_pos_m": TARGET_M,
            "ycb_object_rotation_axis_angle": YCB_OBJECT_ROTATION,
            "max_ray_miss_m": MAX_RAY_MISS_M,
            "max_camera_roll_error_deg": MAX_CAMERA_ROLL_ERROR_DEG,
            "min_camera_z_m": MIN_CAMERA_Z_M,
            "moveit_enabled": bridge_proc is not None,
            "validated_count": len(validated),
            "validated": validated,
            "all_results": results,
        }, indent=2),
        encoding="utf-8",
    )

    # ── 摘要 ──────────────────────────────────────────────────────────────────
    print("\n" + "═" * 58)
    print("  Viewpoint Validator 結果摘要")
    print("═" * 58)
    n_skip = sum(1 for r in results if any("plan_failed" in s for s in r.get("reasons", [])))
    n_fail = len(results) - len(validated) - n_skip
    print(f"  候選點: {len(results)}  |  通過: {len(validated)}  |  "
          f"跳過(規劃失敗): {n_skip}  |  失敗: {n_fail}  |  "
          f"MoveIt: {'已啟用' if bridge_proc is not None else '未啟用'}")
    print("─" * 58)
    print(f"  {'id':^6}  {'規劃':^6}  {'WPs':^4}  {'碰撞':^4}  {'Ray':^6}  {'Roll':^6}  結果")
    print("─" * 58)
    for r in results:
        p   = r["planning"]
        c   = r.get("contacts", {})
        ray = r.get("ray", {})
        plan_str = {True: "成功", False: "失敗", None: "跳過"}.get(p["success"], "-")
        coll_str = f"+{c['ur5e_delta']}" if c.get("ur5e_delta") else "  ok"
        miss_str = f"{ray.get('ray_miss_m', 0)*1000:.1f}mm" if ray else "  -"
        roll_str = f"{ray.get('roll_err_deg', 0):.1f}°"     if ray else "  -"
        if r["ok"]:
            ok_str = "PASS"
        elif any("plan_failed" in s for s in r.get("reasons", [])):
            ok_str = "SKIP"
        else:
            ok_str = "FAIL"
        reason   = f"  ({', '.join(r['reasons'][:1])})" if r["reasons"] else ""
        print(f"  {str(r['id']):^6}  {plan_str:^6}  {p['waypoints']:^4}  "
              f"{coll_str:^4}  {miss_str:^6}  {roll_str:^6}  {ok_str}{reason}")
    print("═" * 58)
    print(f"  通過 {len(validated)}/{len(results)} 個候選點")
    print(f"  結果已寫入 {OUTPUT_PATH}")
    shutil.copy2(OUTPUT_PATH, OUTPUT_PATH_LATEST)
    print(f"  最新結果: {OUTPUT_PATH_LATEST.name}")
    print("═" * 58 + "\n")


def _reset_home(supervisor, timestep, emitter, receiver):
    send_waypoint(emitter, HOME_POSE_RAD, "home_reset")
    wait_for_arm_arrival(supervisor, timestep, receiver, "home_reset", 30.0)
    wait_seconds(supervisor, timestep, 2.0)


def _make_record(candidate, joint_deg, plan_success, plan_error, plan_waypoints, fail_reason):
    return {
        "id":        candidate["id"],
        "ok":        False,
        "reasons":   [fail_reason],
        "joint_deg": joint_deg,
        "planning": {
            "enabled":   True,
            "success":   plan_success,
            "waypoints": plan_waypoints,
            "error":     plan_error,
        },
        "ray": {},
        "contacts": {
            "ur5e": 0, "camera": 0,
            "ur5e_delta": 0, "camera_delta": 0,
        },
        "source": candidate.get("source", ""),
        "meta":   candidate.get("meta", {}),
    }


if __name__ == "__main__":
    main()
