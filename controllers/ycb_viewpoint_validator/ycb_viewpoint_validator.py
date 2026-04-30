from controller import Supervisor
import importlib.util
import json
import math
import os
import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
FOUR_VIEW_DIR = REPO_ROOT / "controllers" / "ycb_supervisor_capture"
UR5E_TEST_DIR = REPO_ROOT / "controllers" / "ur5e_test_controller"

for path in (FOUR_VIEW_DIR, UR5E_TEST_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import candidate_viewpoint_config as planner_config  # noqa: E402


ARM_COMMAND_EMITTER = "arm_command_emitter"
ARM_STATUS_RECEIVER = "arm_status_receiver"
CANDIDATE_PATH = FOUR_VIEW_DIR / "candidate_viewpoints.json"
OUTPUT_PATH = FOUR_VIEW_DIR / "validated_viewpoints.json"
TARGET_M = planner_config.OBJECT_CENTER_M
CAMERA_AIM_AXIS_LOCAL = planner_config.CAMERA_AIM_AXIS_LOCAL
CAMERA_UP_AXIS_LOCAL = planner_config.CAMERA_UP_AXIS_LOCAL
CAMERA_SENSOR_OFFSET_LOCAL = planner_config.T_D455_TO_SENSOR_M
WORLD_UP_AXIS = planner_config.WORLD_UP_AXIS
WORLD_ROLL_FALLBACK_AXIS = planner_config.WORLD_ROLL_FALLBACK_AXIS
ARM_SETTLE_TIME_SEC = 2.0
ARM_ARRIVAL_TIMEOUT_SEC = 30.0
POST_ARRIVAL_PAUSE_SEC = 0.75
MAX_RAY_MISS_M = 0.005
MAX_CAMERA_ROLL_ERROR_DEG = planner_config.MAX_CAMERA_ROLL_ERROR_DEG
MIN_CAMERA_Z_M = 0.08
HOME_POSE_DEG = [0.0, -90.0, 90.0, -90.0, -90.0, 0.0]


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
        with CANDIDATE_PATH.open("r", encoding="utf-8") as file:
            data = json.load(file)
        candidates = []
        for index, item in enumerate(data, start=1):
            if isinstance(item, dict) and "joint_deg" in item:
                candidates.append({
                    "id": item.get("id", index),
                    "joint_deg": item["joint_deg"],
                    "source": str(CANDIDATE_PATH),
                    "meta": item,
                })
        if candidates:
            return candidates
    return load_ur5e_camera_poses()


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
            message = receiver.getString()
            receiver.nextPacket()
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                continue
            if (
                data.get("status") == "arrived"
                and str(data.get("command_id")) == str(command_id)
            ):
                print(
                    "[Validator] Arm arrived "
                    f"(max_error={float(data.get('max_error_rad', 0.0)):.4f} rad)"
                )
                return True
    print(f"[Validator] Arm arrival timeout for command {command_id}; stopping sequence.")
    return False


def send_joint_command(emitter, joint_deg, command_id):
    payload = json.dumps({"joint_deg": joint_deg, "command_id": str(command_id)})
    emitter.send(payload.encode("utf-8"))


def mat_vec_mul(matrix9, vector3):
    return [
        matrix9[0] * vector3[0] + matrix9[1] * vector3[1] + matrix9[2] * vector3[2],
        matrix9[3] * vector3[0] + matrix9[4] * vector3[1] + matrix9[5] * vector3[2],
        matrix9[6] * vector3[0] + matrix9[7] * vector3[1] + matrix9[8] * vector3[2],
    ]


def norm(vector):
    length = math.sqrt(sum(value * value for value in vector))
    if length < 1e-12:
        return [0.0, 0.0, 0.0]
    return [value / length for value in vector]


def subtract(a, b):
    return [a[i] - b[i] for i in range(3)]


def add(a, b):
    return [a[i] + b[i] for i in range(3)]


def scale(vector, scalar):
    return [value * scalar for value in vector]


def dot(a, b):
    return sum(a[i] * b[i] for i in range(3))


def project_onto_view_plane(axis, ray_axis, fallback_axis=None):
    projected = subtract(axis, scale(ray_axis, dot(axis, ray_axis)))
    if math.sqrt(dot(projected, projected)) > 1e-9:
        return norm(projected)
    if fallback_axis is None:
        return [0.0, 0.0, 0.0]
    fallback = subtract(fallback_axis, scale(ray_axis, dot(fallback_axis, ray_axis)))
    return norm(fallback)


def roll_reference_axis(ray_axis):
    desired = project_onto_view_plane(
        norm(WORLD_UP_AXIS),
        ray_axis,
        norm(WORLD_ROLL_FALLBACK_AXIS),
    )
    if math.sqrt(dot(desired, desired)) > 1e-9:
        return desired
    return [0.0, 1.0, 0.0]


def roll_error_deg(orientation, ray_axis):
    camera_up = norm(mat_vec_mul(orientation, CAMERA_UP_AXIS_LOCAL))
    desired_up = roll_reference_axis(ray_axis)
    actual_up = project_onto_view_plane(camera_up, ray_axis, desired_up)
    return math.degrees(math.acos(max(-1.0, min(1.0, dot(actual_up, desired_up)))))


def distance(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def camera_ray_report(camera_node):
    body_position = camera_node.getPosition()
    orientation = camera_node.getOrientation()
    sensor_offset = mat_vec_mul(orientation, CAMERA_SENSOR_OFFSET_LOCAL)
    ray_origin = add(body_position, sensor_offset)
    ray_axis = norm(mat_vec_mul(orientation, CAMERA_AIM_AXIS_LOCAL))
    target_delta = subtract(TARGET_M, ray_origin)
    projection = dot(target_delta, ray_axis)
    closest = add(ray_origin, scale(ray_axis, max(0.0, projection)))
    miss = distance(TARGET_M, closest)
    angle = math.degrees(math.acos(max(-1.0, min(1.0, dot(norm(target_delta), ray_axis)))))
    return {
        "body_position_m": body_position,
        "ray_origin_m": ray_origin,
        "ray_axis_world": ray_axis,
        "ray_projection_m": projection,
        "ray_miss_m": miss,
        "target_angle_deg": angle,
        "roll_err_deg": roll_error_deg(orientation, ray_axis),
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


def main():
    supervisor = Supervisor()
    timestep = int(supervisor.getBasicTimeStep())
    emitter = supervisor.getDevice(ARM_COMMAND_EMITTER)
    status_receiver = supervisor.getDevice(ARM_STATUS_RECEIVER)
    if status_receiver is not None:
        status_receiver.enable(timestep)
    else:
        print(f"[Validator] 找不到 {ARM_STATUS_RECEIVER}，改用時間等待。")
    ur5e_node = supervisor.getFromDef(planner_config.UR5E_DEF)
    camera_node = supervisor.getFromDef(planner_config.CAMERA_DEF)

    if emitter is None:
        print("[Validator] 找不到 arm_command_emitter")
        return
    if ur5e_node is None:
        print(f"[Validator] 找不到 DEF {planner_config.UR5E_DEF}")
        return
    if camera_node is None:
        print(f"[Validator] 找不到 DEF {planner_config.CAMERA_DEF}")
        return

    candidates = load_candidates()
    print(f"[Validator] Candidates: {len(candidates)}")
    print(f"[Validator] Target: {TARGET_M}")
    print(f"[Validator] Output: {OUTPUT_PATH}")

    wait_seconds(supervisor, timestep, 3.0)
    baseline_ur5e_contacts = contact_count(ur5e_node, include_descendants=True)
    baseline_camera_contacts = contact_count(camera_node, include_descendants=True)

    results = []
    validated = []
    for index, candidate in enumerate(candidates, start=1):
        joint_deg = [float(value) for value in candidate["joint_deg"]]
        command_id = candidate["id"]
        print(f"[Validator] Testing {index}/{len(candidates)} id={candidate['id']} ...")
        clear_receiver(status_receiver)
        send_joint_command(emitter, joint_deg, command_id)
        if not wait_for_arm_arrival(
            supervisor,
            timestep,
            status_receiver,
            command_id,
            ARM_ARRIVAL_TIMEOUT_SEC,
        ):
            print(f"[Validator] Timeout id={command_id} — skipping, returning home ...")
            clear_receiver(status_receiver)
            send_joint_command(emitter, HOME_POSE_DEG, "home_reset")
            wait_for_arm_arrival(supervisor, timestep, status_receiver, "home_reset", ARM_ARRIVAL_TIMEOUT_SEC)
            wait_seconds(supervisor, timestep, 2.0)
            baseline_ur5e_contacts = contact_count(ur5e_node, include_descendants=True)
            baseline_camera_contacts = contact_count(camera_node, include_descendants=True)
            results.append({
                "id": candidate["id"],
                "ok": False,
                "reasons": ["timeout"],
                "joint_deg": joint_deg,
                "source": candidate.get("source", ""),
                "meta": candidate.get("meta", {}),
            })
            continue
        if POST_ARRIVAL_PAUSE_SEC > 0.0:
            if not wait_seconds(supervisor, timestep, POST_ARRIVAL_PAUSE_SEC):
                break

        ray = camera_ray_report(camera_node)
        ur5e_contacts = contact_count(ur5e_node, include_descendants=True)
        camera_contacts = contact_count(camera_node, include_descendants=True)
        contact_delta = max(0, ur5e_contacts - baseline_ur5e_contacts)
        camera_contact_delta = max(0, camera_contacts - baseline_camera_contacts)

        ok = True
        reasons = []
        if ray["ray_projection_m"] <= 0.0:
            ok = False
            reasons.append("target behind camera ray")
        if ray["ray_miss_m"] > MAX_RAY_MISS_M:
            ok = False
            reasons.append(f"ray miss {ray['ray_miss_m'] * 1000.0:.1f} mm")
        if ray["roll_err_deg"] > MAX_CAMERA_ROLL_ERROR_DEG:
            ok = False
            reasons.append(f"roll {ray['roll_err_deg']:.1f} deg")
        if ray["ray_origin_m"][2] < MIN_CAMERA_Z_M:
            ok = False
            reasons.append(f"camera z {ray['ray_origin_m'][2]:.3f} m")
        if contact_delta > 0 or camera_contact_delta > 0:
            ok = False
            reasons.append(
                f"contacts ur5e+{contact_delta} camera+{camera_contact_delta}"
            )

        record = {
            "id": candidate["id"],
            "ok": ok,
            "reasons": reasons,
            "joint_deg": joint_deg,
            "ray": ray,
            "contacts": {
                "ur5e": ur5e_contacts,
                "camera": camera_contacts,
                "ur5e_delta": contact_delta,
                "camera_delta": camera_contact_delta,
            },
            "source": candidate.get("source", ""),
            "meta": candidate.get("meta", {}),
        }
        results.append(record)
        if ok:
            validated.append(record)
            print(
                f"  PASS ray_miss={ray['ray_miss_m'] * 1000.0:.2f} mm "
                f"roll={ray['roll_err_deg']:.1f} deg contacts={contact_delta}"
            )
        else:
            print(f"  FAIL {', '.join(reasons)}")
            if contact_delta > 0 or camera_contact_delta > 0:
                print("[Validator] Collision detected — returning to home to reset physics ...")
                clear_receiver(status_receiver)
                send_joint_command(emitter, HOME_POSE_DEG, "home_reset")
                if wait_for_arm_arrival(
                    supervisor, timestep, status_receiver, "home_reset", ARM_ARRIVAL_TIMEOUT_SEC
                ):
                    wait_seconds(supervisor, timestep, 2.0)
                    baseline_ur5e_contacts = contact_count(ur5e_node, include_descendants=True)
                    baseline_camera_contacts = contact_count(camera_node, include_descendants=True)
                    print(
                        f"[Validator] Home reset done. "
                        f"New baseline: ur5e={baseline_ur5e_contacts} camera={baseline_camera_contacts}"
                    )

    OUTPUT_PATH.write_text(
        json.dumps({
            "target_m": TARGET_M,
            "max_ray_miss_m": MAX_RAY_MISS_M,
            "max_camera_roll_error_deg": MAX_CAMERA_ROLL_ERROR_DEG,
            "min_camera_z_m": MIN_CAMERA_Z_M,
            "validated_count": len(validated),
            "validated": validated,
            "all_results": results,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"[Validator] Validated {len(validated)}/{len(results)} poses.")
    print(f"[Validator] Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
