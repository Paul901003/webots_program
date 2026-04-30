import cv2
import json
import numpy as np
import math
import shutil
from pathlib import Path
from controller import Robot

FRAME_WARMUP_STEPS = 2
REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_IMAGES_DIR = REPO_ROOT / "Grounded-Segment-Anything" / "test_images"


def parse_sampling_period_ms(robot: Robot, default_period_ms: int) -> int:
    custom_data = robot.getCustomData().strip()
    if not custom_data:
        return default_period_ms

    for item in custom_data.split(";"):
        key, sep, value = item.partition("=")
        if sep and key.strip().lower() == "fps":
            try:
                fps = max(1, int(value.strip()))
                return max(default_period_ms, int(round(1000 / fps)))
            except ValueError:
                break

    return default_period_ms


def parse_custom_data(raw_text: str):
    data = {}
    for item in raw_text.split(";"):
        key, sep, value = item.partition("=")
        if sep:
            data[key.strip().lower()] = value.strip()
    return data


def build_capture_dir(root_name: str = "captures") -> Path:
    capture_dir = TEST_IMAGES_DIR / root_name
    capture_dir.mkdir(parents=True, exist_ok=True)
    return capture_dir


def sanitize_filename_part(value: str) -> str:
    cleaned = []
    for char in value.strip():
        if char.isalnum() or char in ("_", "-", "+"):
            cleaned.append(char)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("_") or "scene"


def format_vec3(values) -> str:
    return f"x={values[0]:.6f}, y={values[1]:.6f}, z={values[2]:.6f}"


def format_rpy_rad_deg(roll: float, pitch: float, yaw: float) -> str:
    return (
        f"roll={roll:.6f} rad ({math.degrees(roll):.2f} deg), "
        f"pitch={pitch:.6f} rad ({math.degrees(pitch):.2f} deg), "
        f"yaw={yaw:.6f} rad ({math.degrees(yaw):.2f} deg)"
    )


def get_separator_line(fill_char: str = "=") -> str:
    terminal_width = shutil.get_terminal_size(fallback=(80, 20)).columns
    return fill_char * max(terminal_width - 1, 20)


def make_depth_colormap(depth_array: np.ndarray) -> np.ndarray:
    valid_mask = np.isfinite(depth_array)
    if not np.any(valid_mask):
        return np.zeros(depth_array.shape, dtype=np.uint8)

    sanitized = np.where(valid_mask, depth_array, 0.0).astype(np.float32)
    valid_values = sanitized[valid_mask]
    min_depth = float(valid_values.min())
    max_depth = float(valid_values.max())

    if max_depth - min_depth < 1e-9:
        depth_gray = np.zeros(depth_array.shape, dtype=np.uint8)
    else:
        normalized = (sanitized - min_depth) / (max_depth - min_depth)
        normalized = np.clip(normalized * 255.0, 0, 255)
        depth_gray = normalized.astype(np.uint8)

    depth_gray[~valid_mask] = 0
    return cv2.applyColorMap(depth_gray, cv2.COLORMAP_JET)


def save_capture(scene_dir: Path, view_name: str, rgb_image, depth_array, position, roll_pitch_yaw):
    scene_dir.mkdir(parents=True, exist_ok=True)
    rgb_path = scene_dir / f"{view_name}.png"
    depth_vis_path = scene_dir / f"{view_name}_depth.png"
    depth_raw_path = scene_dir / f"{view_name}_depth.npy"
    meta_path = scene_dir / f"{view_name}_pose.json"

    rgb_to_save = np.ascontiguousarray(rgb_image)
    if not cv2.imwrite(str(rgb_path), rgb_to_save):
        raise RuntimeError(f"RGB image save failed: {rgb_path}")

    depth_colormap = make_depth_colormap(depth_array)
    depth_to_save = np.ascontiguousarray(depth_colormap)
    if not cv2.imwrite(str(depth_vis_path), depth_to_save):
        raise RuntimeError(f"Depth visualization save failed: {depth_vis_path}")
    np.save(depth_raw_path, depth_array)

    roll, pitch, yaw = roll_pitch_yaw
    metadata = {
        "capture_name": view_name,
        "position_m": {
            "x": float(position[0]),
            "y": float(position[1]),
            "z": float(position[2]),
        },
        "rotation_rpy_rad": {
            "roll": float(roll),
            "pitch": float(pitch),
            "yaw": float(yaw),
        },
        "rotation_rpy_deg": {
            "roll": float(math.degrees(roll)),
            "pitch": float(math.degrees(pitch)),
            "yaw": float(math.degrees(yaw)),
        },
        "files": {
            "rgb": rgb_path.name,
            "depth_visualization": depth_vis_path.name,
            "depth_raw_npy": depth_raw_path.name,
        },
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return rgb_path, depth_vis_path, depth_raw_path, meta_path


def main():
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())
    sampling_period = parse_sampling_period_ms(robot, timestep)

    base_name = "IntelRealsenseD455"
    camera_name = f"{base_name}_rgb"
    depth_name = f"{base_name}_depth"
    gps_name = f"{base_name}_gps"
    imu_name = f"{base_name}_imu"

    camera = robot.getDevice(camera_name)
    range_finder = robot.getDevice(depth_name)
    gps = robot.getDevice(gps_name)
    imu = robot.getDevice(imu_name)
    capture_dir = build_capture_dir()

    last_capture_token = None
    latest_raw_img = None
    latest_raw_depth = None
    latest_position = None
    latest_roll_pitch_yaw = None
    pending_capture = None

    if camera and range_finder and gps and imu:
        camera.enable(sampling_period)
        range_finder.enable(sampling_period)
        gps.enable(sampling_period)
        imu.enable(sampling_period)
        print(f"已啟用自動拍攝相機: {camera_name}")
        print(f"拍照輸出資料夾: {capture_dir}")
    else:
        print("找不到相機或姿態設備，請確認名稱是否正確。")
        return

    while robot.step(timestep) != -1:
        raw_img = camera.getImage()
        raw_depth = range_finder.getRangeImage()
        position = gps.getValues()
        roll_pitch_yaw = imu.getRollPitchYaw()

        if raw_img:
            latest_raw_img = raw_img
        if raw_depth:
            latest_raw_depth = raw_depth
        if position:
            latest_position = position
        if roll_pitch_yaw:
            latest_roll_pitch_yaw = roll_pitch_yaw

        raw_data = robot.getCustomData().strip()
        data = parse_custom_data(raw_data)
        capture_token = data.get("capture_token")
        view = sanitize_filename_part(data.get("view", "0"))
        label = sanitize_filename_part(data.get("label", "scene"))
        capture_root = sanitize_filename_part(data.get("capture_root", "captures"))
        num_views_raw = data.get("num_views", "").strip()
        if num_views_raw.isdigit():
            label = f"{label}_{num_views_raw}views"
        active_capture_dir = build_capture_dir(capture_root)

        pending_token = pending_capture["token"] if pending_capture is not None else None
        if capture_token and capture_token != last_capture_token and capture_token != pending_token:
            pending_capture = {
                "token": capture_token,
                "view": view,
                "label": label,
                "capture_dir": active_capture_dir,
                "warmup_steps": FRAME_WARMUP_STEPS,
            }

        if pending_capture is not None:
            if latest_raw_img and latest_raw_depth and latest_position and latest_roll_pitch_yaw:
                if pending_capture["warmup_steps"] > 0:
                    pending_capture["warmup_steps"] -= 1
                    continue

                img_array = np.frombuffer(latest_raw_img, dtype=np.uint8).reshape(
                    (camera.getHeight(), camera.getWidth(), 4)
                )
                rgb_image = img_array[:, :, :3].copy()
                depth_array = np.array(latest_raw_depth, dtype=np.float32).reshape(
                    (range_finder.getHeight(), range_finder.getWidth())
                )
                scene_dir = pending_capture["capture_dir"] / pending_capture["label"]
                view_name = pending_capture["view"]
                capture_dir_name = pending_capture["capture_dir"].name
                try:
                    rgb_path, depth_vis_path, depth_raw_path, meta_path = save_capture(
                        scene_dir,
                        view_name,
                        rgb_image,
                        depth_array,
                        latest_position,
                        latest_roll_pitch_yaw,
                    )
                    last_capture_token = pending_capture["token"]
                    pending_capture = None
                    print("\n[Auto Capture]")
                    print(f"根目錄: {capture_dir_name}")
                    print(f"資料夾: {scene_dir.name}")
                    print(f"視角: {view_name}")
                    print(f"位置: {format_vec3(latest_position)} m")
                    print(f"旋轉: {format_rpy_rad_deg(*latest_roll_pitch_yaw)}")
                    print(f"RGB: {rgb_path}")
                    print(f"Depth(vis): {depth_vis_path}")
                    print(f"Depth(raw): {depth_raw_path}")
                    print(f"Pose: {meta_path}")
                    print(get_separator_line())
                except Exception as error:
                    pending_capture = None
                    last_capture_token = capture_token
                    print(f"[Auto Capture] 存檔失敗: {error}")
                    print(get_separator_line())


if __name__ == "__main__":
    main()
