import cv2
import json
import numpy as np
import math
import shutil
from datetime import datetime
from pathlib import Path
from controller import Robot, Keyboard

KEY_DEBOUNCE_MS = 250


def parse_sampling_period_ms(robot: Robot, default_period_ms: int) -> int:
    custom_data = robot.getCustomData().strip()
    if not custom_data:
        return default_period_ms

    for item in custom_data.split(';'):
        key, sep, value = item.partition('=')
        if sep and key.strip().lower() == "fps":
            try:
                fps = max(1, int(value.strip()))
                return max(default_period_ms, int(round(1000 / fps)))
            except ValueError:
                break

    return default_period_ms


def build_capture_dir() -> Path:
    capture_dir = Path(__file__).resolve().parent / "captures"
    capture_dir.mkdir(parents=True, exist_ok=True)
    return capture_dir


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


def save_capture(capture_dir: Path, rgb_image, depth_array, position, roll_pitch_yaw):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    rgb_path = capture_dir / f"rgb_{timestamp}.png"
    depth_vis_path = capture_dir / f"depth_{timestamp}.png"
    depth_raw_path = capture_dir / f"depth_{timestamp}.npy"
    meta_path = capture_dir / f"pose_{timestamp}.json"

    cv2.imwrite(str(rgb_path), rgb_image)

    depth_colormap = make_depth_colormap(depth_array)
    cv2.imwrite(str(depth_vis_path), depth_colormap)
    np.save(depth_raw_path, depth_array)

    roll, pitch, yaw = roll_pitch_yaw
    metadata = {
        "timestamp": timestamp,
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
    # 建立 Robot 實例
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())
    sampling_period = parse_sampling_period_ms(robot, timestep)

    # 啟用 Webots 鍵盤監聽
    keyboard = robot.getKeyboard()
    keyboard.enable(timestep)

    # 根據 PROTO 檔案定義的名稱規則
    base_name = "IntelRealsenseD455"
    camera_name = f"{base_name}_rgb"
    depth_name = f"{base_name}_depth"
    gps_name = f"{base_name}_gps"
    imu_name = f"{base_name}_imu"

    # 取得相機設備
    camera = robot.getDevice(camera_name)
    range_finder = robot.getDevice(depth_name)
    gps = robot.getDevice(gps_name)
    imu = robot.getDevice(imu_name)
    capture_dir = build_capture_dir()

    # 設定初始狀態為「開啟」
    camera_active = True
    max_range = 6.0  # 預設值，若成功啟用會更新
    last_key_time_ms = -KEY_DEBOUNCE_MS

    if camera and range_finder and gps and imu:
        camera.enable(sampling_period)
        range_finder.enable(sampling_period)
        gps.enable(sampling_period)
        imu.enable(sampling_period)
        max_range = range_finder.getMaxRange()
        print(f"已啟用彩色相機: {camera_name} (解析度: {camera.getWidth()}x{camera.getHeight()})")
        print(f"已啟用深度相機: {depth_name} (最大偵測距離: {max_range}m)")
        print(f"相機取樣週期: {sampling_period} ms (~{1000 / sampling_period:.2f} FPS)")
        print(f"拍照輸出資料夾: {capture_dir}")
    else:
        print("找不到相機或姿態設備，請確認名稱是否正確。")

    print("\n=========================================")
    print("【重要】請點擊 Webots 的 3D 模擬視窗來操作鍵盤！")
    print("按鍵 [B] : 切換相機與視窗開關")
    print("按鍵 [M] : 拍照、儲存深度圖，並輸出相機座標與三軸旋轉")
    print("=========================================\n")

    # 主控制迴圈
    while robot.step(timestep) != -1:
        current_time_ms = robot.getTime() * 1000.0

        # 讀取 Webots 鍵盤輸入
        key = keyboard.getKey()

        # 處理相機開關邏輯
        if key != -1 and (current_time_ms - last_key_time_ms) >= KEY_DEBOUNCE_MS:
            last_key_time_ms = current_time_ms
            if key in (ord('B'), ord('b')):
                if camera_active:
                    print("關閉相機...")
                    camera.disable()
                    range_finder.disable()
                    gps.disable()
                    imu.disable()
                    cv2.destroyAllWindows() # 關閉所有 OpenCV 視窗
                    camera_active = False
                else:
                    print("開啟相機...")
                    camera.enable(sampling_period)
                    range_finder.enable(sampling_period)
                    gps.enable(sampling_period)
                    imu.enable(sampling_period)
                    camera_active = True
            elif key in (ord('M'), ord('m')):
                if camera_active and camera and range_finder and gps and imu:
                    raw_img = camera.getImage()
                    raw_depth = range_finder.getRangeImage()
                    position = gps.getValues()
                    roll_pitch_yaw = imu.getRollPitchYaw()

                    if raw_img and raw_depth:
                        img_array = np.frombuffer(raw_img, dtype=np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
                        rgb_image = img_array[:, :, :3]
                        depth_array = np.array(raw_depth, dtype=np.float32).reshape((range_finder.getHeight(), range_finder.getWidth()))
                        rgb_path, depth_vis_path, depth_raw_path, meta_path = save_capture(
                            capture_dir,
                            rgb_image,
                            depth_array,
                            position,
                            roll_pitch_yaw,
                        )
                        print("\n[Capture]")
                        print(f"位置: {format_vec3(position)} m")
                        print(f"旋轉: {format_rpy_rad_deg(*roll_pitch_yaw)}")
                        print(f"RGB: {rgb_path}")
                        print(f"Depth(vis): {depth_vis_path}")
                        print(f"Depth(raw): {depth_raw_path}")
                        print(f"Pose: {meta_path}\n")
                        print(get_separator_line())
                    else:
                        print("目前無法拍照，請確認相機影像已更新。")
                        print(get_separator_line())
        # 如果相機處於開啟狀態，才擷取並顯示影像
        if camera_active and camera and range_finder:
            # --- 處理 RGB 彩色影像 ---
            raw_img = camera.getImage()
            if raw_img:
                img_array = np.frombuffer(raw_img, dtype=np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
                img_bgr = img_array[:, :, :3]
                cv2.imshow("RGB Camera", img_bgr)

            # --- 處理 深度影像 (Depth) ---
            raw_depth = range_finder.getRangeImage()
            if raw_depth:
                depth_array = np.array(raw_depth, dtype=np.float32).reshape((range_finder.getHeight(), range_finder.getWidth()))
                depth_colormap = make_depth_colormap(depth_array)
                cv2.imshow("Depth Camera", depth_colormap)

            # 讓 OpenCV 更新畫面 (不綁定按鍵功能)
            cv2.waitKey(1)

if __name__ == '__main__':
    main()
