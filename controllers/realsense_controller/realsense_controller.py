import cv2
import numpy as np
from controller import Robot, Keyboard

def main():
    # 建立 Robot 實例
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    # 啟用 Webots 鍵盤監聽
    keyboard = robot.getKeyboard()
    keyboard.enable(timestep)

    # 根據 PROTO 檔案定義的名稱規則
    base_name = "IntelRealsenseD455"
    camera_name = f"{base_name}_rgb"
    depth_name = f"{base_name}_depth"

    # 取得相機設備
    camera = robot.getDevice(camera_name)
    range_finder = robot.getDevice(depth_name)

    # 設定初始狀態為「開啟」
    camera_active = True
    max_range = 6.0 # 預設值，若成功啟用會更新

    if camera and range_finder:
        camera.enable(timestep)
        range_finder.enable(timestep)
        max_range = range_finder.getMaxRange()
        print(f"已啟用彩色相機: {camera_name} (解析度: {camera.getWidth()}x{camera.getHeight()})")
        print(f"已啟用深度相機: {depth_name} (最大偵測距離: {max_range}m)")
    else:
        print(f"找不到相機設備，請確認名稱是否正確。")

    print("\n=========================================")
    print("【重要】請點擊 Webots 的 3D 模擬視窗來操作鍵盤！")
    print("按鍵 [O] : 開啟相機與視窗")
    print("按鍵 [X] : 關閉相機與視窗")
    print("=========================================\n")

    # 主控制迴圈
    while robot.step(timestep) != -1:
        
        # 讀取 Webots 鍵盤輸入
        key = keyboard.getKey()
        
        # 處理相機開關邏輯
        if key != -1:
            if key == ord('O'):
                if not camera_active:
                    print("開啟相機...")
                    camera.enable(timestep)
                    range_finder.enable(timestep)
                    camera_active = True
            elif key == ord('X'):
                if camera_active:
                    print("關閉相機...")
                    camera.disable()
                    range_finder.disable()
                    cv2.destroyAllWindows() # 關閉所有 OpenCV 視窗
                    camera_active = False

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
                depth_normalized = np.clip(depth_array / max_range * 255, 0, 255).astype(np.uint8)
                depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                cv2.imshow("Depth Camera", depth_colormap)

            # 讓 OpenCV 更新畫面 (不綁定按鍵功能)
            cv2.waitKey(1)

if __name__ == '__main__':
    main()