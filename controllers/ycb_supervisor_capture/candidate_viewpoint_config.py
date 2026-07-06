"""Editable parameters for generate_candidate_viewpoints.py."""

# Scene file used to derive the actual UR5e/camera/gripper mounting
# relationship. ★A-1 的相機 mount 由此 world 檔的 DEF UR5E_CAMERA translation 解析而來
# （config 下方的 T_FLANGE_TO_D455_M 只是解析失敗時的 fallback）→ 改相機 mount 要改這個 world。
# 指向實際 armmove 拍攝 world，作為單一真相來源。
WORLD_FILE = "../../worlds/ycb_supervisor_four_view_capture_multi.wbt"
UR5E_DEF = "UR5E"
CAMERA_DEF = "UR5E_CAMERA"

# World layout (Z-up, metres).
ROBOT_BASE_M = [-0.4, 0.0, 0.0]
# Target point the camera should look at, in world coordinates.
# In this scene, +X is the arm's forward direction and the table target center
# is at the world origin.
OBJECT_CENTER_M = [0.0, 0.0, 0.0]
# YCB 物體資訊（供 generate_labels.py 使用）
YCB_OBJECT_NAME     = "024_bowl"         # YCB 物體資料夾名稱
YCB_OBJECT_ROTATION = [0, 1, 0, 0]      # Webots axis-angle (ax,ay,az,angle_rad)，0=不旋轉
CAPTURE_ROOT        = "captures_single"  # scene_plan.json 中的 capture_root
TABLE_Z_M = 0.0
LINK_CLEARANCE_M = 0.06

# Camera-to-flange geometry.
# The official Webots UR5e toolSlot is 0.1 m along wrist_3_link local +Y.
WEBOTS_TOOL_SLOT_TRANSLATION_M = [0.0, 0.1, 0.0]

# toolSlot mounts the D455 body at translation 0 -0.03 0.10 / rotation 0 0 1 1.5708.
# （相機沿 toolSlot +Z 由 5cm 改為 10cm;multicam world 對應 mount=0 -0.09994 -0.02989）
# The Camera/RangeFinder/GPS children inside IntelRealsenseD455.proto sit at
# translation 0.005 0 0 in the D455 body frame.
R_FLANGE_TO_CAM = [
    [0.0, -1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
]
T_FLANGE_TO_D455_M = [0.0, -0.03, 0.10]
T_D455_TO_SENSOR_M = [0.005, 0.0, 0.0]

# Local axis that should point at OBJECT_CENTER_M.
# If the visible D455 body appears sideways, change this axis and rerun planner.
# Common candidates:
#   [1, 0, 0]   D455 body/front axis in this scene
#   [0, 0, -1]  Webots Camera node optical axis convention
CAMERA_AIM_AXIS_LOCAL = [1.0, 0.0, 0.0]

# Local camera axis that should stay visually upright after the aim axis points
# at OBJECT_CENTER_M.  Roll is measured around CAMERA_AIM_AXIS_LOCAL by
# projecting this axis and WORLD_UP_AXIS onto the plane perpendicular to the
# camera ray; 0 deg means the camera image is upright relative to world +Z.
CAMERA_UP_AXIS_LOCAL = [0.0, 0.0, 1.0]
WORLD_UP_AXIS = [0.0, 0.0, 1.0]
# Fallback reference when the ray is almost parallel to WORLD_UP_AXIS.
WORLD_ROLL_FALLBACK_AXIS = [0.0, 1.0, 0.0]
CAMERA_ROLL_WEIGHT = 3.0
MAX_CAMERA_ROLL_ERROR_DEG = 10.0

# Hemisphere sampling.
HEMISPHERE_RADIUS_M = 0.65
HEMISPHERE_RADII_M = [0.65]  # 多半徑模式用
ELEVATION_ANGLES_DEG = [20, 30, 45, 60, 75, 90]
AZIMUTH_STEPS = 24  # 每個仰角的方位角數量
EXTRA_VIEWPOINTS_DEG = [
    (20, 160, None),
    (20, 170, None),
    (20, 190, None),
    (20, 200, None),
    (30, 160, None),
    (30, 170, None),
    (30, 190, None),
    (30, 200, None),
]  # 額外指定的 (仰角, 方位角, 半徑m) 視角；半徑 None 表示套用所有半徑

# Joint limits and IK solution preference (degrees).
JOINT_LIMITS_DEG = [
    (-180, 180),   # J1 shoulder pan
    (-175, -5),    # J2 shoulder lift
    (-5, 175),     # J3 elbow
    (-175, 5),     # J4 wrist 1
    (-180, 180),   # J5 wrist 2
    (-180, 180),   # J6 wrist 3
]
REFERENCE_DEG = [0.0, -90.0, 90.0, -90.0, -90.0, 0.0]
NUM_OUTPUT_POSES = 12

# Self-collision capsule radii (mm), one radius per UR5e link capsule.
LINK_RADII_MM = [65.0, 60.0, 55.0, 40.0, 40.0, 35.0]

# Existing poses kept for optional local sanity checks.
EXISTING_POSES_DEG = {
}
