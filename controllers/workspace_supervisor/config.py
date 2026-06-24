"""Configuration for drawing the UR5e workspace hemisphere in Webots."""

# DEF name of the UR5e node in the world file.
UR5E_DEF = "UR5E"

# Fallback base pose if the DEF node cannot be found.
ROBOT_BASE_X = -0.5
ROBOT_BASE_Y = 0.0
ROBOT_BASE_Z = 0.0

# The workspace sphere center is offset upward from the tabletop.
# 163 mm comes from the UR5e working-area drawing.
WORKSPACE_CENTER_Z = 0.163

# Recommended reach from the drawing: diameter 1700 mm.
WORKSPACE_RADIUS = 0.85

# Only keep the portion at or above the tabletop.
TABLE_HEIGHT_Z = 0.0

# Capture hemisphere center in world coordinates.
# 已對齊鎖定值:相機繞行半球 cam_r=0.65,look-at 中心 = 世界 [0.15, 0, 0]。
CAPTURE_CENTER_X = 0.15
CAPTURE_CENTER_Y = 0.0
CAPTURE_CENTER_Z = 0.0

# Visual style.
WORKSPACE_DEF = "UR5E_WORKSPACE_HEMISPHERE"
WORKSPACE_NAME = "ur5e_workspace_hemisphere"
COLOR_RGB = (0.0, 0.75, 1.0)
TRANSPARENCY = 0.0

# Smaller hemisphere used for camera capture viewpoints (= 拍攝球, cam_r).
CAPTURE_WORKSPACE_DEF = "UR5E_CAPTURE_HEMISPHERE"
CAPTURE_WORKSPACE_NAME = "ur5e_capture_hemisphere"
CAPTURE_RADIUS = 0.65
CAPTURE_COLOR_RGB = (1.0, 0.65, 0.0)

# Object workspace sphere (= 工作球):YCB 物體可擺放區,ws_r ≈ cam_r − 淨空。
# 由 find_capture_workspace 定出:中心 [0.15,0,0]、半徑 0.355 m。
WS_OBJECT_DEF = "YCB_OBJECT_WORKSPACE"
WS_OBJECT_NAME = "ycb_object_workspace"
WS_OBJECT_CENTER_X = 0.15
WS_OBJECT_CENTER_Y = 0.0
WS_OBJECT_CENTER_Z = 0.0
WS_OBJECT_RADIUS = 0.355
WS_OBJECT_COLOR_RGB = (0.0, 1.0, 0.3)

# Sampling density for the wireframe.
LATITUDE_BANDS = 8
LONGITUDE_BANDS = 12
SEGMENTS_PER_RING = 72
