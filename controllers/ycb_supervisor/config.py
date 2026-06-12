# ── 生成參數 ──────────────────────────────────────────────
# TARGET_OBJECTS 留空時，資料集排程會使用 ALL_OBJECTS。
# 若只想測特定物件集合，直接在 TARGET_OBJECTS 中填入名稱即可。
NUM_OBJECTS  = 3
GRID_COLS    = 3
SPACING      = 0.15
SPAWN_HEIGHT = 0.02
# 參考點（世界座標，平面 x/y）：網格中心會先對齊到這裡
REFERENCE_X  = -1.0
REFERENCE_Y  = 0.0
# 偏移量：在參考點基礎上再平移
X_OFFSET     = 0.0
Z_OFFSET     = 0.1
SPAWN_CLEARANCE = 0.0   # 直接放在靜止高度，不從高處落下（搭配 WorldInfo defaultDamping/bounce0 防滾走）
SPACING_MARGIN = 0.02
ARM_SETTLE_TIME_SEC = 2.0
POST_ARRIVAL_PAUSE_SEC = 0.75
ARM_MOTOR_VELOCITY_RAD_PER_SEC = 1.5
ARM_SETTLE_TIME_BUFFER_SEC = 8.0
ASSET_BASE   = "../urdfs/ycb_assets"

# ── 資料集拍攝排程 ────────────────────────────────────────
# single_and_multi:
#   1. 單物體：每種物體各拍一次
#   2. 多物體：每次隨機/自動挑 3 種物體，直到每種物體至少出現 MULTI_MIN_APPEARANCES 次
DATASET_CAPTURE_MODE = "single_and_multi"
MULTI_OBJECT_COUNT = 3
MULTI_MIN_APPEARANCES = 5

# ── 質量表 ────────────────────────────────────────────────
MASS_TABLE = {
    "002_master_chef_can":       0.414,
    "003_cracker_box":           0.411,
    "004_sugar_box":             0.514,
    "005_tomato_soup_can":       0.349,
    "006_mustard_bottle":        0.603,
    "007_tuna_fish_can":         0.171,
    "008_pudding_box":           0.187,
    "009_gelatin_box":           0.097,
    "010_potted_meat_can":       0.370,
    "011_banana":                0.066,
    "012_strawberry":            0.018,
    "013_apple":                 0.068,
    "014_lemon":                 0.029,
    "015_peach":                 0.098,
    "016_pear":                  0.049,
    "017_orange":                0.047,
    "018_plum":                  0.025,
    "019_pitcher_base":          0.178,
    "021_bleach_cleanser":       1.131,
    "022_windex_bottle":         1.022,
    "024_bowl":                  0.147,
    "025_mug":                   0.118,
    "026_sponge":                0.030,
    "028_skillet_lid":           0.652,
    "029_plate":                 0.279,
    "030_fork":                  0.034,
    "031_spoon":                 0.030,
    "032_knife":                 0.031,
    "033_spatula":               0.052,
    "035_power_drill":           0.895,
    "036_wood_block":            0.729,
    "037_scissors":              0.082,
    "038_padlock":               0.208,
    "040_large_marker":          0.016,
    "042_adjustable_wrench":     0.252,
    "043_phillips_screwdriver":  0.097,
    "044_flat_screwdriver":      0.098,
    "048_hammer":                0.665,
    "050_medium_clamp":          0.059,
    "051_large_clamp":           0.125,
    "052_extra_large_clamp":     0.202,
    "053_mini_soccer_ball":      0.123,
    "054_softball":              0.191,
    "055_baseball":              0.138,
    "056_tennis_ball":           0.058,
    "057_racquetball":           0.041,
    "058_golf_ball":             0.046,
    "059_chain":                 0.100,
    "061_foam_brick":            0.028,
    "062_dice":                  0.005,
    # "063-a_marbles":             0.040,
    # "063-b_marbles":             0.040,
    "065-a_cups":                0.013,
    "065-b_cups":                0.014,
    "065-c_cups":                0.017,
    "065-d_cups":                0.019,
    "065-e_cups":                0.021,
    "065-f_cups":                0.026,
    "065-g_cups":                0.028,
    "065-h_cups":                0.031,
    "065-i_cups":                0.035,
    "065-j_cups":                0.038,
    "070-a_colored_wood_blocks": 0.011,
    "070-b_colored_wood_blocks": 0.011,
    "071_nine_hole_peg_test":    1.435,
    # "072-a_toy_airplane":        0.100,
    # "072-b_toy_airplane":        0.100,
    # "072-c_toy_airplane":        0.100,
    # "072-d_toy_airplane":        0.100,
    # "072-e_toy_airplane":        0.100,
    # "073-a_lego_duplo":          0.025,
    # "073-b_lego_duplo":          0.025,
    # "073-c_lego_duplo":          0.025,
    # "073-d_lego_duplo":          0.025,
    # "073-e_lego_duplo":          0.025,
    # "073-f_lego_duplo":          0.025,
    # "073-g_lego_duplo":          0.025,
    "077_rubiks_cube":           0.094,
}

ALL_OBJECTS = list(MASS_TABLE.keys())

# ── SAM prompt 對照表(物體名 → 交給 Grounded-SAM 的 prompt 文字)──────────────
# 與名稱解耦:可逐項調整 prompt 而不動 MASS_TABLE/資產名稱。
# 初值等同舊規則(去純數字前綴、底線換空格);cups/colored_wood_blocks 的
# 編號前綴(065-b / 070-a)目前仍保留,可視需要改成 "cups" / "colored wood blocks"。
PROMPT_TABLE = {
    "002_master_chef_can": "master chef can",
    "003_cracker_box": "cracker box",
    "004_sugar_box": "sugar box",
    "005_tomato_soup_can": "tomato soup can",
    "006_mustard_bottle": "mustard bottle",
    "007_tuna_fish_can": "tuna fish can",
    "008_pudding_box": "pudding box",
    "009_gelatin_box": "gelatin box",
    "010_potted_meat_can": "potted meat can",
    "011_banana": "banana",
    "012_strawberry": "strawberry",
    "013_apple": "apple",
    "014_lemon": "lemon",
    "015_peach": "peach",
    "016_pear": "pear",
    "017_orange": "orange",
    "018_plum": "plum",
    "019_pitcher_base": "pitcher base",
    "021_bleach_cleanser": "bleach cleanser",
    "022_windex_bottle": "windex bottle",
    "024_bowl": "bowl",
    "025_mug": "mug",
    "026_sponge": "sponge",
    "028_skillet_lid": "skillet lid",
    "029_plate": "plate",
    "030_fork": "fork",
    "031_spoon": "spoon",
    "032_knife": "knife",
    "033_spatula": "spatula",
    "035_power_drill": "power drill",
    "036_wood_block": "wood block",
    "037_scissors": "scissors",
    "038_padlock": "padlock",
    "040_large_marker": "large marker",
    "042_adjustable_wrench": "adjustable wrench",
    "043_phillips_screwdriver": "phillips screwdriver",
    "044_flat_screwdriver": "flat screwdriver",
    "048_hammer": "hammer",
    "050_medium_clamp": "medium clamp",
    "051_large_clamp": "large clamp",
    "052_extra_large_clamp": "extra large clamp",
    "053_mini_soccer_ball": "mini soccer ball",
    "054_softball": "softball",
    "055_baseball": "baseball",
    "056_tennis_ball": "tennis ball",
    "057_racquetball": "racquetball",
    "058_golf_ball": "golf ball",
    "059_chain": "chain",
    "061_foam_brick": "foam brick",
    "062_dice": "dice",
    "065-a_cups": "orange cups",
    "065-b_cups": "blue cups",
    "065-c_cups": "green cups",
    "065-d_cups": "yellow cups",
    "065-e_cups": "red cups",
    "065-f_cups": "purple cups",
    "065-g_cups": "orange cups",
    "065-h_cups": "blue cups",
    "065-i_cups": "green cups",
    "065-j_cups": "yellow cups",
    "070-a_colored_wood_blocks": "colored wood blocks",
    "070-b_colored_wood_blocks": "blue wood blocks",
    "071_nine_hole_peg_test": "nine hole peg test",
    "077_rubiks_cube": "rubiks cube",
}

TARGET_OBJECTS = [
    # "005_tomato_soup_can",
    # "002_master_chef_can",
]

# ── 碰撞形狀預設值與形狀表 ────────────────────────────────
# SHAPE_TABLE 只保留碰撞形狀種類，實際尺寸由 ycb_geometries.json 提供。
# 沒有填入的物件會使用預設值 DEFAULT_SHAPE。
# ────────────────────────────────────────────────────────────
DEFAULT_SHAPE = "Box"

SHAPE_TABLE = {
    # ── 罐頭 / 瓶子（圓柱形）──
    "002_master_chef_can":   "Cylinder",
    "005_tomato_soup_can":   "Cylinder",
    "007_tuna_fish_can":     "Cylinder",
    "010_potted_meat_can":   "Box",
    "006_mustard_bottle":    "Box",
    "021_bleach_cleanser":   "Cylinder",
    "022_windex_bottle":     "Box",
    "019_pitcher_base":      "Box",

    # ── 盒子（Box）──
    "003_cracker_box":       "Box",
    "004_sugar_box":         "Box",
    "008_pudding_box":       "Box",
    "009_gelatin_box":       "Box",
    "036_wood_block":        "Box",
    "061_foam_brick":        "Box",
    "062_dice":              "Box",

    # ── 水果（Sphere 近似）──
    "012_strawberry":        "Sphere",
    "013_apple":             "Sphere",
    "014_lemon":             "Sphere",
    "015_peach":             "Sphere",
    "016_pear":              "Box",
    "017_orange":            "Sphere",
    "018_plum":              "Sphere",
    "011_banana":            "Box",

    # ── 球類 ──
    "053_mini_soccer_ball":  "Sphere",
    "054_softball":          "Sphere",
    "055_baseball":          "Sphere",
    "056_tennis_ball":       "Sphere",
    "057_racquetball":       "Sphere",
    "058_golf_ball":         "Sphere",

    # ── 餐具（細長 Box）──
    "024_bowl":              "Box",
    "025_mug":               "Box",
    "029_plate":             "Cylinder",
    "028_skillet_lid":       "Cylinder",
    "030_fork":              "Box",
    "031_spoon":             "Box",
    "032_knife":             "Box",
    "033_spatula":           "Box",
    "026_sponge":            "Box",

    # ── 工具 ──
    "035_power_drill":       "Box",
    "037_scissors":          "Box",
    "038_padlock":           "Box",
    "040_large_marker":      "Cylinder",
    "042_adjustable_wrench": "Box",
    "043_phillips_screwdriver": "Cylinder",
    "044_flat_screwdriver":  "Cylinder",
    "048_hammer":            "Box",
    "050_medium_clamp":      "Box",
    "051_large_clamp":       "Box",
    "052_extra_large_clamp": "Box",
    "059_chain":             "Box",

    # ── 玩具 / 積木 ──
    "070-a_colored_wood_blocks": "Box",
    "070-b_colored_wood_blocks": "Box",
    "071_nine_hole_peg_test":    "Box",
    "072-a_toy_airplane":        "Box",
    "072-b_toy_airplane":        "Box",
    "072-c_toy_airplane":        "Box",
    "072-d_toy_airplane":        "Box",
    "072-e_toy_airplane":        "Box",
    "073-a_lego_duplo":          "Box",
    "073-b_lego_duplo":          "Box",
    "073-c_lego_duplo":          "Box",
    "073-d_lego_duplo":          "Box",
    "073-e_lego_duplo":          "Box",
    "073-f_lego_duplo":          "Box",
    "073-g_lego_duplo":          "Box",
    "077_rubiks_cube":           "Box",

    # ── 彈珠 ──
    "063-a_marbles":         "Sphere",
    "063-b_marbles":         "Sphere",
    "065-a_cups":            "Cylinder",
    "065-b_cups":            "Cylinder",
    "065-c_cups":            "Cylinder",
    "065-d_cups":            "Cylinder",
    "065-e_cups":            "Cylinder",
    "065-f_cups":            "Cylinder",
    "065-g_cups":            "Cylinder",
    "065-h_cups":            "Cylinder",
    "065-i_cups":            "Cylinder",
    "065-j_cups":            "Cylinder",
}
# ────────────────────────────────────────────────────────────
