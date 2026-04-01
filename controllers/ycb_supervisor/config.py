# ── 生成參數 ──────────────────────────────────────────────
# TARGET_OBJECTS 留空時，supervisor 會進入隨機模式，從 ALL_OBJECTS 中抽取 NUM_OBJECTS 個。
# 若只想放特定物件，直接在 TARGET_OBJECTS 中填入名稱即可。
# 若想一次放全部物件，請使用：TARGET_OBJECTS = ALL_OBJECTS.copy()
NUM_OBJECTS  = 5
GRID_COLS    = 3
SPACING      = 0.15
SPAWN_HEIGHT = 0.02
X_OFFSET     = 0.3
Z_OFFSET     = 0.1
ASSET_BASE   = "../urdfs/ycb_assets"

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
