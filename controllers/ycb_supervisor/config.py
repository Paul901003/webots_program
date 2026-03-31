# ── 生成參數 ──────────────────────────────────────────────
NUM_OBJECTS  = 5
GRID_COLS    = 3
SPACING      = 0.15
SPAWN_HEIGHT = 0.02
X_OFFSET     = 0.3
Z_OFFSET     = 0.1
ASSET_BASE   = "../urdfs/ycb_assets"

TARGET_OBJECTS = [
    # "005_tomato_soup_can",
    # "002_master_chef_can",
]

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

# ── 碰撞形狀預設值與形狀表 ────────────────────────────────
# 格式：
#   "物件名稱": ("形狀", (尺寸...))
#
#   Box      → ("Box",      (x, y, z))          單位：公尺
#   Sphere   → ("Sphere",   (radius,))
#   Cylinder → ("Cylinder", (radius, height))
#
# 沒有填入的物件會使用預設值 DEFAULT_SHAPE。
# ────────────────────────────────────────────────────────────
DEFAULT_SHAPE = ("Box", (0.1, 0.1, 0.1))

SHAPE_TABLE = {
    # ── 罐頭 / 瓶子（圓柱形）──
    "002_master_chef_can":   ("Cylinder", (0.102/2, 0.139)),
    "005_tomato_soup_can":   ("Cylinder", (0.066/2, 0.101)),
    "007_tuna_fish_can":     ("Cylinder", (0.042, 0.034)),
    "010_potted_meat_can":   ("Box",      (0.058, 0.083, 0.054)),
    "006_mustard_bottle":    ("Box",      (0.058, 0.095, 0.191)),
    "021_bleach_cleanser":   ("Cylinder", (0.032, 0.250)),
    "022_windex_bottle":     ("Box",      (0.050, 0.097, 0.280)),
    "019_pitcher_base":      ("Box",      (0.097, 0.133, 0.177)),

    # ── 盒子（Box）──
    "003_cracker_box":       ("Box",      (0.060, 0.158, 0.209)),
    "004_sugar_box":         ("Box",      (0.045, 0.089, 0.175)),
    "008_pudding_box":       ("Box",      (0.086, 0.112, 0.042)),
    "009_gelatin_box":       ("Box",      (0.057, 0.085, 0.031)),
    "036_wood_block":        ("Box",      (0.086, 0.054, 0.054)),
    "061_foam_brick":        ("Box",      (0.076, 0.051, 0.051)),
    "062_dice":              ("Box",      (0.018, 0.018, 0.018)),

    # ── 水果（Sphere 近似）──
    "012_strawberry":        ("Sphere",   (0.022,)),
    "013_apple":             ("Sphere",   (0.037,)),
    "014_lemon":             ("Sphere",   (0.030,)),
    "015_peach":             ("Sphere",   (0.037,)),
    "016_pear":              ("Box",      (0.055, 0.055, 0.087)),
    "017_orange":            ("Sphere",   (0.035,)),
    "018_plum":              ("Sphere",   (0.025,)),
    "011_banana":            ("Box",      (0.030, 0.190, 0.040)),

    # ── 球類 ──
    "053_mini_soccer_ball":  ("Sphere",   (0.065,)),
    "054_softball":          ("Sphere",   (0.048,)),
    "055_baseball":          ("Sphere",   (0.037,)),
    "056_tennis_ball":       ("Sphere",   (0.033,)),
    "057_racquetball":       ("Sphere",   (0.029,)),
    "058_golf_ball":         ("Sphere",   (0.021,)),

    # ── 餐具（細長 Box）──
    "024_bowl":              ("Box",      (0.161, 0.161, 0.053)),
    "025_mug":               ("Box",      (0.117, 0.093, 0.081)),
    "029_plate":             ("Cylinder", (0.126, 0.020)),
    "028_skillet_lid":       ("Cylinder", (0.133, 0.024)),
    "030_fork":              ("Box",      (0.018, 0.186, 0.008)),
    "031_spoon":             ("Box",      (0.017, 0.178, 0.035)),
    "032_knife":             ("Box",      (0.015, 0.238, 0.009)),
    "033_spatula":           ("Box",      (0.033, 0.304, 0.010)),
    "026_sponge":            ("Box",      (0.111, 0.069, 0.042)),

    # ── 工具 ──
    "035_power_drill":       ("Box",      (0.093, 0.184, 0.214)),
    "037_scissors":          ("Box",      (0.025, 0.200, 0.065)),
    "038_padlock":           ("Box",      (0.048, 0.073, 0.028)),
    "040_large_marker":      ("Cylinder", (0.012, 0.148)),
    "042_adjustable_wrench": ("Box",      (0.031, 0.230, 0.039)),
    "043_phillips_screwdriver": ("Cylinder", (0.010, 0.220)),
    "044_flat_screwdriver":  ("Cylinder", (0.010, 0.228)),
    "048_hammer":            ("Box",      (0.040, 0.285, 0.090)),
    "050_medium_clamp":      ("Box",      (0.025, 0.152, 0.073)),
    "051_large_clamp":       ("Box",      (0.031, 0.198, 0.090)),
    "052_extra_large_clamp": ("Box",      (0.038, 0.248, 0.112)),
    "059_chain":             ("Box",      (0.020, 0.300, 0.020)),

    # ── 玩具 / 積木 ──
    "070-a_colored_wood_blocks": ("Box", (0.055, 0.055, 0.055)),
    "070-b_colored_wood_blocks": ("Box", (0.055, 0.055, 0.055)),
    "071_nine_hole_peg_test":    ("Box", (0.125, 0.125, 0.020)),
    "072-a_toy_airplane":        ("Box", (0.180, 0.155, 0.055)),
    "072-b_toy_airplane":        ("Box", (0.180, 0.155, 0.055)),
    "072-c_toy_airplane":        ("Box", (0.180, 0.155, 0.055)),
    "072-d_toy_airplane":        ("Box", (0.180, 0.155, 0.055)),
    "072-e_toy_airplane":        ("Box", (0.180, 0.155, 0.055)),
    "073-a_lego_duplo":          ("Box", (0.031, 0.031, 0.038)),
    "073-b_lego_duplo":          ("Box", (0.031, 0.031, 0.038)),
    "073-c_lego_duplo":          ("Box", (0.031, 0.063, 0.038)),
    "073-d_lego_duplo":          ("Box", (0.031, 0.063, 0.038)),
    "073-e_lego_duplo":          ("Box", (0.063, 0.063, 0.038)),
    "073-f_lego_duplo":          ("Box", (0.063, 0.063, 0.038)),
    "073-g_lego_duplo":          ("Box", (0.031, 0.095, 0.038)),
    "077_rubiks_cube":           ("Box", (0.057, 0.057, 0.057)),

    # ── 彈珠 ──
    "063-a_marbles":         ("Sphere",   (0.013,)),
    "063-b_marbles":         ("Sphere",   (0.013,)),
    "065-a_cups":            ("Cylinder", (0.020, 0.050)),
    "065-b_cups":            ("Cylinder", (0.020, 0.050)),
    "065-c_cups":            ("Cylinder", (0.020, 0.050)),
    "065-d_cups":            ("Cylinder", (0.020, 0.050)),
    "065-e_cups":            ("Cylinder", (0.020, 0.050)),
    "065-f_cups":            ("Cylinder", (0.020, 0.050)),
    "065-g_cups":            ("Cylinder", (0.020, 0.050)),
    "065-h_cups":            ("Cylinder", (0.020, 0.050)),
    "065-i_cups":            ("Cylinder", (0.020, 0.050)),
    "065-j_cups":            ("Cylinder", (0.020, 0.050)),
}
# ────────────────────────────────────────────────────────────
