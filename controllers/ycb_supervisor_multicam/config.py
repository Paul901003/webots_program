"""多相機同步拍攝設定檔。

決定「要用哪份視角檔 → 建立幾台、什麼位姿的固定相機」。改這裡再重跑
gen_multicam_world.py 即可,不必動 .wbt 或控制器。
"""

# 視角來源:data/viewpoints/ 下的檔名。相機「數量」與「位姿」皆由此檔決定。
#   - validated_viewpoints_multi_latest.json  全部路徑可行視角(目前 49)
#   - selected_viewpoints_multi_latest.json   既有資料用的 12 視角
# 只取 ok=true 且 planning.success(若有)的視角;需含 ray.ray_origin_m + ray.ray_axis_world。
VIEWPOINTS_FILE = "validated_viewpoints_multi_latest.json"

# 拍攝輸出根(data/ 下);與既有 captures 分開避免覆蓋。
CAPTURE_ROOT_NAME = "captures_multicam"

# 每場景:spawn 後靜置秒數 / 觸發後等待相機存檔的秒數。
SCENE_SETTLE_SEC = 1.5
CAPTURE_WAIT_SEC = 2.5

# 分批拍攝:0=全部同時觸發;>0=每批 N 台,分批拍攝+存圖(降低同時開 N 台 HD+depth 的負載)。
# 場景內物體不動,分批不影響結果。可用 env MULTICAM_BATCH 覆寫。
CAPTURE_BATCH_SIZE = 0
