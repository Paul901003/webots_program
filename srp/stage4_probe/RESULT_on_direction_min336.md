# 關係評分(llava_pair_relation_min336.csv):偵測 + 方向

## 偵測(有沒有 on;逐視角關鍵詞 + 投票)

| 需幾票 | 召回率 | 假陽性率 | 精確率 | 平衡準確率 |
|---|---|---|---|---|
| ≥1/3 | 96.6% | 80.0% | 36.8% | **58.3%** |
| ≥2/3 | 96.6% | 40.0% | 53.8% | **78.3%** |
| ≥3/3 | 79.3% | 20.0% | 65.7% | **79.7%** |

# on 關係的【方向】驗證:誰在上?

- 建檔 2026-09-25;程式 `srp/stage4_probe/parse_on_direction.py`;**純文字分析,不跑模型**。
- 資料:`llava_pair_relation.csv`(89 對 × 3 視角完整回答,Open3DSG 成對裁切 + 原句 prompt)。
- ⚠ 先前 `RESULT_llava_pair_relation.md` 只判「有沒有 on 字眼」,**未驗證方向**;本檔補上。
- 解析不出者標【無法判定】,**不猜**。

## 對層級(29 個 GT on 對;3 視角多數決)

| 項目 | 數量 | 佔 29 對 |
|---|---|---|
| 能解析出方向 | 25 | 86.2% |
| **方向正確** | **15** | **51.7%** |
| 方向錯誤 | 10 | 34.5% |
| 無法判定 | 4 | 13.8% |

- 在**能解析**的 25 對中,方向正確率 = **60.0%**
  (亂猜的基準是 50%)

## 視角層級(每個視角各算一次)

| 項目 | 數量 |
|---|---|
| 可解析的視角數 | 64 |
| **方向正確** | **39**(60.9%) |
| 方向錯誤 | 25(39.1%) |

## 逐對明細(29 對全列)

| 場景 | A | B | GT上物 | 視角1 | 視角2 | 視角3 | 多數決 | 對? |
|---|---|---|---|---|---|---|---|---|
| stack3_scene0005 | foam_brick | gelatin_box | **foam_brick** | foam_brick | foam_brick | foam_brick | foam_brick | ✅ |
| stack3_scene0006 | tomato_soup_can | tuna_fish_can | **tomato_soup_can** | — | — | — | — | — |
| stack3_scene0011 | tomato_soup_can | tuna_fish_can | **tomato_soup_can** | — | — | tomato_soup_can | tomato_soup_can | ✅ |
| stack3_scene0013 | foam_brick | sponge | **foam_brick** | foam_brick | sponge | foam_brick | foam_brick | ✅ |
| stack3_scene0014 | master_chef_can | tomato_soup_can | **tomato_soup_can** | master_chef_can | — | master_chef_can | master_chef_can | ❌ |
| stack3_scene0015 | foam_brick | gelatin_box | **foam_brick** | foam_brick | foam_brick | foam_brick | foam_brick | ✅ |
| stack3_scene0016 | tomato_soup_can | tuna_fish_can | **tomato_soup_can** | — | tuna_fish_can | tuna_fish_can | tuna_fish_can | ❌ |
| stack3_scene0017 | master_chef_can | tomato_soup_can | **tomato_soup_can** | master_chef_can | master_chef_can | tomato_soup_can | master_chef_can | ❌ |
| stack3_scene0019 | foam_brick | wood_block | **foam_brick** | wood_block | foam_brick | foam_brick | foam_brick | ✅ |
| stack4_scene0002 | tomato_soup_can | tuna_fish_can | **tomato_soup_can** | — | — | — | — | — |
| stack4_scene0003 | master_chef_can | pudding_box | **master_chef_can** | master_chef_can | master_chef_can | master_chef_can | master_chef_can | ✅ |
| stack4_scene0006 | gelatin_box | tomato_soup_can | **tomato_soup_can** | tomato_soup_can | tomato_soup_can | gelatin_box | tomato_soup_can | ✅ |
| stack4_scene0007 | sponge | sugar_box | **sugar_box** | sugar_box | sponge | sponge | sponge | ❌ |
| stack4_scene0009 | master_chef_can | pudding_box | **master_chef_can** | master_chef_can | — | master_chef_can | master_chef_can | ✅ |
| stack4_scene0010 | master_chef_can | tuna_fish_can | **tuna_fish_can** | — | master_chef_can | — | master_chef_can | ❌ |
| stack4_scene0011 | master_chef_can | tomato_soup_can | **tomato_soup_can** | tomato_soup_can | master_chef_can | — | tomato_soup_can | ✅ |
| stack4_scene0012 | gelatin_box | tomato_soup_can | **tomato_soup_can** | gelatin_box | gelatin_box | gelatin_box | gelatin_box | ❌ |
| stack4_scene0014 | sponge | sugar_box | **sugar_box** | sugar_box | sugar_box | sponge | sugar_box | ✅ |
| stack4_scene0017 | foam_brick | master_chef_can | **foam_brick** | foam_brick | foam_brick | foam_brick | foam_brick | ✅ |
| stack4_scene0018 | tomato_soup_can | tuna_fish_can | **tomato_soup_can** | — | — | tuna_fish_can | tuna_fish_can | ❌ |
| stack5_scene0005 | pudding_box | tuna_fish_can | **tuna_fish_can** | pudding_box | tuna_fish_can | pudding_box | pudding_box | ❌ |
| stack5_scene0008 | tomato_soup_can | tuna_fish_can | **tomato_soup_can** | tuna_fish_can | tomato_soup_can | tuna_fish_can | tuna_fish_can | ❌ |
| stack5_scene0011 | foam_brick | sponge | **foam_brick** | — | — | — | — | — |
| stack5_scene0013 | foam_brick | sponge | **foam_brick** | foam_brick | foam_brick | foam_brick | foam_brick | ✅ |
| stack5_scene0014 | cracker_box | sugar_box | **sugar_box** | sugar_box | sugar_box | — | sugar_box | ✅ |
| stack5_scene0016 | tomato_soup_can | wood_block | **tomato_soup_can** | tomato_soup_can | tomato_soup_can | tomato_soup_can | tomato_soup_can | ✅ |
| stack5_scene0018 | gelatin_box | tomato_soup_can | **tomato_soup_can** | gelatin_box | tomato_soup_can | gelatin_box | gelatin_box | ❌ |
| stack5_scene0019 | cracker_box | sugar_box | **sugar_box** | — | — | — | — | — |
| stack5_scene0020 | pudding_box | tuna_fish_can | **tuna_fish_can** | tuna_fish_can | tuna_fish_can | pudding_box | tuna_fish_can | ✅ |