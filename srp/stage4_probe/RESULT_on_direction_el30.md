# 關係評分(llava_pair_relation_el30.csv):偵測 + 方向

## 偵測(有沒有 on;逐視角關鍵詞 + 投票)

| 需幾票 | 召回率 | 假陽性率 | 精確率 | 平衡準確率 |
|---|---|---|---|---|
| ≥1票 | 100.0% | 76.7% | 38.7% | **61.7%** |
| ≥2票 | 100.0% | 46.7% | 50.9% | **76.7%** |
| ≥3票 | 86.2% | 28.3% | 59.5% | **78.9%** |
| ≥4票 | 62.1% | 18.3% | 62.1% | **71.9%** |

# on 關係的【方向】驗證:誰在上?

- 建檔 2026-09-25;程式 `srp/stage4_probe/parse_on_direction.py`;**純文字分析,不跑模型**。
- 資料:`llava_pair_relation.csv`(89 對 × 3 視角完整回答,Open3DSG 成對裁切 + 原句 prompt)。
- ⚠ 先前 `RESULT_llava_pair_relation.md` 只判「有沒有 on 字眼」,**未驗證方向**;本檔補上。
- 解析不出者標【無法判定】,**不猜**。

## 對層級(29 個 GT on 對;3 視角多數決)

| 項目 | 數量 | 佔 29 對 |
|---|---|---|
| 能解析出方向 | 27 | 93.1% |
| **方向正確** | **20** | **69.0%** |
| 方向錯誤 | 7 | 24.1% |
| 無法判定 | 2 | 6.9% |

- 在**能解析**的 27 對中,方向正確率 = **74.1%**
  (亂猜的基準是 50%)

## 視角層級(每個視角各算一次)

| 項目 | 數量 |
|---|---|
| 可解析的視角數 | 66 |
| **方向正確** | **55**(83.3%) |
| 方向錯誤 | 11(16.7%) |

## 逐對明細(29 對全列)

| 場景 | A | B | GT上物 | 逐視角判讀 | 對/錯票 | 多數決 | 對? |
|---|---|---|---|---|---|---|---|
| stack3_scene0005 | foam_brick | gelatin_box | **foam_brick** | foam_brick|foam_brick|foam_brick|foam_brick | 4/0 | foam_brick | ✅ |
| stack3_scene0006 | tomato_soup_can | tuna_fish_can | **tomato_soup_can** | None|None|None|None | 0/0 | — | — |
| stack3_scene0011 | tomato_soup_can | tuna_fish_can | **tomato_soup_can** | None|None|None|None | 0/0 | — | — |
| stack3_scene0013 | foam_brick | sponge | **foam_brick** | foam_brick|foam_brick|sponge|foam_brick | 3/1 | foam_brick | ✅ |
| stack3_scene0014 | master_chef_can | tomato_soup_can | **tomato_soup_can** | master_chef_can|None|None|None | 0/1 | master_chef_can | ❌ |
| stack3_scene0015 | foam_brick | gelatin_box | **foam_brick** | foam_brick|foam_brick|foam_brick|foam_brick | 4/0 | foam_brick | ✅ |
| stack3_scene0016 | tomato_soup_can | tuna_fish_can | **tomato_soup_can** | None|None|None|tuna_fish_can | 0/1 | tuna_fish_can | ❌ |
| stack3_scene0017 | master_chef_can | tomato_soup_can | **tomato_soup_can** | None|None|None|master_chef_can | 0/1 | master_chef_can | ❌ |
| stack3_scene0019 | foam_brick | wood_block | **foam_brick** | foam_brick|foam_brick|foam_brick|wood_block | 3/1 | foam_brick | ✅ |
| stack4_scene0002 | tomato_soup_can | tuna_fish_can | **tomato_soup_can** | None|None|tuna_fish_can|None | 0/1 | tuna_fish_can | ❌ |
| stack4_scene0003 | master_chef_can | pudding_box | **master_chef_can** | None|master_chef_can|None|None | 1/0 | master_chef_can | ✅ |
| stack4_scene0006 | gelatin_box | tomato_soup_can | **tomato_soup_can** | tomato_soup_can|tomato_soup_can|tomato_soup_can|gelatin_box | 3/1 | tomato_soup_can | ✅ |
| stack4_scene0007 | sponge | sugar_box | **sugar_box** | None|None|sugar_box|sugar_box | 2/0 | sugar_box | ✅ |
| stack4_scene0009 | master_chef_can | pudding_box | **master_chef_can** | master_chef_can|master_chef_can|None | 2/0 | master_chef_can | ✅ |
| stack4_scene0010 | master_chef_can | tuna_fish_can | **tuna_fish_can** | tuna_fish_can|tuna_fish_can|None|None | 2/0 | tuna_fish_can | ✅ |
| stack4_scene0011 | master_chef_can | tomato_soup_can | **tomato_soup_can** | None|None|master_chef_can|None | 0/1 | master_chef_can | ❌ |
| stack4_scene0012 | gelatin_box | tomato_soup_can | **tomato_soup_can** | gelatin_box|tomato_soup_can|tomato_soup_can|tomato_soup_can | 3/1 | tomato_soup_can | ✅ |
| stack4_scene0014 | sponge | sugar_box | **sugar_box** | None|None|sugar_box|None | 1/0 | sugar_box | ✅ |
| stack4_scene0017 | foam_brick | master_chef_can | **foam_brick** | foam_brick|foam_brick|foam_brick|foam_brick | 4/0 | foam_brick | ✅ |
| stack4_scene0018 | tomato_soup_can | tuna_fish_can | **tomato_soup_can** | None|tomato_soup_can|None|None | 1/0 | tomato_soup_can | ✅ |
| stack5_scene0005 | pudding_box | tuna_fish_can | **tuna_fish_can** | tuna_fish_can|tuna_fish_can|tuna_fish_can|tuna_fish_can | 4/0 | tuna_fish_can | ✅ |
| stack5_scene0008 | tomato_soup_can | tuna_fish_can | **tomato_soup_can** | None|tomato_soup_can|None|None | 1/0 | tomato_soup_can | ✅ |
| stack5_scene0011 | foam_brick | sponge | **foam_brick** | foam_brick|foam_brick|None | 2/0 | foam_brick | ✅ |
| stack5_scene0013 | foam_brick | sponge | **foam_brick** | foam_brick|foam_brick|foam_brick|None | 3/0 | foam_brick | ✅ |
| stack5_scene0014 | cracker_box | sugar_box | **sugar_box** | None|None|cracker_box|None | 0/1 | cracker_box | ❌ |
| stack5_scene0016 | tomato_soup_can | wood_block | **tomato_soup_can** | tomato_soup_can|tomato_soup_can|tomato_soup_can|tomato_soup_can | 4/0 | tomato_soup_can | ✅ |
| stack5_scene0018 | gelatin_box | tomato_soup_can | **tomato_soup_can** | tomato_soup_can|tomato_soup_can|tomato_soup_can|tomato_soup_can | 4/0 | tomato_soup_can | ✅ |
| stack5_scene0019 | cracker_box | sugar_box | **sugar_box** | None|cracker_box|None|None | 0/1 | cracker_box | ❌ |
| stack5_scene0020 | pudding_box | tuna_fish_can | **tuna_fish_can** | tuna_fish_can|tuna_fish_can|tuna_fish_can|tuna_fish_can | 4/0 | tuna_fish_can | ✅ |