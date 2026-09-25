# 封閉問句直接問「誰在上」(對照:開放式 Open3DSG 原句)

- 建檔 2026-09-25;程式 `srp/stage4_probe/llava_pair_direction.py`;模型 `llava-hf/llama3-llava-next-8b-hf`,4-bit NF4。
- 影像與開放式版**完全相同**(pair_crops,兩物 bbox 聯集,top-3 視角)→ **單一變因 = prompt**。
- 判定:3 視角需 **≥2 票**一致;GT on 對的正解 = 上物,非 on 對的正解 = NEITHER。
- ⚠ 物體名用 GT(上界模擬);**位置偏誤**由 (場景,A,B) 雜湊決定選項順序來控制。

## 位置偏誤檢查

| 項目 | 值 |
|---|---|
| 模型選了「某個物體」的次數 | 161 |
| 其中選中**第一個選項**的比例 | **51.6%**(理想 ≈50%) |

## 主結果

| 母體 | 數量 | 指標 | 值 |
|---|---|---|---|
| GT 有 on | 29 | **方向完全正確** | **21 (72.4%)** |
| GT 有 on | 29 | 有說某物在上(不論方向) | 29 (100.0%) |
| GT 沒 on | 60 | **正確答 NEITHER** | **34 (56.7%)** |
| GT 沒 on | 60 | 誤報有堆疊(假陽性) | 25 (41.7%) |
| GT 沒 on | 60 | 票數不足/無法判定 | 1 |

**含方向的平衡準確率 = 64.5%**

### 與開放式版對照

| 版本 | 偵測平衡準確率 | 方向正確率(29 對) |
|---|---|---|
| 開放式(Open3DSG 原句) | 78.8%(3/3票) | **69.0%** |
| **封閉問句(本次)** | — | **72.4%** |

## 逐對(GT on 的 29 對)

| 場景 | A | B | GT上物 | 選項順序 | 三視角投票 | 判定 | 對? |
|---|---|---|---|---|---|---|---|
| stack3_scene0005 | foam_brick | gelatin_box | **foam_brick** | gelatin_box→foam_brick | foam_brick|foam_brick|foam_brick | foam_brick | ✅ |
| stack3_scene0006 | tomato_soup_can | tuna_fish_can | **tomato_soup_can** | tuna_fish_can→tomato_soup_can | tuna_fish_can|tuna_fish_can|tuna_fish_can | tuna_fish_can | ❌ |
| stack3_scene0011 | tomato_soup_can | tuna_fish_can | **tomato_soup_can** | tuna_fish_can→tomato_soup_can | tuna_fish_can|tuna_fish_can|tuna_fish_can | tuna_fish_can | ❌ |
| stack3_scene0013 | foam_brick | sponge | **foam_brick** | sponge→foam_brick | foam_brick|foam_brick|foam_brick | foam_brick | ✅ |
| stack3_scene0014 | master_chef_can | tomato_soup_can | **tomato_soup_can** | tomato_soup_can→master_chef_can | tomato_soup_can|tomato_soup_can|tomato_soup_can | tomato_soup_can | ✅ |
| stack3_scene0015 | foam_brick | gelatin_box | **foam_brick** | gelatin_box→foam_brick | foam_brick|foam_brick|foam_brick | foam_brick | ✅ |
| stack3_scene0016 | tomato_soup_can | tuna_fish_can | **tomato_soup_can** | tuna_fish_can→tomato_soup_can | tuna_fish_can|tuna_fish_can|tuna_fish_can | tuna_fish_can | ❌ |
| stack3_scene0017 | master_chef_can | tomato_soup_can | **tomato_soup_can** | tomato_soup_can→master_chef_can | tomato_soup_can|tomato_soup_can|tomato_soup_can | tomato_soup_can | ✅ |
| stack3_scene0019 | foam_brick | wood_block | **foam_brick** | foam_brick→wood_block | foam_brick|foam_brick|foam_brick | foam_brick | ✅ |
| stack4_scene0002 | tomato_soup_can | tuna_fish_can | **tomato_soup_can** | tuna_fish_can→tomato_soup_can | tuna_fish_can|tuna_fish_can|tuna_fish_can | tuna_fish_can | ❌ |
| stack4_scene0003 | master_chef_can | pudding_box | **master_chef_can** | master_chef_can→pudding_box | master_chef_can|master_chef_can|master_chef_can | master_chef_can | ✅ |
| stack4_scene0006 | gelatin_box | tomato_soup_can | **tomato_soup_can** | tomato_soup_can→gelatin_box | tomato_soup_can|tomato_soup_can|tomato_soup_can | tomato_soup_can | ✅ |
| stack4_scene0007 | sponge | sugar_box | **sugar_box** | sponge→sugar_box | sponge|sponge|sponge | sponge | ❌ |
| stack4_scene0009 | master_chef_can | pudding_box | **master_chef_can** | pudding_box→master_chef_can | master_chef_can|pudding_box|master_chef_can | master_chef_can | ✅ |
| stack4_scene0010 | master_chef_can | tuna_fish_can | **tuna_fish_can** | master_chef_can→tuna_fish_can | tuna_fish_can|tuna_fish_can|tuna_fish_can | tuna_fish_can | ✅ |
| stack4_scene0011 | master_chef_can | tomato_soup_can | **tomato_soup_can** | tomato_soup_can→master_chef_can | tomato_soup_can|tomato_soup_can|tomato_soup_can | tomato_soup_can | ✅ |
| stack4_scene0012 | gelatin_box | tomato_soup_can | **tomato_soup_can** | gelatin_box→tomato_soup_can | tomato_soup_can|tomato_soup_can|tomato_soup_can | tomato_soup_can | ✅ |
| stack4_scene0014 | sponge | sugar_box | **sugar_box** | sponge→sugar_box | sponge|sponge|sponge | sponge | ❌ |
| stack4_scene0017 | foam_brick | master_chef_can | **foam_brick** | master_chef_can→foam_brick | foam_brick|foam_brick|foam_brick | foam_brick | ✅ |
| stack4_scene0018 | tomato_soup_can | tuna_fish_can | **tomato_soup_can** | tuna_fish_can→tomato_soup_can | tuna_fish_can|tuna_fish_can|tuna_fish_can | tuna_fish_can | ❌ |
| stack5_scene0005 | pudding_box | tuna_fish_can | **tuna_fish_can** | tuna_fish_can→pudding_box | tuna_fish_can|tuna_fish_can|tuna_fish_can | tuna_fish_can | ✅ |
| stack5_scene0008 | tomato_soup_can | tuna_fish_can | **tomato_soup_can** | tomato_soup_can→tuna_fish_can | tuna_fish_can|tuna_fish_can|tuna_fish_can | tuna_fish_can | ❌ |
| stack5_scene0011 | foam_brick | sponge | **foam_brick** | foam_brick→sponge | foam_brick|foam_brick|foam_brick | foam_brick | ✅ |
| stack5_scene0013 | foam_brick | sponge | **foam_brick** | foam_brick→sponge | foam_brick|foam_brick|foam_brick | foam_brick | ✅ |
| stack5_scene0014 | cracker_box | sugar_box | **sugar_box** | cracker_box→sugar_box | sugar_box|NEITHER|sugar_box | sugar_box | ✅ |
| stack5_scene0016 | tomato_soup_can | wood_block | **tomato_soup_can** | wood_block→tomato_soup_can | tomato_soup_can|tomato_soup_can|tomato_soup_can | tomato_soup_can | ✅ |
| stack5_scene0018 | gelatin_box | tomato_soup_can | **tomato_soup_can** | gelatin_box→tomato_soup_can | tomato_soup_can|tomato_soup_can|tomato_soup_can | tomato_soup_can | ✅ |
| stack5_scene0019 | cracker_box | sugar_box | **sugar_box** | cracker_box→sugar_box | sugar_box|sugar_box|NEITHER | sugar_box | ✅ |
| stack5_scene0020 | pudding_box | tuna_fish_can | **tuna_fish_can** | pudding_box→tuna_fish_can | tuna_fish_can|tuna_fish_can|tuna_fish_can | tuna_fish_can | ✅ |