# 成對關係提問(Open3DSG 做法):VLM 能不能說出 on?

- 建檔 2026-09-25;程式 `srp/stage4_probe/llava_pair_relation.py`;模型 `llava-hf/llama3-llava-next-8b-hf`,4-bit NF4。
- **忠於 Open3DSG(CVPR2024)補充材料 Sec.A**:影像=兩物 bbox 聯集裁切(`box_ij = box_ik ∪ box_jk`);
  prompt=原句 `"Describe the relationship between [object1] and [object2]?"`;物體名當 context。
- ⚠ 物體名用 **GT = 上界模擬**(論文由 CLIP 先推論);視角選擇用 GT modal 面積(論文用深度判遮擋,本專案不可用)。
- 母體 **89 對**:on **29**、非 on **60**(對照組,來自同樣 60 場 stack;排 GEX)。
- 判讀:關鍵詞(on top of / stacked on / resting on / underneath / supporting …),需人工覆核。

## 混淆矩陣(只判「有沒有 on」)

| | 模型說有 on | 模型沒說 on | 合計 |
|---|---|---|---|
| **GT 有 on** | **TP 24** | FN 5 | 29 |
| **GT 沒 on** | **FP 27** | TN 33 | 60 |

| 指標 | 值 |
|---|---|
| 召回率 | 82.8% |
| **假陽性率** | **45.0%** |
| 精確率 | 47.1% |
| **平衡準確率** | **68.9%** |

(對照:先前整張場景圖版 = 52.8%,`RESULT_llava_stack_batch.md`)

## 逐對(GT 有 on 的 29 對全列)

| 場景 | A | B | 上物 | 判讀 | 回答(截斷) |
|---|---|---|---|---|---|
| stack3_scene0005 | foam_brick | gelatin_box | foam_brick | ✅on | The image shows a foam brick placed on top of a gelatin box. The foam brick appears to be a lightwei |
| stack3_scene0006 | tomato_soup_can | tuna_fish_can | tomato_soup_can | ✅on | The image shows two cans of food stacked on top of each other. The can on top appears to be a can of |
| stack3_scene0011 | tomato_soup_can | tuna_fish_can | tomato_soup_can | ✅on | The image shows two cans of food stacked on top of each other. The top can is a can of "Campbells To |
| stack3_scene0013 | foam_brick | sponge | foam_brick | ✅on | The image shows a foam brick placed on top of a sponge. The foam brick appears to be a lightweight,  |
| stack3_scene0014 | master_chef_can | tomato_soup_can | tomato_soup_can | ✅on | The image shows two cans of food stacked on top of each other. The can on top appears to be a can of |
| stack3_scene0015 | foam_brick | gelatin_box | foam_brick | ✅on | The image shows a foam brick placed on top of a gelatin box. The foam brick appears to be a lightwei |
| stack3_scene0016 | tomato_soup_can | tuna_fish_can | tomato_soup_can | ❌無 | The image shows two cans of food. The can on the left appears to be a can of tomato soup, as indicat |
| stack3_scene0017 | master_chef_can | tomato_soup_can | tomato_soup_can | ✅on | The image shows two cans of food stacked on top of each other. The can on top appears to be a can of |
| stack3_scene0019 | foam_brick | wood_block | foam_brick | ✅on | The image shows a foam brick placed on top of a wooden block. The foam brick appears to be a lightwe |
| stack4_scene0002 | tomato_soup_can | tuna_fish_can | tomato_soup_can | ❌無 | The image shows two cans of food. The top can is labeled "Campbell's Tomato Soup," indicating that i |
| stack4_scene0003 | master_chef_can | pudding_box | master_chef_can | ❌無 | The image shows a can of "Master Chef" coffee and a box of "Master Chef" pudding. The can and the bo |
| stack4_scene0006 | gelatin_box | tomato_soup_can | tomato_soup_can | ✅on | The image shows a can of tomato soup placed on top of a box of gelatin. The can appears to be a stan |
| stack4_scene0007 | sponge | sugar_box | sugar_box | ✅on | The image shows a box of Domino Sugar placed on top of a sponge. The sponge is likely being used to  |
| stack4_scene0009 | master_chef_can | pudding_box | master_chef_can | ✅on | The image shows a can of "Master Chef" coffee and a box of "Master Chef" pudding. The can and the bo |
| stack4_scene0010 | master_chef_can | tuna_fish_can | tuna_fish_can | ❌無 | The image shows a can of coffee and a can of tuna fish. The coffee can is on the left, and the tuna  |
| stack4_scene0011 | master_chef_can | tomato_soup_can | tomato_soup_can | ✅on | The image shows two cans of food stacked on top of each other. The top can appears to be a can of co |
| stack4_scene0012 | gelatin_box | tomato_soup_can | tomato_soup_can | ✅on | The image shows a gelatin box and a can of tomato soup. The gelatin box is placed on top of the toma |
| stack4_scene0014 | sponge | sugar_box | sugar_box | ✅on | The image shows a box of Domino Sugar with a sponge placed underneath it. The sponge is likely there |
| stack4_scene0017 | foam_brick | master_chef_can | foam_brick | ✅on | The image shows a foam brick placed on top of a can of coffee. The foam brick appears to be a toy or |
| stack4_scene0018 | tomato_soup_can | tuna_fish_can | tomato_soup_can | ❌無 | The image shows a can of "Campbell's Tomato Soup" and a can of "StarKist Tuna Fish." These two produ |
| stack5_scene0005 | pudding_box | tuna_fish_can | tuna_fish_can | ✅on | The image shows a can of "DarkKist" tuna fish placed on top of a box of "Pudding" which appears to b |
| stack5_scene0008 | tomato_soup_can | tuna_fish_can | tomato_soup_can | ✅on | The image shows two cans of food stacked on top of each other. The can on top appears to be a can of |
| stack5_scene0011 | foam_brick | sponge | foam_brick | ✅on | The image you've provided appears to show a foam brick with three holes in it, which is likely a typ |
| stack5_scene0013 | foam_brick | sponge | foam_brick | ✅on | The image you've provided appears to show a foam brick with three holes in it, which is likely a par |
| stack5_scene0014 | cracker_box | sugar_box | sugar_box | ✅on | The image shows a stack of three boxes. The topmost box is labeled "DOMINO SUGAR," indicating that i |
| stack5_scene0016 | tomato_soup_can | wood_block | tomato_soup_can | ✅on | The image shows a can of tomato soup placed on top of a wooden block. The can is open, with the lid  |
| stack5_scene0018 | gelatin_box | tomato_soup_can | tomato_soup_can | ✅on | The image shows a can of "Campbells Tomato Soup" placed on top of a box of gelatin. The gelatin box  |
| stack5_scene0019 | cracker_box | sugar_box | sugar_box | ✅on | The image shows two boxes stacked on top of each other. The top box is a box of Domino Sugar, which  |
| stack5_scene0020 | pudding_box | tuna_fish_can | tuna_fish_can | ✅on | The image shows a can of StarKist brand light tuna fish placed on top of a box of pudding. The relat |

## 對照組中被誤判為 on 的(前 15)

| 場景 | A | B | 回答(截斷) |
|---|---|---|---|
| stack5_scene0019 | orange | sugar_box | The image shows a box of Cheez-It crackers and an orange. The box of Cheez-It crackers is placed on  |
| stack5_scene0004 | apple | large_clamp | The image shows a large clamp with a red apple placed on it. The clamp appears to be holding the app |
| stack3_scene0019 | mug | wood_block | The image shows a red mug and a wooden block with holes in it. The mug appears to be placed on the g |
| stack4_scene0010 | lemon | master_chef_can | The image shows a lemon and a coffee can. The lemon is positioned to the left of the coffee can, and |
| stack5_scene0013 | sponge | sugar_box | The image shows a sponge and a sugar box. The sponge is placed on the right side of the image, and i |
| stack5_scene0019 | bowl | cracker_box | The image shows a red bowl with a box of Domino Sugar on top of it. The box of Domino Sugar is leani |
| stack4_scene0013 | cracker_box | cups | The image shows a box of Cheez-It crackers with a blue lid on top, and two green cups. The box is po |
| stack5_scene0018 | cups | tomato_soup_can | The image shows a can of tomato soup placed on top of a red box, which appears to be a cardboard box |
| stack4_scene0006 | cups | gelatin_box | The image shows a 3D rendering of a cup and a gelatin box. The cup appears to be a simple, cylindric |
| stack4_scene0019 | medium_clamp | wood_block | The image shows a medium clamp, which appears to be a type of clamp used in woodworking or other cra |
| stack4_scene0004 | rubiks_cube | wood_block | In the image, there is a Rubik's Cube placed on top of a wooden block. The wooden block appears to b |
| stack5_scene0018 | golf_ball | tomato_soup_can | The image shows a can of "Campbells Tomato Soup" placed on top of a box of the same product. In the  |
| stack4_scene0007 | sponge | tomato_soup_can | The image shows a box of Domino Sugar and a can of tomato soup. The box of Domino Sugar is standing  |
| stack5_scene0019 | bowl | orange | The image shows a red bowl and an orange. The bowl is positioned above the orange, suggesting that t |
| stack5_scene0017 | master_chef_can | padlock | In the image, there is a can of "Master Chef" coffee and a padlock. The can is placed on the left si |