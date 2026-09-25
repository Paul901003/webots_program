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
| **GT 有 on** | **TP 29** | FN 0 | 29 |
| **GT 沒 on** | **FP 28** | TN 32 | 60 |

| 指標 | 值 |
|---|---|
| 召回率 | 100.0% |
| **假陽性率** | **46.7%** |
| 精確率 | 50.9% |
| **平衡準確率** | **76.7%** |

(對照:整張場景圖版 52.8%;單視角成對版 68.9%)

## 投票門檻掃描(同一批推論,只改判定門檻)

| 需幾票 | 召回率 | 假陽性率 | 精確率 | 平衡準確率 |
|---|---|---|---|---|
| ≥1/3 | 100.0% | 76.7% | 38.7% | **61.7%** |
| ≥2/3 | 100.0% | 46.7% | 50.9% | **76.7%** |
| ≥3/3 | 86.2% | 28.3% | 59.5% | **78.9%** |

## 逐對(GT 有 on 的 29 對全列)

| 場景 | A | B | 上物 | 票數 | 判讀 | 回答(截斷) |
|---|---|---|---|---|---|---|
| stack3_scene0005 | foam_brick | gelatin_box | foam_brick | 4/4 | ✅on | [view_el30_az135] The image shows a foam brick placed on top of a gelatin box. The foam brick appear |
| stack3_scene0006 | tomato_soup_can | tuna_fish_can | tomato_soup_can | 3/4 | ✅on | [view_el30_az240] The image shows two cans of food stacked on top of each other. The can on top appe |
| stack3_scene0011 | tomato_soup_can | tuna_fish_can | tomato_soup_can | 4/4 | ✅on | [view_el30_az135] The image shows two cans of food stacked on top of each other. The top can is a ca |
| stack3_scene0013 | foam_brick | sponge | foam_brick | 4/4 | ✅on | [view_el30_az195] The image shows a foam brick placed on top of a sponge. The foam brick appears to  |
| stack3_scene0014 | master_chef_can | tomato_soup_can | tomato_soup_can | 4/4 | ✅on | [view_el30_az210] The image shows two cans of food stacked on top of each other. The can on top appe |
| stack3_scene0015 | foam_brick | gelatin_box | foam_brick | 4/4 | ✅on | [view_el30_az240] The image shows a foam brick placed on top of a gelatin box. The foam brick appear |
| stack3_scene0016 | tomato_soup_can | tuna_fish_can | tomato_soup_can | 3/4 | ✅on | [view_el30_az240] The image shows two cans of food. The can on the left appears to be a can of tomat |
| stack3_scene0017 | master_chef_can | tomato_soup_can | tomato_soup_can | 4/4 | ✅on | [view_el30_az210] The image shows two cans of food stacked on top of each other. The can on top appe |
| stack3_scene0019 | foam_brick | wood_block | foam_brick | 4/4 | ✅on | [view_el30_az210] The image shows a foam brick on top of a wooden block. The foam brick appears to b |
| stack4_scene0002 | tomato_soup_can | tuna_fish_can | tomato_soup_can | 3/4 | ✅on | [view_el30_az135] The image shows two cans of food. The top can is labeled "Campbell's Tomato Soup," |
| stack4_scene0003 | master_chef_can | pudding_box | master_chef_can | 3/4 | ✅on | [view_el30_az240] The image shows a can of "Master Chef" coffee and a box of "Master Chef" pudding.  |
| stack4_scene0006 | gelatin_box | tomato_soup_can | tomato_soup_can | 4/4 | ✅on | [view_el30_az210] The image shows a can of tomato soup placed on top of a box of gelatin. The can ap |
| stack4_scene0007 | sponge | sugar_box | sugar_box | 4/4 | ✅on | [view_el30_az210] The image shows a box of Domino Sugar with a sponge placed on top of it. The spong |
| stack4_scene0009 | master_chef_can | pudding_box | master_chef_can | 2/3 | ✅on | [view_el30_az135] The image shows a can of "Master Chef" coffee and a box of "Master Chef" pudding.  |
| stack4_scene0010 | master_chef_can | tuna_fish_can | tuna_fish_can | 3/4 | ✅on | [view_el30_az195] The image shows a can of "Master Chef" coffee and a can of "Master Chef" tuna fish |
| stack4_scene0011 | master_chef_can | tomato_soup_can | tomato_soup_can | 4/4 | ✅on | [view_el30_az135] The image shows two cans of food stacked on top of each other. The top can is a re |
| stack4_scene0012 | gelatin_box | tomato_soup_can | tomato_soup_can | 4/4 | ✅on | [view_el30_az240] The image shows a gelatin box and a can of tomato soup. The gelatin box is placed  |
| stack4_scene0014 | sponge | sugar_box | sugar_box | 2/4 | ✅on | [view_el30_az135] The image shows a box of Domino Sugar with a sponge placed underneath it. The spon |
| stack4_scene0017 | foam_brick | master_chef_can | foam_brick | 4/4 | ✅on | [view_el30_az210] In the image, there is a foam brick placed on top of a can of "Master Chef" coffee |
| stack4_scene0018 | tomato_soup_can | tuna_fish_can | tomato_soup_can | 3/4 | ✅on | [view_el30_az135] The image shows a can of "Campbell's Tomato Soup" and a can of "StarKist Tuna Fish |
| stack5_scene0005 | pudding_box | tuna_fish_can | tuna_fish_can | 4/4 | ✅on | [view_el30_az135] The image shows a can of tuna fish placed on top of a box of pudding. The tuna can |
| stack5_scene0008 | tomato_soup_can | tuna_fish_can | tomato_soup_can | 4/4 | ✅on | [view_el30_az195] The image shows two cans of food stacked on top of each other. The can on top appe |
| stack5_scene0011 | foam_brick | sponge | foam_brick | 2/3 | ✅on | [view_el30_az135] The image shows a foam brick placed on top of a sponge. The foam brick appears to  |
| stack5_scene0013 | foam_brick | sponge | foam_brick | 3/4 | ✅on | [view_el30_az135] The image shows a foam brick placed on top of a sponge. The foam brick appears to  |
| stack5_scene0014 | cracker_box | sugar_box | sugar_box | 4/4 | ✅on | [view_el30_az210] The image shows a box of Cheez-It crackers and a box of sugar. The Cheez-It box is |
| stack5_scene0016 | tomato_soup_can | wood_block | tomato_soup_can | 4/4 | ✅on | [view_el30_az240] In the image, there is a can of tomato soup placed on top of a wooden block. The c |
| stack5_scene0018 | gelatin_box | tomato_soup_can | tomato_soup_can | 4/4 | ✅on | [view_el30_az135] The image shows a can of "Campbells Tomato Soup" placed on top of a box of gelatin |
| stack5_scene0019 | cracker_box | sugar_box | sugar_box | 2/4 | ✅on | [view_el30_az135] The image shows two boxes of Cheez-It crackers. The box on the left is the origina |
| stack5_scene0020 | pudding_box | tuna_fish_can | tuna_fish_can | 4/4 | ✅on | [view_el30_az135] The image shows a can of StarKist brand light tuna fish placed on top of a box of  |

## 對照組中被誤判為 on 的(前 15)

| 場景 | A | B | 回答(截斷) |
|---|---|---|---|
| stack5_scene0019 | orange | sugar_box | [view_el30_az135] The image shows a box of Cheez-It crackers and an orange. The box of Cheez-It crac |
| stack5_scene0011 | tennis_ball | wood_block | [view_el30_az135] The image shows a tennis ball and a wooden block placed on a flat surface. The ten |
| stack3_scene0019 | mug | wood_block | [view_el30_az240] The image shows a red mug placed on a flat surface next to a wooden block. The mug |
| stack5_scene0011 | sponge | wood_block | [view_el30_az135] In the image, there is a wooden block and a sponge. The sponge is placed on a surf |
| stack5_scene0013 | sponge | sugar_box | [view_el30_az135] The image shows a 3D rendering of a sugar box and a sponge. The sugar box is posit |
| stack5_scene0019 | bowl | cracker_box | [view_el30_az135] The image shows a red bowl and a box of Cheez-It crackers. The bowl is placed to t |
| stack4_scene0013 | cracker_box | cups | [view_el30_az135] The image shows a box of Cheez-It crackers and two green cups. The box is standing |
| stack5_scene0018 | cups | tomato_soup_can | [view_el30_az240] The image shows a 3D rendering of a yellow cup and a can of tomato soup. The cup i |
| stack4_scene0006 | cups | gelatin_box | [view_el30_az210] The image shows a 3D rendering of a cup and a gelatin box. The cup appears to be a |
| stack4_scene0019 | medium_clamp | wood_block | [view_el30_az240] In the image, there is a medium clamp placed on top of a wooden block. The clamp a |
| stack4_scene0004 | rubiks_cube | wood_block | [view_el30_az135] In the image, there is a Rubik's Cube placed on top of a wooden block. The wooden  |
| stack4_scene0007 | sponge | tomato_soup_can | [view_el30_az210] The image shows a box of Domino Sugar and a can of tomato soup. The box of Domino  |
| stack5_scene0012 | gelatin_box | plum | [view_el30_az135] The image shows a gelatin box and a plum. The gelatin box is a red container with  |
| stack5_scene0001 | cracker_box | flat_screwdriver | [view_el30_az210] The image shows a box of Cheez-It crackers with a flat screwdriver placed next to  |
| stack5_scene0017 | master_chef_can | padlock | [view_el30_az240] In the image, there is a can of "Master Chef" coffee and a padlock. The can is pla |