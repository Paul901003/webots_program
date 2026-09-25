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
| **GT 有 on** | **TP 28** | FN 1 | 29 |
| **GT 沒 on** | **FP 24** | TN 36 | 60 |

| 指標 | 值 |
|---|---|
| 召回率 | 96.6% |
| **假陽性率** | **40.0%** |
| 精確率 | 53.8% |
| **平衡準確率** | **78.3%** |

(對照:整張場景圖版 52.8%;單視角成對版 68.9%)

## 投票門檻掃描(同一批推論,只改判定門檻)

| 需幾票 | 召回率 | 假陽性率 | 精確率 | 平衡準確率 |
|---|---|---|---|---|
| ≥1/3 | 96.6% | 80.0% | 36.8% | **58.3%** |
| ≥2/3 | 96.6% | 40.0% | 53.8% | **78.3%** |
| ≥3/3 | 79.3% | 20.0% | 65.7% | **79.7%** |

## 逐對(GT 有 on 的 29 對全列)

| 場景 | A | B | 上物 | 票數 | 判讀 | 回答(截斷) |
|---|---|---|---|---|---|---|
| stack3_scene0005 | foam_brick | gelatin_box | foam_brick | 3/3 | ✅on | [view_el30_az135] The image shows a foam brick placed on top of a gelatin box. The foam brick appear |
| stack3_scene0006 | tomato_soup_can | tuna_fish_can | tomato_soup_can | 2/3 | ✅on | [view_el30_az240] The image shows two cans of food stacked on top of each other. The can on top appe |
| stack3_scene0011 | tomato_soup_can | tuna_fish_can | tomato_soup_can | 3/3 | ✅on | [view_el30_az135] The image shows two cans of food stacked on top of each other. The top can is a ca |
| stack3_scene0013 | foam_brick | sponge | foam_brick | 3/3 | ✅on | [view_el60_az180] The image shows a foam brick with three holes in it, placed on top of a sponge. Th |
| stack3_scene0014 | master_chef_can | tomato_soup_can | tomato_soup_can | 3/3 | ✅on | [view_el45_az225] The image shows two cans of food stacked on top of each other. The can on top appe |
| stack3_scene0015 | foam_brick | gelatin_box | foam_brick | 3/3 | ✅on | [view_el30_az240] The image shows a foam brick placed on top of a gelatin box. The foam brick appear |
| stack3_scene0016 | tomato_soup_can | tuna_fish_can | tomato_soup_can | 2/3 | ✅on | [view_el30_az240] The image shows a can of tomato soup and a can of tuna fish. These two items are o |
| stack3_scene0017 | master_chef_can | tomato_soup_can | tomato_soup_can | 3/3 | ✅on | [view_el45_az225] The image shows two cans of food stacked on top of each other. The can on top appe |
| stack3_scene0019 | foam_brick | wood_block | foam_brick | 3/3 | ✅on | [view_el60_az195] The image shows a foam brick and a wooden block placed on top of each other. The f |
| stack4_scene0002 | tomato_soup_can | tuna_fish_can | tomato_soup_can | 3/3 | ✅on | [view_el30_az135] The image shows two cans of food stacked on top of each other. The top can is labe |
| stack4_scene0003 | master_chef_can | pudding_box | master_chef_can | 3/3 | ✅on | [view_el30_az240] In the image, there is a can of "Master Chef" coffee and a box of "Master Chef" pu |
| stack4_scene0006 | gelatin_box | tomato_soup_can | tomato_soup_can | 3/3 | ✅on | [view_el30_az210] The image shows a gelatin box and a can of tomato soup placed on a flat surface. T |
| stack4_scene0007 | sponge | sugar_box | sugar_box | 3/3 | ✅on | [view_el45_az225] The image shows a box of Domino Sugar placed on top of a sponge. The sponge is lik |
| stack4_scene0009 | master_chef_can | pudding_box | master_chef_can | 3/3 | ✅on | [view_el30_az135] The image shows a blue coffee can with the text "Coffee" visible on it, placed on  |
| stack4_scene0010 | master_chef_can | tuna_fish_can | tuna_fish_can | 2/3 | ✅on | [view_el60_az180] The image shows a can of coffee and a can of tuna fish. The can of coffee is posit |
| stack4_scene0011 | master_chef_can | tomato_soup_can | tomato_soup_can | 3/3 | ✅on | [view_el60_az180] The image shows two cans of food stacked on top of each other. The top can appears |
| stack4_scene0012 | gelatin_box | tomato_soup_can | tomato_soup_can | 3/3 | ✅on | [view_el30_az240] The image shows a gelatin box and a can of tomato soup. The gelatin box is placed  |
| stack4_scene0014 | sponge | sugar_box | sugar_box | 3/3 | ✅on | [view_el45_az135] The image shows a box of Domino sugar placed on top of a yellow sponge. The sponge |
| stack4_scene0017 | foam_brick | master_chef_can | foam_brick | 3/3 | ✅on | [view_el60_az195] The image shows a 3D rendering of a foam brick placed on top of a can of coffee. T |
| stack4_scene0018 | tomato_soup_can | tuna_fish_can | tomato_soup_can | 2/3 | ✅on | [view_el30_az135] The image shows a can of tomato soup and a can of tuna fish. These two products ar |
| stack5_scene0005 | pudding_box | tuna_fish_can | tuna_fish_can | 3/3 | ✅on | [view_el45_az135] The image shows a 3D rendering of a pudding box and a can of tuna fish. The puddin |
| stack5_scene0008 | tomato_soup_can | tuna_fish_can | tomato_soup_can | 3/3 | ✅on | [view_el30_az195] The image shows two cans of food placed side by side. The can on the left appears  |
| stack5_scene0011 | foam_brick | sponge | foam_brick | 0/3 | ❌無 | [view_el90] The image you've provided appears to be a composite of various objects that are not rela |
| stack5_scene0013 | foam_brick | sponge | foam_brick | 3/3 | ✅on | [view_el90] The image you've provided appears to show a foam brick with two black dots on it, which  |
| stack5_scene0014 | cracker_box | sugar_box | sugar_box | 3/3 | ✅on | [view_el60_az255] The image shows a cracker box and a sugar box stacked on top of each other. The cr |
| stack5_scene0016 | tomato_soup_can | wood_block | tomato_soup_can | 3/3 | ✅on | [view_el60_az255] In the image, there is a can of tomato soup placed on top of a wooden block. The c |
| stack5_scene0018 | gelatin_box | tomato_soup_can | tomato_soup_can | 3/3 | ✅on | [view_el30_az135] The image shows a gelatin box and a can of tomato soup. The gelatin box is placed  |
| stack5_scene0019 | cracker_box | sugar_box | sugar_box | 2/3 | ✅on | [view_el60_az180] The image shows two boxes stacked on top of each other. The top box is a box of Do |
| stack5_scene0020 | pudding_box | tuna_fish_can | tuna_fish_can | 3/3 | ✅on | [view_el30_az135] The image shows a can of tuna fish placed on top of a box of pudding. The relation |

## 對照組中被誤判為 on 的(前 15)

| 場景 | A | B | 回答(截斷) |
|---|---|---|---|
| stack5_scene0019 | orange | sugar_box | [view_el30_az135] The image shows a box of Cheez-It crackers and an orange. The box of Cheez-It crac |
| stack3_scene0019 | mug | wood_block | [view_el60_az255] The image shows a red mug and a wooden block with holes in it. The mug appears to  |
| stack5_scene0011 | sponge | wood_block | [view_el90] The image shows a sponge and a wooden block. The sponge appears to be in the foreground, |
| stack5_scene0013 | sponge | sugar_box | [view_el90] The image shows a sponge and a sugar box. The sponge is positioned below the sugar box,  |
| stack5_scene0019 | bowl | cracker_box | [view_el75_az105] The image shows a red bowl with a box of Domino Sugar on top of it. The box of Dom |
| stack5_scene0018 | cups | tomato_soup_can | [view_el60_az255] The image shows a 3D rendering of a yellow cup and a can of tomato soup. The cup a |
| stack4_scene0006 | cups | gelatin_box | [view_el30_az210] In the image provided, there is a can of food placed on top of a red gelatin box.  |
| stack4_scene0004 | rubiks_cube | wood_block | [view_el30_az135] In the image, there is a Rubik's Cube placed on top of a wooden block. The wooden  |
| stack4_scene0007 | sponge | tomato_soup_can | [view_el45_az225] The image shows a box of Domino Sugar and a can of tomato soup. The box of Domino  |
| stack4_scene0014 | pitcher_base | sugar_box | [view_el45_az135] The image shows a blue pitcher base and a box of Domino Sugar. The pitcher base is |
| stack5_scene0017 | master_chef_can | padlock | [view_el30_az240] In the image provided, there is a can of "Master Chef" coffee and a padlock. The c |
| stack5_scene0016 | fork | wood_block | [view_el60_az180] The image shows a fork and a wooden block. The fork is positioned in front of the  |
| stack4_scene0001 | cups | wood_block | [view_el30_az240] In the image, there is a wooden block with a blue object on top of it. Below the b |
| stack5_scene0018 | cups | golf_ball | [view_el45_az135] In the image provided, there is a can of tomatoes placed on top of a box of what a |
| stack5_scene0013 | padlock | sugar_box | [view_el45_az135] The image shows a padlock and a box of Domino sugar. The padlock is a security dev |