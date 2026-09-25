# 擴大測試:裁切輸入下 LLaVA 的堆疊偵測(含對照組)

- 建檔 2026-09-25;程式 `srp/stage4_probe/llava_stack_batch.py`;模型 `llava-hf/llama3-llava-next-8b-hf`,4-bit NF4。
- 影像:`make_vlm_inputs_batch.py` 產的 **crop**;視角=el30 中 kept 面積最大者(**可部署,不用 GT**)。
- 母體 **90 場**:stack 組 60(29 有 on 對 / 31 無)+ occ 組 30(全無)。
- 真值 = `relations.json` 的 on 關係(排 GEX);判讀 = 關鍵詞抽取(需人工覆核)。

## 混淆矩陣

| | 模型說「有堆疊」 | 模型說「沒有」或不明 | 合計 |
|---|---|---|---|
| **實際有堆疊** | **TP 23** | FN 6 | 29 |
| **實際沒堆疊** | **FP 45** | TN 16 | 61 |

| 指標 | 值 |
|---|---|
| 召回率(有堆疊被抓到) | 79.3% |
| **假陽性率(沒堆疊卻說有)** | **73.8%** |
| 精確率(說有堆疊時是對的) | 33.8% |
| 平衡準確率 | 52.8% |

⚠ **假陽性率是關鍵**:若接近召回率,代表模型只是傾向說「有堆疊」,訊號無價值。

## 逐場(前 40)

| 場景 | 組 | 實際有on | 判讀 | 回答(截斷) |
|---|---|---|---|---|
| stack3_scene0001 | stack3 | 否 | YES | 1. A purple cylindrical object with a striped pattern. 2. A blue cube-shaped object. 3. A yellow rectangular o |
| stack3_scene0002 | stack3 | 否 | YES | 1. A blue cube-shaped object. 2. A green cylindrical object with a blue top. 3. A red rectangular object.  The |
| stack3_scene0003 | stack3 | 否 | NO | 1. A blue canister or container. 2. A yellow tennis ball.  The blue canister is not stacked on top of the tenn |
| stack3_scene0004 | stack3 | 否 | NO | 1. A baseball. 2. A bottle of Domino Sugar.  The baseball is not stacked on top of the bottle of Domino Sugar. |
| stack3_scene0005 | stack3 | 是 | YES | 1. A red rectangular block with three holes. 2. A red rectangular box with text on it. 3. A blue spherical obj |
| stack3_scene0006 | stack3 | 是 | YES | 1. A can of food. 2. A bottle of cleaning solution.  The can of food is stacked on top of the bottle of cleani |
| stack3_scene0007 | stack3 | 否 | YES | 1. A blue cube-shaped object on top of a red box. 2. A red box with a yellow label that reads "Cheez-It" and " |
| stack3_scene0008 | stack3 | 否 | YES | 1. A blue cube on top of a colorful cube structure. 2. A colorful cube structure, which appears to be made up  |
| stack3_scene0009 | stack3 | 否 | NO | 1. A blue container with a green top. 2. A peach with a red and orange color gradient.  The blue container is  |
| stack3_scene0010 | stack3 | 否 | YES | 1. A blue block on top of a red box. 2. A red box with a label that reads "Cheez-It" and "Are these cheeses re |
| stack3_scene0011 | stack3 | 是 | YES | 1. A can of food. 2. A can of food. 3. A power drill.  The can of food is stacked on top of the power drill. |
| stack3_scene0012 | stack3 | 否 | YES | 1. A can of food. 2. A sponge.  The can of food is stacked on top of the sponge. |
| stack3_scene0013 | stack3 | 是 | YES | 1. A black screwdriver. 2. A red block with three holes. 3. A yellow block with one hole.  The red block is st |
| stack3_scene0014 | stack3 | 是 | ? | 1. A can of "Master Chef" coffee. 2. A can of "Master Chef" coffee. 3. A can of "Master Chef" coffee. 4. A can |
| stack3_scene0015 | stack3 | 是 | YES | 1. A red block. 2. A red box. 3. A wrench.  The red block is stacked on top of the red box. |
| stack3_scene0016 | stack3 | 是 | YES | 1. A can of food. 2. A box of food.  The can of food is stacked on top of the box of food. |
| stack3_scene0017 | stack3 | 是 | YES | 1. A can of "Master Chef" food. 2. A baseball. 3. A can of "Master Chef" food, which is stacked on top of the  |
| stack3_scene0018 | stack3 | 否 | ? | 1. Domino Sugar box 2. Domino Sugar box lid 3. Domino Sugar box top 4. Domino Sugar box label 5. Domino Sugar  |
| stack3_scene0019 | stack3 | 是 | YES | 1. A red cube-shaped object. 2. A wooden block with a flat top. 3. A red cylindrical object.  The red cube-sha |
| stack3_scene0020 | stack3 | 否 | YES | 1. A green cylindrical object. 2. A blue cylindrical object.  The blue cylindrical object is stacked on top of |
| stack4_scene0001 | stack4 | 否 | YES | 1. A blue cube 2. A red bucket 3. A wooden block  The blue cube is stacked on top of the wooden block. |
| stack4_scene0002 | stack4 | 是 | YES | 1. A can of food. 2. A red marker or pen. 3. A blue object, possibly a box or a container.  The can of food is |
| stack4_scene0003 | stack4 | 是 | YES | 1. Apple 2. Wrench 3. Coffee canister  The coffee canister is stacked on top of the box. |
| stack4_scene0004 | stack4 | 否 | YES | 1. A white spray bottle with a green label. 2. A box of "Color Cubes" with a blue label. 3. A stack of colorfu |
| stack4_scene0005 | stack4 | 否 | NO | 1. Fork 2. Knife 3. Cup 4. Block 5. Napkin  The objects are not stacked on top of each other. They are placed  |
| stack4_scene0006 | stack4 | 是 | YES | 1. A colorful box of Kleenex cubes. 2. A can of Campbell's Tomato Soup. 3. A can of Campbell's Tomato Soup on  |
| stack4_scene0007 | stack4 | 是 | YES | 1. Domino Sugar box 2. Frozen Mocha Latte Mix packet 3. Can of "Hull" product  The Frozen Mocha Latte Mix pack |
| stack4_scene0008 | stack4 | 否 | NO | 1. Blue canister with a blue lid. 2. Orange canister with a blue lid. 3. Blue cup. 4. Orange ball.  The orange |
| stack4_scene0009 | stack4 | 是 | YES | 1. A yellow spray bottle with a blue label. 2. A red-handled knife. 3. A blue can with a yellow label.  The sp |
| stack4_scene0010 | stack4 | 是 | NO | 1. A can of "Master Chef" brand food. 2. A yellow ball. 3. A red and white chef's hat. 4. A black and yellow c |
| stack4_scene0011 | stack4 | 是 | YES | 1. Tomato Soup can 2. Coffee can 3. Sponge 4. Peach  The Tomato Soup can is stacked on top of the Coffee can,  |
| stack4_scene0012 | stack4 | 是 | YES | 1. Banana 2. Can of food 3. Spoon  The can of food is stacked on top of the banana. |
| stack4_scene0013 | stack4 | 否 | YES | 1. A box of Cheez-It crackers. 2. A green pear. 3. A blue cube on top of the Cheez-It box. 4. A label on the C |
| stack4_scene0014 | stack4 | 是 | YES | 1. Blue trash can 2. Domino Sugar box 3. Rubik's Cube  The Domino Sugar box is stacked on top of the Rubik's C |
| stack4_scene0015 | stack4 | 否 | YES | 1. A can of food. 2. A stack of plates. 3. A colorful block structure. 4. A box of Jell-O.  The can of food is |
| stack4_scene0016 | stack4 | 否 | YES | 1. Domino Sugar box 2. Banana 3. Blue cup  The Domino Sugar box is stacked on top of the banana. |
| stack4_scene0017 | stack4 | 是 | ? | 1. A brown rectangular block with holes. 2. A blue cylindrical can with a label that reads "Master Chef." 3. A |
| stack4_scene0018 | stack4 | 是 | ? | 1. A can of "Cheez-It" crackers. 2. A can of "Cheez-It" crackers. 3. A can of "Cheez-It" crackers. 4. A can of |
| stack4_scene0019 | stack4 | 否 | YES | 1. A wooden block. 2. A blue cube. 3. A red strawberry with green leaves.  The blue cube is stacked on top of  |
| stack4_scene0020 | stack4 | 否 | YES | 1. A blue cup. 2. A green can with a blue lid. 3. A white ball.  The green can is stacked on top of the blue c |