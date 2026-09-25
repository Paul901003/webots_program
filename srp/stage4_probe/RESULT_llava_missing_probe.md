# VLM 差集探測:給已偵測清單,問畫面中還有沒有清單外的物體

- 建檔 2026-09-25;程式 `srp/stage4_probe/llava_missing_probe.py`;模型 `llava-hf/llama3-llava-next-8b-hf`,**4-bit NF4**。
- 可見性門檻:GT modal 遮罩面積 ≥ 300 px。GEX 物體列入清單(畫面中確實存在)。
- ⚠ **清單用 GT 物體名 = 上界模擬**,非可部署方法(真實 instance 無名字)。
- **對照組**:`complete`(清單完整)正確答案是「沒有」;若也答「有」則訊號無效。

## stack3_scene0005 / view_el30_az135

- 該視角可見物體(面積 px):{'foam_brick': 8313, 'gelatin_box': 4583, 'racquetball': 2989}
- 被管線漏掉:**gelatin box**

**條件 `missing`** — 清單:foam brick, rubber ball  (應答:有,gelatin box;2s)

```
Yes, there is another object in the image that is not a foam brick or a rubber ball. It appears to be a small, dark-colored object located in the lower left corner of the image. It looks like a small, three-dimensional shape, possibly a toy or a small figurine, but it's not clear enough to determine its exact nature.
```

**條件 `complete`** — 清單:foam brick, gelatin box, rubber ball  (應答:沒有;2s)

```
Yes, there is another object in the image that is not in your list. It appears to be a small, cylindrical object with a metallic finish, located on the left side of the image, slightly above the foam brick. It has a reflective surface and a smooth, rounded shape.
```

## stack3_scene0005 / view_el75_az225

- 該視角可見物體(面積 px):{'foam_brick': 6487, 'gelatin_box': 2561, 'racquetball': 3760}
- 被管線漏掉:**gelatin box**

**條件 `missing`** — 清單:foam brick, rubber ball  (應答:有,gelatin box;2s)

```
Yes, there is another object in the image that is not in the list you provided. It appears to be a small, black, rectangular object with a flat surface, located in the upper left corner of the image. It is positioned slightly above the foam brick and to the left of the rubber ball.
```

**條件 `complete`** — 清單:foam brick, gelatin box, rubber ball  (應答:沒有;2s)

```
No, there are no other physical objects visible in this image that are not in the list you provided. The image shows a foam brick, a gelatin box, and a rubber ball, which are the only objects present.
```

## stack4_scene0007 / view_el45_az225

- 該視角可見物體(面積 px):{'sponge': 4412, 'sugar_box': 23170, 'tomato_soup_can': 8833}
- 被管線漏掉:**sponge**

**條件 `missing`** — 清單:sugar box, tomato soup can  (應答:有,sponge;2s)

```
Yes, there is another object in the image that is not in the list. It appears to be a small, cylindrical object with a metallic sheen, possibly a can or a container, located to the right of the sugar box. It is standing upright and has a label on it, but the specific contents or brand are not clearly visible.
```

**條件 `complete`** — 清單:sponge, sugar box, tomato soup can  (應答:沒有;2s)

```
Yes, there is another object in the image that is not in your list. It appears to be a small, cylindrical object with a label on it, located on the right side of the image. It is not a sponge, a sugar box, or a tomato soup can. The object is standing upright and is positioned slightly to the right of the central object, which is the sugar box.
```

## stack4_scene0007 / view_el30_az210

- 該視角可見物體(面積 px):{'sponge': 4407, 'sugar_box': 27151, 'tomato_soup_can': 7410}
- 被管線漏掉:**sponge**

**條件 `missing`** — 清單:sugar box, tomato soup can  (應答:有,sponge;2s)

```
Yes, there is another object in the image that is not in the list. It appears to be a small, cylindrical object with a metallic sheen, possibly a can or a container, located to the right of the sugar box. It is standing upright and has a label on it, but the specific details of the label are not clear.
```

**條件 `complete`** — 清單:sponge, sugar box, tomato soup can  (應答:沒有;2s)

```
Yes, there is another object in the image that is not in the list you provided. It appears to be a small, cylindrical object with a metallic sheen, possibly a can or a container, located to the right of the sugar box. It is standing upright and has a label on it, but the specific contents or brand are not clearly visible.
```

## stack4_scene0010 / view_el75_az225

- 該視角可見物體(面積 px):{'lemon': 3082, 'master_chef_can': 5750, 'tuna_fish_can': 12484, 'windex_bottle': 24490}
- 被管線漏掉:**tuna fish can**

**條件 `missing`** — 清單:lemon, coffee can, spray bottle  (應答:有,tuna fish can;2s)

```
Yes, there is another object in the image that is not in your list. It appears to be a red and black object, possibly a toy or a small figurine, located in the lower right corner of the image. It is not clearly identifiable, but it is distinct from the lemon, coffee can, and spray bottle.
```

**條件 `complete`** — 清單:lemon, coffee can, tuna fish can, spray bottle  (應答:沒有;2s)

```
Yes, there is another object in the image that is not in your list. It appears to be a red and black object, possibly a toy or a small figurine, located in the lower right corner of the image. It is not clearly identifiable from this angle, but it is distinct from the other objects you've listed.
```

## stack4_scene0010 / view_el30_az195

- 該視角可見物體(面積 px):{'master_chef_can': 20025, 'tuna_fish_can': 7350, 'windex_bottle': 23535}
- 被管線漏掉:**tuna fish can**

**條件 `missing`** — 清單:coffee can, spray bottle  (應答:有,tuna fish can;2s)

```
Yes, there is another object in the image that is not in the list. It appears to be a red cap or lid, which is floating in the air above the spray bottle. The cap is positioned to the right of the spray bottle and is slightly above it.
```

**條件 `complete`** — 清單:coffee can, tuna fish can, spray bottle  (應答:沒有;2s)

```
Yes, there is another object in the image that is not in your list. It appears to be a red cap or lid, which is floating in the air above the spray bottle. The cap is located in the upper right corner of the image.
```
