# 輸入形式對 VLM 的影響:orig vs crop vs maskbg

- 建檔 2026-09-25;程式 `srp/stage4_probe/llava_input_variants.py`;模型 `llava-hf/llama3-llava-next-8b-hf`,4-bit NF4。
- 影像由 `make_vlm_inputs.py` 產出:**裁切/去背都用 kept_object_masks + 手臂過濾(非 GT,可部署)**。
- `maskbg` 用來驗證 DISC Sec.II-C 的主張(遮罩去背造成 domain shift)。
- 差集探測的清單用 **GT 名 = 上界模擬**;`complete` 是**對照組**,正確答案為「沒有」。

## stack3_scene0005 / view_el30_az135(漏掉:**gelatin box**)

- 該視角可見:foam brick, gelatin box, rubber ball

### 輸入 `orig`(圖 1280x720)

**`count_stack`**(應指出堆疊;2s)

```
1. A red box-like object.
2. A blue spherical object.
3. A pair of black gun-like objects.

The red box-like object is stacked on top of the blue spherical object.
```

**`missing`**(應答 YES + gelatin box;2s)

```
Yes, there is another object in the image that is not a foam brick or a rubber ball. It appears to be a small, dark-colored object located in the lower left corner of the image. It looks like a small, three-dimensional shape, possibly a toy or a small figurine, but it's not clear enough to determine its exact nature.
```

**`complete`**(**應答 NO(對照組)**;2s)

```
Yes, there is another object in the image that is not in your list. It appears to be a small, cylindrical object with a metallic finish, located on the left side of the image, slightly above the foam brick. It has a reflective surface and a smooth, rounded shape.
```

### 輸入 `crop`(圖 374x219)

**`count_stack`**(應指出堆疊;1s)

```
1. A brown cube-shaped object.
2. A red rectangular object with text on it.

The brown cube-shaped object is stacked on top of the red rectangular object.
```

**`missing`**(應答 YES + gelatin box;2s)

```
Yes, there is another object in the image that is not a foam brick or a rubber ball. It appears to be a small, red, cylindrical object with a label on it. It is located on the left side of the image, slightly below the foam brick. The object has a shiny surface and seems to be a type of container or packaging.
```

**`complete`**(**應答 NO(對照組)**;1s)

```
No, there are no other physical objects visible in this image that are not in the list you provided. The image only contains a foam brick, a gelatin box, and a rubber ball.
```

### 輸入 `maskbg`(圖 1280x720)

**`count_stack`**(應指出堆疊;1s)

```
1. A red rectangular box.
2. A blue spherical object.

The red rectangular box is stacked on top of the blue spherical object.
```

**`missing`**(應答 YES + gelatin box;1s)

```
No, there are no other physical objects visible in this image that are not already listed. The image shows a foam brick and a rubber ball, which are the only objects present.
```

**`complete`**(**應答 NO(對照組)**;1s)

```
No, there are no other physical objects visible in this image that are not in the list you provided. The image shows a foam brick, a gelatin box, and a rubber ball.
```

## stack4_scene0007 / view_el45_az225(漏掉:**sponge**)

- 該視角可見:sponge, sugar box, tomato soup can

### 輸入 `orig`(圖 1280x720)

**`count_stack`**(應指出堆疊;2s)

```
1. A white rectangular sign with blue and yellow text and a blue and yellow logo.
2. A cylindrical can with a red label and a white label.

The white rectangular sign is not stacked on top of any object. The cylindrical can is not stacked on top of any object. Both objects are placed on a flat surface.
```

**`missing`**(應答 YES + sponge;2s)

```
Yes, there is another object in the image that is not in the list. It appears to be a small, cylindrical object with a metallic sheen, possibly a can or a container, located to the right of the sugar box. It is standing upright and has a label on it, but the specific contents or brand are not clearly visible.
```

**`complete`**(**應答 NO(對照組)**;2s)

```
Yes, there is another object in the image that is not in your list. It appears to be a small, cylindrical object with a label on it, located on the right side of the image. It is not a sponge, a sugar box, or a tomato soup can. The object is standing upright and is positioned slightly to the right of the central object, which is the sugar box.
```

### 輸入 `crop`(圖 400x356)

**`count_stack`**(應指出堆疊;1s)

```
1. Domino Sugar box
2. Can of food

The Domino Sugar box is stacked on top of a yellow sponge.
```

**`missing`**(應答 YES + sponge;2s)

```
No, there are no other physical objects visible in this image that are not in the list. The image only shows a sugar box and a can of tomato soup.
```

**`complete`**(**應答 NO(對照組)**;2s)

```
No, there are no other physical objects visible in this image that are not in the list you provided. The image shows a sponge, a sugar box, and a tomato soup can.
```

### 輸入 `maskbg`(圖 1280x720)

**`count_stack`**(應指出堆疊;2s)

```
1. A white rectangular sign with blue and yellow text and a blue border.
2. A cylindrical object with a red label and a silver top.

The white sign is not stacked on top of any object. The cylindrical object is not stacked on top of the sign. They are both separate objects in the image.
```

**`missing`**(應答 YES + sponge;1s)

```
No, there are no other physical objects visible in this image that are not in the list you provided. The image only shows a sugar box and a tomato soup can.
```

**`complete`**(**應答 NO(對照組)**;1s)

```
No, there are no other physical objects visible in this image that are not in the list you provided. The image shows a sponge, a sugar box, and a tomato soup can.
```

## stack4_scene0010 / view_el30_az195(漏掉:**tuna fish can**)

- 該視角可見:coffee can, tuna fish can, spray bottle

### 輸入 `orig`(圖 1280x720)

**`count_stack`**(應指出堆疊;3s)

```
1. A blue can with a white label and a blue stripe.
2. A red can with a white label and a blue stripe.
3. A black can with a yellow label and a blue stripe.
4. A red spray can with a white label and a blue stripe.
5. A black spray can with a yellow label and a blue stripe.

The objects are not stacked on top of each other. They are all placed separately on the surface.
```

**`missing`**(應答 YES + tuna fish can;2s)

```
Yes, there is another object in the image that is not in the list. It appears to be a red cap or lid, which is floating in the air above the spray bottle. The cap is positioned to the right of the spray bottle and is slightly above it.
```

**`complete`**(**應答 NO(對照組)**;2s)

```
Yes, there is another object in the image that is not in your list. It appears to be a red cap or lid, which is floating in the air above the spray bottle. The cap is located in the upper right corner of the image.
```

### 輸入 `crop`(圖 434x450)

**`count_stack`**(應指出堆疊;2s)

```
1. A can of MasterChef coffee.
2. A can of Windex glass cleaner.
3. A red and white spray bottle.

The can of MasterChef coffee is stacked on top of the can of Windex glass cleaner.
```

**`missing`**(應答 YES + tuna fish can;2s)

```
Yes, there is another object in the image that is not in the list you provided. It is a can of "Master Chees" located on the left side of the image, below the spray bottle. The can is blue and has a picture of a cup of coffee on it.
```

**`complete`**(**應答 NO(對照組)**;2s)

```
Yes, there is another object in the image that is not in your list. It is a can of "Master Chees" located on the left side of the image, below the spray bottle. The can is blue and has a picture of a cup of coffee on it.
```

### 輸入 `maskbg`(圖 1280x720)

**`count_stack`**(應指出堆疊;2s)

```
1. A blue can with a white label and a blue stripe.
2. A red spray can with a white nozzle.
3. A black spray can with a yellow label and a blue stripe.

The spray can is stacked on top of the blue can.
```

**`missing`**(應答 YES + tuna fish can;1s)

```
No, there are no other physical objects visible in this image that are not already listed. The image shows a spray bottle and a coffee can, which are the only objects present.
```

**`complete`**(**應答 NO(對照組)**;1s)

```
No, there are no other physical objects visible in this image that are not in the list you provided. The image shows a spray bottle, a coffee can, and a tuna fish can.
```
