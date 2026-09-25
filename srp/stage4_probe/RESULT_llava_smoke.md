# LLaVA-1.6 對「管線漏找」失敗案例的描述

- 建檔 2026-09-25;程式 `srp/stage4_probe/llava_scene_desc.py`;模型 `llava-hf/llama3-llava-next-8b-hf`,**4-bit NF4**。
- 目的:看 VLM 能不能指出被幾何管線漏掉的堆疊物(獨立於 hull 幾何的訊號)。
- 案例來自 `RESULT_match_eval_*`(瘦中心drop / 瘦fp drop 的沒找到物體)。
- **原始輸出未經編輯**;判讀另外標註。

## stack4_scene0010 / view_el75_az225

- 管線漏掉的物體:**tuna_fish_can**;視角備註:俯視,tuna 面積最大但疑似看不出兩罐

**prompt `bbq`** (5s)

> Describe visible object in front of you, paying close attention to its spatial dimensions and visual attributes.

```
In the image provided, there are three objects visible against a plain, light-colored background. The objects appear to be floating in mid-air, suggesting they might be part of a 3D animation or a virtual environment.

1. The first object is a cylindrical container with a blue and white striped pattern. It has a metallic sheen and a black top, which could indicate a lid or cap. The container is positioned centrally in the image.

2. The second object is a yellow spherical object, which could be a ball or a similar shape. It is located slightly to the right of the cylindrical container.

3. The third object is a red and black object that resembles a helmet or a headpiece. It is positioned to the right of the yellow sphere and is slightly tilted, giving it a dynamic appearance.

The objects are simple in design, with no intricate details or textures, and they are presented in a way that suggests they are part of a game or a simulation where they might be
```

**prompt `count_stack`** (3s)

> List every distinct physical object you can see on the table. For each one, give its name. Then state clearly whether any object is stacked on top of another object, and which is on top of which.

```
On the table, I can see the following distinct physical objects:

1. A yellow ball.
2. A blue and white can.
3. A red and black can.
4. A red and black bottle.

The yellow ball is not stacked on top of any other object. The blue and white can is not stacked on top of any other object. The red and black can is not stacked on top of any other object. The red and black bottle is not stacked on top of any other object.
```

## stack4_scene0010 / view_el30_az195

- 管線漏掉的物體:**tuna_fish_can**;視角備註:側視,兩罐應清楚可分

**prompt `bbq`** (5s)

> Describe visible object in front of you, paying close attention to its spatial dimensions and visual attributes.

```
In the image, there are three objects visible in front of you. The first object is a blue can with a white label that features a graphic of a bird and the word "MASCOT" in white letters. The can has a silver cap and is positioned on the left side of the image.

The second object is a red can with a white label that reads "Windex" in white letters. This can is located in the center of the image.

The third object is a black can with a yellow label that has the word "Windex" in white letters. This can is on the right side of the image.

All three cans appear to be of a similar size and shape, with a cylindrical form and a flat base. The cans are placed on a flat surface that is white in color. The background is a gradient of gray, suggesting a three-dimensional space. The image is rendered in a 3D graphics style, which gives it a realistic appearance.
```

**prompt `count_stack`** (3s)

> List every distinct physical object you can see on the table. For each one, give its name. Then state clearly whether any object is stacked on top of another object, and which is on top of which.

```
On the table, I can see the following distinct physical objects:

1. A blue can with a label that reads "Mascot" and a picture of a mascot.
2. A red can with a label that reads "Windex" and a picture of a window.
3. A black spray bottle with a red cap.

The objects are not stacked on top of each other. The blue can is on the left, the red can is in the middle, and the black spray bottle is on the right.
```
