# Result: Foreground Patch and Positive Evidence Guards

## Fixed Scene

- Scene: `stack4_scene0007`
- Input: `srp_hull_mv2_v12_am1`
- Baseline photo: `srp_hull_mv2_v12_am1_photo`
- Voxel size: 5 mm
- Views: 12
- NCC threshold: 0.1
- Sources required: 2
- Patch: 5 x 5
- Normal candidates: hull normal plus eight 20 degree tangent offsets

## Variants

| Variant | Remaining voxels | Removed voxels | Sugar visible-surface removed | GT coverage | Ghost ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| No photo | 12649 | 0 | 0 / 1202 | 0.8979 | 0.1420 |
| Original mean | 11986 | 663 | 275 / 1202 | 0.8745 | 0.1181 |
| Foreground patch plus mean | 12018 | 631 | 275 / 1202 | 0.8775 | 0.1175 |
| Foreground patch plus max | 12649 | 0 | 0 / 1202 | 0.8979 | 0.1420 |
| Foreground patch plus second | 12478 | 171 | 120 / 1202 | 0.8901 | 0.1378 |

## Interpretation

Foreground union patch checking alone does not fix the stacked-object failure:
the false source evidence is still foreground. Taking the maximum NCC over all
sources is too conservative and performs no carving.

The second-best NCC rule is the current candidate. It retains a voxel when one
source is occluded or mismatched but two sources support a shared plane normal.
It removes less ghost volume than the original mean, but cuts the sugar-box
visible-surface loss from 22.9 percent to 10.0 percent.

## Decision Gate

Do not promote this variant yet. It must next be evaluated on a fixed,
representative scene set and then through the downstream Stage 2 separation
metrics. Promotion requires lower surface loss without giving back all of the
original photo carving's ghost reduction.
