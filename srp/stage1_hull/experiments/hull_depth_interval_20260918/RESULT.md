# Hull Depth Interval: Stage 1

## Purpose

Render a conservative per-pixel camera-depth interval `[z_enter, z_exit]`
from the existing `srp_hull_mv2_v12_am1` voxel hull.  This is a diagnostic
and search bound for a later RGB-only depth search; it does not alter a hull.

## Fixed input contract

- Hull: `data/eval/srp_hull_mv2_v12_am1/<scene>/hull.npz`
- Capture and masks: `captures_fast` and `mobilesamv2_fast`
- Views: the exact 12 names recorded in `build_meta`
- Geometry: every occupied voxel is conservatively rendered as a cube.

The script rejects a source hull whose metadata does not meet that contract.

## Command

```bash
/home/cho/.pyenv/versions/webots_visual_hull/bin/python3 \
  srp/stage1_hull/experiments/hull_depth_interval_20260918/render_intervals.py \
  stack3_scene0007 stack4_scene0007 stack5_scene0007
```

Outputs are written under `data/eval/hull_depth_interval_mv2_v12_am1/<scene>/`:

- `<view>_interval.npz`: `z_enter`, `z_exit`, and the input foreground mask
- `<view>_coverage.png`, `<view>_thickness.png`, `<view>_overlap.png`
- `summary.json`

In `overlap.png`, green is foreground supported by a hull interval, red is
foreground with no interval, and blue is an interval outside foreground.

## Initial result

| Scene | Mean foreground supported by interval | Minimum view support |
| --- | ---: | ---: |
| `stack3_scene0007` | 62.45% | 31.14% |
| `stack4_scene0007` | 51.73% | 23.53% |
| `stack5_scene0007` | 59.07% | 25.83% |

All views passed `z_enter <= z_exit`; the low coverage is therefore not an
invalid depth interval.  It shows that a per-view MobileSAM foreground union
contains substantial silhouette area that is not consistent with the 12-view,
allow-miss-1 hull.  A later photo-depth stage may search only pixels with an
interval.  Pixels without one must retain the original hull rather than being
treated as free space or failed depth evidence.

## Relation to the prior ray-search experiment

`photo_ray_search_20260917` searched only 0--3 voxels behind an outer surface.
It reduced ghost slightly but was not suitable as the main result.  The next
experiment should use this full ray interval, sample it coarsely then refine
near a supported NCC maximum, and carve only camera-to-selected-depth space
with two-source evidence.
