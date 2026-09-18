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

## Stage 2: Interval-Ray NCC on `stack4_scene0007`

`interval_candidates.py` expands each outer surface voxel only along the ray
of its most front-facing camera.  It retains an inward candidate only when it
is both inside that pixel's interval and still occupied by the source hull.
For `stride=3` voxels (15 mm) and `max-step=60` (300 mm), the candidate set is
small enough to score directly:

| Quantity | Value |
| --- | ---: |
| Surface voxels | 3,710 |
| Referenceable surface voxels | 3,139 |
| With a deeper interval | 2,856 |
| Coarse candidate points | 15,735 |
| Candidate count p50 / p95 per surface voxel | 4 / 11 |
| NCC time | 1.7 s |

`run_interval_ray_search.py` uses the exact Stage-1 foreground contract
(`mobilesamv2_fast`, FK arm subtraction, `captures_fast`) and exactly the old
ray-search NCC thresholds: two sources, foreground patch guard, reject outer
NCC `<0.10`, accept deeper NCC `>=0.25`, and require a `>=0.10` improvement.

| Method | Removed | Coverage | Ghost |
| --- | ---: | ---: | ---: |
| Raw am1 | 0 | 89.79% | 14.20% |
| Existing `fgpatch_second` | 171 | 89.01% | 13.78% |
| Previous 0--3 voxel ray search | 275 | 88.46% | 13.59% |
| Full interval, one reference | 1,163 | 81.99% | 13.72% |
| Full interval, two reference agreement | 365 | 87.36% | 14.04% |
| Full interval, three reference agreement | 55 | 89.50% | 14.10% |

The one-reference result over-carves.  Requiring two or three distinct
reference rays to choose nearby endpoints reduces that error, but does not
beat the existing photo result.  Do **not** run semantic clustering or div on
these outputs.  The missing evidence is source-view depth consistency: a
candidate needs to be selected as a depth optimum in the source view after
reprojection, not merely obtain a good pairwise NCC there.

## Stage 3: Sparse Independent Depth Maps on `stack4_scene0007`

`multi_ref_candidates.py` gives every front-facing, foreground-supported hull
surface pixel its own reference-ray candidates. It keeps only the nearest
current hull surface for a reference pixel. `depth_consistency.py` then holds
that reference fixed during NCC, creates sparse per-view depth maps, and
requires a deeper proposal to reproject within a 3-D tolerance of independently
selected points in other source depth maps.

The input produced 18,686 reference rays, 79,996 candidates, 72,176 points
with a two-source NCC score, and 16,902 sparse depth-map entries. NCC took
about 8 seconds. The following are mesh-occupancy measurements:

| Method | Source support / tolerance | Coverage | Ghost |
| --- | --- | ---: | ---: |
| Raw am1 | - | 89.79% | 14.20% |
| Existing `fgpatch_second` | - | 89.01% | 13.78% |
| Depth consistency | 2 / 2 voxels | 82.32% | 10.30% |
| Depth consistency | 2 / 1 voxel | 85.66% | 11.79% |
| Depth consistency | 3 / 1 voxel | 87.50% | 12.83% |

The geometric check does reduce ghost, but even its conservative version
removes too much true surface for this project. This is a self-consistent
wrong-depth failure: multiple sparse NCC maps can select a similar incorrect
interior point when a real surface has weak or view-dependent appearance.
Do **not** feed these hulls to semantic clustering or div.

To approach COLMAP quality, a later method needs more than pointwise NCC plus
depth agreement: it needs spatially regularized, view-visibility-aware depth
optimization with robust multi-scale matching. The interval maps remain a
valid search-domain restriction for that future method.
