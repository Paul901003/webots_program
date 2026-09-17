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

The fixed-cohort geometry and downstream Stage 2 separation gate is complete;
the final decision is recorded below. Promotion requires lower clean leak without
reducing correct assignment or introducing a merged main pair.

## Footprint Input

The same guarded second-best rule was run from the wider Stage 1 footprint hull.
Metrics use that footprint hull's own pre-photo visible surface; they are not
directly comparable by raw voxel count with the center-point hull above.

| Variant | Remaining voxels | Removed voxels | Sugar visible-surface removed | GT coverage | Ghost ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Footprint no photo | 15211 | 0 | 0 / 1304 | 0.9844 | 0.2178 |
| Footprint guarded mean | 14288 | 923 | 374 / 1304 | 0.9729 | 0.1770 |
| Footprint original mean | 14206 | 1005 | 409 / 1304 | 0.9650 | 0.1789 |
| Footprint guarded second | 14920 | 291 | 176 / 1304 | 0.9760 | 0.2093 |

The footprint variant begins with more true coverage and more ghost volume. Its
guarded second-best photo pass has much less sugar loss than its original mean
pass. Under the same foreground-patch guard, mean removes 28.7 percent of the
sugar surface while second removes 13.5 percent. The improvement is therefore
from the source-evidence aggregation rule rather than the footprint input alone.

## Fixed Stack Cohort

The fixed cohort is the 60 legacy stack scenes: `stack3_scene0001` through
`stack5_scene0020`. Both candidates use the same 12 views, donut-mask CLIP
semantics, center-point voting, connected-or-drop reassignment, and the clean
GT-surface leak metric. The only variable in each pair is the Stage 1 photo pass.

| Stage 1 root | Mesh missed | Ghost | Mesh IoU |
| --- | ---: | ---: | ---: |
| `am1` | 8.7% | 20.8% | 0.737 |
| `am1` + guarded second | 9.2% | 20.3% | 0.738 |
| `am1_fp` | 1.4% | 26.6% | 0.726 |
| `am1_fp` + guarded second | 1.8% | 25.9% | 0.731 |

| Hull and photo variant | Correct | Leak | Unassigned | Fragment | Main pair merged |
| --- | ---: | ---: | ---: | ---: | ---: |
| `am1` without photo | 78.47% | 4.29% | 17.24% | 15.23% | 0.0% |
| `am1` + guarded second | 76.43% | 4.98% | 18.59% | 15.76% | 0.0% |
| `am1_fp` without photo | 79.54% | 4.82% | 15.64% | 16.54% | 0.0% |
| `am1_fp` + guarded second | 77.47% | 4.91% | 17.62% | 16.34% | 3.4% |

## Decision

Do not promote either guarded-second output. The rule substantially limits the
single-scene sugar-box deletion, but on the fixed cohort it still removes enough
useful surface to reduce correct assignment and increase unassigned voxels. The
footprint candidate also introduces a merged main pair. Keep the current Stage 1
and Stage 2 baseline: `srp_hull_mv2_v12_am1` followed by
`srp_hull_semcluster_reNNcSd_am1`.

The guarded photo code remains opt-in on this experiment branch. A future photo
experiment must improve clean leak without reducing correct assignment, rather
than relying on the Stage 1 mesh volume metrics alone.

## DivB Follow-up

`merge_div_guard.py` was run with the existing `divB` span rule and `theta=0.5`
on the same 60-scene cohort. The inputs remain the guarded-second photo hulls.

| Hull and merge variant | Correct | Leak | Unassigned | Fragment | Main pair merged |
| --- | ---: | ---: | ---: | ---: | ---: |
| `am1` photo before div | 76.43% | 4.98% | 18.59% | 15.76% | 0.0% |
| `am1` photo plus divB | 74.03% | 7.38% | 18.59% | 2.19% | 13.8% |
| `am1_fp` photo before div | 77.47% | 4.91% | 17.62% | 16.34% | 3.4% |
| `am1_fp` photo plus divB | 72.50% | 9.88% | 17.62% | 1.25% | 20.7% |

Footprint photo plus divB is rejected. Center-photo divB has a lower aggregate
leak than mainline divB and is assessed by the direct shared-voxel flow below.
It remains an investigation candidate, not promoted: it retains four merged main
pairs and creates one newly merged pair.

## Direct Center-Div Flow

See [DIVB_PAIR_FLOW.md](DIVB_PAIR_FLOW.md) for all 29 on relations. It follows
only mainline-div leak voxels at common world coordinates. Of 2,405 such voxels,
13.9 percent become correct under center-photo div, only 0.3 percent become
unassigned, 75.1 percent remain leak, and 10.6 percent are no longer the same
GT-labelled surface after photo carving. Main-pair merging remains 4/29, with one
newly merged and one newly separated relation.
