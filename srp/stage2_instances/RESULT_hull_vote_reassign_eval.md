> # ⚠⚠ 作廢(2026-09-16,使用者指正)⚠⚠
> **本表用錯指標。** found@/mIoU/s@ 不是最終定案指標,且各線用自己 hull 當宇宙、跨胖瘦不可比。
> **最終定案指標 = 重投影回 GT 遮罩,判「堆疊有沒有分開 / 混太嚴重」,並分 n / occ / stack / all。**
> 本檔僅保留當作「錯誤示範 / 曾用過的數字」,**不可據此比較或宣稱最佳**。正解見重投影版(reproj_labels 系列),待重建落地。

# 胖瘦 hull × 中心/fp 投票 × 三種 reassign — div 評估結果(落地存檔)

- 建檔:2026-09-16。**來源=記憶 `footprint-vote-beats-center.md` 的 #1~#8 表(2026-09-14)**;此檔把它攤成完整 2×2×3 格並標記從缺項,避免再只存記憶。
- **provenance / 四問稽核**:
  - 產表 eval:`eval_solid_4lines.py` / `eval_1.py` / `eval_23.py`(**⚠ 這三支已消失,當時在 job tmp**);堆疊 `d2_stack_clean.py`(29 on 對,τ 版)。
  - 分母=**各線自己 hull 的可見表面**當宇宙;基準=**mesh-3D 真值**(solid_mesh_occ);配對=Hungarian;**排除 GLOBAL_EXCLUDE(skillet_lid/windex/colored_wood_blocks/dice)**;母體=**303 多物場**(n3/n4/n5/occ/stack)。
  - 量的:found@0.5、found@0.7、mIoU(過切軸);堆疊 s@0.5/s@0.7(乾淨分開,含 τ)。
- 全 div root 前綴 `srp_hull_divB_t50_`;hull=`srp_hull_mv2_v12_am1`(瘦)/`srp_hull_mv2_v12_am1_fp`(胖)。reassign 腳本 `voxel_sem_cluster_reassign_solid{vis=併最近,drop,erode=侵蝕}.py`(`VOTE=center|footprint`)。

## 完整 2×2×3 格(12 組;有數字=已評估記錄,缺=從未記錄)

| hull | 投票 | reassign | div root(前綴 srp_hull_divB_t50_) | f@0.5 | f@0.7 | mIoU | s@0.5 | s@0.7 | 來源# |
|---|---|---|---|---|---|---|---|---|---|
| 瘦 am1 | 中心 | 併最近 | reNNcS_am1 | — | — | — | — | — | 缺(僅有薄殼bug版 reNN_guard:0.923/0.713/0.718、0.76/0.72=#1,物理錯不採) |
| 瘦 am1 | 中心 | drop | reNNcSd_am1 | 0.920 | 0.528 | 0.679 | 0.79 | 0.69 | #6 |
| 瘦 am1 | 中心 | 侵蝕 | reNNcSe_am1 | — | — | — | — | — | 缺 |
| 瘦 am1 | fp | 併最近 | reNNfpS_am1 | — | — | — | — | — | 缺(實心版) |
| 瘦 am1 | fp | drop | reNNfpSd_am1 | 0.970 | 0.901 | 0.870 | 0.83 | 0.76 | #5 |
| 瘦 am1 | fp | 侵蝕 | reNNfpSe_am1 | 0.968 | 0.901 | 0.869 | 0.83 | 0.76 | #8 |
| 胖 am1fp | 中心 | 併最近 | reNNcS_am1fp | 0.827 | 0.179 | 0.596 | 0.79 | 0.69 | #2 |
| 胖 am1fp | 中心 | drop | reNNcSd_am1fp | — | — | — | — | — | 缺 |
| 胖 am1fp | 中心 | 侵蝕 | reNNcSe_am1fp | — | — | — | — | — | 缺 |
| 胖 am1fp | fp | 併最近 | reNNfpS_am1fp | 0.921 | 0.783 | 0.754 | 0.79 | 0.66 | #3 |
| 胖 am1fp | fp | drop | reNNfpSd_am1fp | 0.924 | 0.780 | 0.752 | 0.83 | 0.66 | #4 |
| 胖 am1fp | fp | 侵蝕 | reNNfpSe_am1fp | 0.924 | 0.779 | 0.752 | 0.83 | 0.66 | #8 |

- **已記錄 7/12**(#2,#3,#4,#5,#6,#8fp,#8am1);**從缺 5/12**:瘦中心併最近(實心)、瘦中心侵蝕、瘦fp併最近(實心)、胖中心drop、胖中心侵蝕。
- 這 5 組的 div root **本身都存在** `data/eval/`,只是**沒有記錄過 found/mIoU 數字**,且 eval 腳本已失 → 要補得先重建 eval 腳本再跑。
- watershed 版(reNNfpSw_am1fp)已刪(過切細長物、淨負)。侵蝕(Se)實測逐格 = drop,故 Se≈Sd。

## ⚠ 重要:這張表的效力(使用者 2026-09-14 定案)
> **這些指標(found@ / mIoU / 堆疊 s@)『無法顯現哪個方法真正最好』,不可據此宣稱最佳;「#5 最佳」結論作廢。**
> 佐證:①含 zbuffer 薄殼 bug 的 #1 仍拿到 mIoU 0.718 看似不差 → 指標沒罰物理錯誤;②#5 仍有大量「相異 mesh 歸同一 instance」的過併(in-GT voxel 歸錯 4.4%≈27,093 vox,集中相觸/堆疊場),被 303 場平均稀釋、mIoU 沒反映。
> → 需要「per-object 洩漏% / 沒分開率」這類能反映過併/純度的指標(見 [[stacked-merge-eval-perobj-leak]]);該指標的腳本目前也未落地(job tmp 已失),待重建。

相關記憶:footprint-vote-beats-center、stacked-merge-eval-perobj-leak、warpcarve-no-help-stacked、audit-every-metric。
