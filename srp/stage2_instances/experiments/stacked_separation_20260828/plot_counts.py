#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
S = "/tmp/claude-1000/-home-cho-webots-program/338e7fee-7000-4d61-a82a-92a2c5d8b46c/scratchpad"
z = np.load(S + "/mask_nn_scores.npz"); s_all, t_all = z["G_s"], z["G_t"]
zc = np.load(S + "/mask_nn_cand.npz"); s_c, t_c = zc["s"], zc["t"]
bins = np.linspace(0, 1, 21)          # 20 格,每格 0.05
ctr = (bins[:-1] + bins[1:]) / 2
w = 0.02
fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
for a, (s, t, title) in zip(ax, [(s_all, t_all, "All mask pairs"),
                                  (s_c, t_c, "Candidate pairs only (nn>0, adjacent)")]):
    hs, _ = np.histogram(s, bins=bins); ht, _ = np.histogram(t, bins=bins)
    a.bar(ctr - w/2, hs, width=w, color="#2c7fb8", label=f"same object (n={len(s):,})")
    a.bar(ctr + w/2, ht, width=w, color="#d95f0e", label=f"touching different (n={len(t):,})")
    a.set_title(title); a.set_xlabel("nnratio score"); a.set_ylabel("number of pairs (count)")
    a.legend(); a.set_xticks(np.arange(0, 1.01, 0.1))
fig.suptitle("nnratio histogram (counts) — 60 stacked scenes, am0+photo-carved hull", fontsize=13)
fig.tight_layout()
fig.savefig(S + "/mask_nn_counts.png", dpi=120)
print("saved mask_nn_counts.png")
# 也印候選對每格數量表
print("\n候選對 每 0.1 區間 數量:")
b2 = np.linspace(0, 1, 11)
hs, _ = np.histogram(s_c, bins=b2); ht, _ = np.histogram(t_c, bins=b2)
for i in range(10):
    print(f"  [{b2[i]:.1f},{b2[i+1]:.1f}): 同物體 {hs[i]:6d}   相觸異物 {ht[i]:6d}")
