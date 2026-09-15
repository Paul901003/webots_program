#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
S = "/tmp/claude-1000/-home-cho-webots-program/338e7fee-7000-4d61-a82a-92a2c5d8b46c/scratchpad"
z = np.load(S + "/mask_nn_scores.npz")
G_s, G_t, S_s, S_t = z["G_s"], z["G_t"], z["S_s"], z["S_t"]
plt.rcParams["font.family"] = "DejaVu Sans"
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
bins = np.linspace(0, 1, 41)
for a, (same, touch, title) in zip(ax, [(G_s, G_t, "nnratio (2.5cm geometric adjacency)"),
                                        (S_s, S_t, "CLIP cosine (semantic)")]):
    a.hist(same, bins=bins, density=True, alpha=0.55, color="#2c7fb8", label=f"same object (n={len(same)})")
    a.hist(touch, bins=bins, density=True, alpha=0.55, color="#d95f0e", label=f"touching different (n={len(touch)})")
    a.axvline(np.median(same), color="#2c7fb8", ls="--", lw=1.5)
    a.axvline(np.median(touch), color="#d95f0e", ls="--", lw=1.5)
    a.set_title(title); a.set_xlabel("score"); a.set_ylabel("density"); a.legend()
    a.text(0.02, 0.97, f"median same={np.median(same):.3f}\nmedian touch={np.median(touch):.3f}",
           transform=a.transAxes, va="top", fontsize=9, bbox=dict(fc="white", alpha=0.7))
fig.suptitle("Mask-level nnratio & CLIP distribution — 60 stacked scenes (am0 + photo-carved hull)", fontsize=12)
fig.tight_layout()
fig.savefig(S + "/mask_nn_dist.png", dpi=120)
print("saved mask_nn_dist.png")
# 印分位數表
def q(x): return np.percentile(x, [5, 25, 50, 75, 95])
print("nnratio 同物體 5/25/50/75/95:", np.round(q(G_s), 3))
print("nnratio 相觸異物 5/25/50/75/95:", np.round(q(G_t), 3))
print("CLIP 同物體:", np.round(q(S_s), 3))
print("CLIP 相觸異物:", np.round(q(S_t), 3))
