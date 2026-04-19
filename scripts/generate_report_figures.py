"""
generate_report_figures.py

Generates all figures for the VGN adaptive threshold report.
Run from ~/vgn/:
    python scripts/generate_report_figures.py

Output: data/experiments/report_figures/
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pathlib import Path

OUTPUT_DIR = Path("data/experiments/report_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0D1117",
    "axes.facecolor":    "#0D1117",
    "axes.edgecolor":    "#334155",
    "axes.linewidth":    0.8,
    "axes.grid":         True,
    "grid.color":        "#1E293B",
    "grid.linewidth":    0.6,
    "xtick.color":       "#94A3B8",
    "ytick.color":       "#94A3B8",
    "xtick.labelsize":   12,
    "ytick.labelsize":   12,
    "font.family":       "sans-serif",
    "font.size":         12,
    "text.color":        "#E2E8F0",
    "legend.facecolor":  "#1E293B",
    "legend.edgecolor":  "#334155",
    "legend.labelcolor": "#E2E8F0",
})

BLUE   = "#378ADD"
GREEN  = "#1D9E75"
AMBER  = "#EF9F27"
RED    = "#E24B4A"
MUTED  = "#94A3B8"
WHITE  = "#E2E8F0"
BG     = "#0D1117"

# ── data ───────────────────────────────────────────────────────────────────
obj_counts = [3, 5, 7, 10]

static_pile_sr = [86.9, 86.4, 82.8, 81.8]
static_pile_pc = [94.7, 91.0, 86.3, 80.1]
adapt_pile_sr  = [87.6, 85.6, 83.4, 82.4]
adapt_pile_pc  = [94.0, 90.6, 85.6, 80.5]

static_packed_sr = [95.2]
static_packed_pc = [98.8]
adapt_packed_sr  = [93.8]
adapt_packed_pc  = [98.1]

# threshold sweep data
thresholds   = [0.65, 0.75, 0.85, 0.95]
sweep_pile_sr = [76.0, 82.9, 82.9, 92.7]
sweep_pile_pc = [77.5, 87.5, 87.5, 82.0]
sweep_packed_sr = [94.2, 89.5, 93.1, 98.7]
sweep_packed_pc = [100.0, 94.3, 100.0, 93.1]

# ablation
ablation_methods = ["Static\n0.90", "MLP\nonly", "Fallback\nonly", "MLP +\nFallback"]
ablation_pile_sr = [85.6, 87.7, 86.8, 83.9]
ablation_pile_pc = [87.2, 85.8, 91.4, 89.5]


def savefig(name):
    path = OUTPUT_DIR / name
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print("Saved %s" % path)


# ═══════════════════════════════════════════════════════════════════════════
# Fig 1 — SR vs Object Count (main result)
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 4.5))
fig.patch.set_facecolor(BG)

ax.plot(obj_counts, static_pile_sr, "o-", color=BLUE,  lw=2.5, ms=8,
        label="Static τ=0.90", zorder=3)
ax.plot(obj_counts, adapt_pile_sr,  "s-", color=GREEN, lw=2.5, ms=8,
        label="Adaptive (ours)", zorder=3)

# shade region where adaptive wins
for i, (x, s, a) in enumerate(zip(obj_counts, static_pile_sr, adapt_pile_sr)):
    if a >= s:
        ax.axvspan(x-0.3, x+0.3, alpha=0.08, color=GREEN)
        ax.annotate("+%.1f%%" % (a-s), xy=(x, max(a,s)+0.15),
                    ha="center", fontsize=9, color=GREEN, fontweight="bold")

ax.set_xlabel("Number of objects", color=MUTED, fontsize=12)
ax.set_ylabel("Success rate (%)", color=MUTED, fontsize=12)
ax.set_title("Success Rate vs Scene Complexity — Pile Scene", color=WHITE,
             fontsize=13, fontweight="bold", pad=12)
ax.set_xticks(obj_counts)
ax.set_ylim(78, 92)
ax.legend(fontsize=11)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
savefig("fig1_sr_vs_objects.png")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 2 — PC vs Object Count
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 4.5))
fig.patch.set_facecolor(BG)

ax.plot(obj_counts, static_pile_pc, "o-", color=BLUE,  lw=2.5, ms=8,
        label="Static τ=0.90")
ax.plot(obj_counts, adapt_pile_pc,  "s-", color=GREEN, lw=2.5, ms=8,
        label="Adaptive (ours)")

ax.set_xlabel("Number of objects", color=MUTED, fontsize=12)
ax.set_ylabel("% Cleared", color=MUTED, fontsize=12)
ax.set_title("% Cleared vs Scene Complexity — Pile Scene", color=WHITE,
             fontsize=13, fontweight="bold", pad=12)
ax.set_xticks(obj_counts)
ax.set_ylim(75, 100)
ax.legend(fontsize=11)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
savefig("fig2_pc_vs_objects.png")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 3 — SR and PC side by side across object counts
# ═══════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
fig.patch.set_facecolor(BG)

for ax, static_vals, adapt_vals, ylabel, title in [
    (ax1, static_pile_sr, adapt_pile_sr, "Success rate (%)", "Success Rate"),
    (ax2, static_pile_pc, adapt_pile_pc, "% Cleared",        "% Cleared"),
]:
    ax.plot(obj_counts, static_vals, "o-", color=BLUE,  lw=2.5, ms=8, label="Static τ=0.90")
    ax.plot(obj_counts, adapt_vals,  "s-", color=GREEN, lw=2.5, ms=8, label="Adaptive (ours)")
    ax.set_xlabel("Number of objects", color=MUTED, fontsize=12)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=12)
    ax.set_title(title + " — Pile", color=WHITE, fontsize=13, fontweight="bold", pad=10)
    ax.set_xticks(obj_counts)
    ax.legend(fontsize=10)
    ax.spines[["top","right"]].set_visible(False)

ax1.set_ylim(78, 92)
ax2.set_ylim(75, 100)
plt.suptitle("Adaptive vs Static Threshold — Pile Scene", color=WHITE,
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
savefig("fig3_pile_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 4 — Grouped bar chart: static vs adaptive per object count
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(BG)

x = np.arange(len(obj_counts))
w = 0.35
b1 = ax.bar(x - w/2, static_pile_sr, w, color=BLUE,  alpha=0.85, label="Static τ=0.90")
b2 = ax.bar(x + w/2, adapt_pile_sr,  w, color=GREEN, alpha=0.85, label="Adaptive (ours)")

for bar in list(b1) + list(b2):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
            "%.1f" % bar.get_height(), ha="center", va="bottom",
            fontsize=9, color=WHITE)

ax.set_xticks(x)
ax.set_xticklabels(["%d objects" % o for o in obj_counts])
ax.set_ylabel("Success rate (%)", color=MUTED, fontsize=12)
ax.set_title("Success Rate: Static vs Adaptive — Pile Scene",
             color=WHITE, fontsize=13, fontweight="bold", pad=12)
ax.set_ylim(78, 92)
ax.legend(fontsize=11)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
savefig("fig4_bar_sr_pile.png")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 5 — Threshold sweep: tradeoff curve
# ═══════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
fig.patch.set_facecolor(BG)

for ax, sr, pc, title in [
    (ax1, sweep_pile_sr,   sweep_pile_pc,   "Pile scene"),
    (ax2, sweep_packed_sr, sweep_packed_pc, "Packed scene"),
]:
    ax.plot(thresholds, sr, "o-", color=BLUE,  lw=2.5, ms=8, label="Success rate %")
    ax.plot(thresholds, pc, "s--", color=GREEN, lw=2.5, ms=8, label="% cleared")
    ax.axvline(x=0.90, color=AMBER, lw=1.5, linestyle=":", label="Default τ=0.90")
    ax.set_xlabel("Confidence threshold τ", color=MUTED, fontsize=12)
    ax.set_ylabel("%", color=MUTED, fontsize=12)
    ax.set_title(title, color=WHITE, fontsize=13, fontweight="bold", pad=10)
    ax.set_xticks(thresholds)
    ax.set_ylim(60, 108)
    ax.legend(fontsize=10)
    ax.spines[["top","right"]].set_visible(False)

plt.suptitle("Threshold Sweep: SR vs PC Tradeoff", color=WHITE,
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
savefig("fig5_threshold_sweep.png")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 6 — Ablation study
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor(BG)

x = np.arange(len(ablation_methods))
w = 0.35
b1 = ax.bar(x - w/2, ablation_pile_sr, w, color=BLUE,  alpha=0.85, label="Success rate %")
b2 = ax.bar(x + w/2, ablation_pile_pc, w, color=GREEN, alpha=0.85, label="% Cleared")

for bar in list(b1) + list(b2):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
            "%.1f" % bar.get_height(), ha="center", va="bottom",
            fontsize=9, color=WHITE)

# baseline reference lines
ax.axhline(y=85.6, color=BLUE,  lw=1, linestyle="--", alpha=0.4)
ax.axhline(y=87.2, color=GREEN, lw=1, linestyle="--", alpha=0.4)

ax.set_xticks(x)
ax.set_xticklabels(ablation_methods, fontsize=11)
ax.set_ylabel("%", color=MUTED, fontsize=12)
ax.set_title("Ablation Study — Pile Scene (5 objects)",
             color=WHITE, fontsize=13, fontweight="bold", pad=12)
ax.set_ylim(78, 96)
ax.legend(fontsize=11)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
savefig("fig6_ablation.png")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 7 — Pile vs Packed comparison (adaptive vs static)
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.patch.set_facecolor(BG)

# pile
ax = axes[0]
x = np.arange(2)
w = 0.3
ax.bar(x[0]-w/2, 86.4, w, color=BLUE,  alpha=0.85, label="Static τ=0.90")
ax.bar(x[0]+w/2, 85.6, w, color=GREEN, alpha=0.85, label="Adaptive")
ax.bar(x[1]-w/2, 91.0, w, color=BLUE,  alpha=0.85)
ax.bar(x[1]+w/2, 90.6, w, color=GREEN, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(["Success rate %", "% Cleared"], fontsize=11)
ax.set_title("Pile scene (5 objects)", color=WHITE, fontsize=13,
             fontweight="bold", pad=10)
ax.set_ylim(83, 95)
ax.legend(fontsize=10)
ax.spines[["top","right"]].set_visible(False)
for val, xpos in [(86.4, -0.15), (85.6, 0.15), (91.0, 0.85), (90.6, 1.15)]:
    ax.text(xpos, val+0.15, "%.1f" % val, ha="center", va="bottom",
            fontsize=9, color=WHITE)

# packed
ax = axes[1]
ax.bar(x[0]-w/2, 95.2, w, color=BLUE,  alpha=0.85, label="Static τ=0.90")
ax.bar(x[0]+w/2, 93.8, w, color=GREEN, alpha=0.85, label="Adaptive")
ax.bar(x[1]-w/2, 98.8, w, color=BLUE,  alpha=0.85)
ax.bar(x[1]+w/2, 98.1, w, color=GREEN, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(["Success rate %", "% Cleared"], fontsize=11)
ax.set_title("Packed scene (5 objects)", color=WHITE, fontsize=13,
             fontweight="bold", pad=10)
ax.set_ylim(90, 102)
ax.legend(fontsize=10)
ax.spines[["top","right"]].set_visible(False)
for val, xpos in [(95.2, -0.15), (93.8, 0.15), (98.8, 0.85), (98.1, 1.15)]:
    ax.text(xpos, val+0.1, "%.1f" % val, ha="center", va="bottom",
            fontsize=9, color=WHITE)

plt.suptitle("Static vs Adaptive — Pile and Packed Scenes",
             color=WHITE, fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
savefig("fig7_pile_vs_packed.png")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 8 — Delta improvement heatmap
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 4))
fig.patch.set_facecolor(BG)

delta_sr = [a-s for a,s in zip(adapt_pile_sr, static_pile_sr)]
delta_pc = [a-s for a,s in zip(adapt_pile_pc, static_pile_pc)]

x = np.arange(len(obj_counts))
w = 0.35
bars1 = ax.bar(x - w/2, delta_sr, w, color=[GREEN if d>=0 else RED for d in delta_sr],
               alpha=0.85, label="SR delta")
bars2 = ax.bar(x + w/2, delta_pc, w, color=[GREEN if d>=0 else RED for d in delta_pc],
               alpha=0.85, label="PC delta")

for bar in list(bars1) + list(bars2):
    val = bar.get_height()
    ypos = val + 0.05 if val >= 0 else val - 0.25
    ax.text(bar.get_x()+bar.get_width()/2, ypos,
            "%+.1f" % val, ha="center", va="bottom", fontsize=9, color=WHITE)

ax.axhline(y=0, color=MUTED, lw=1, linestyle="-")
ax.set_xticks(x)
ax.set_xticklabels(["%d objects" % o for o in obj_counts])
ax.set_ylabel("Δ (Adaptive − Static)", color=MUTED, fontsize=12)
ax.set_title("Improvement of Adaptive over Static — Pile Scene",
             color=WHITE, fontsize=13, fontweight="bold", pad=12)
ax.legend(fontsize=11)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
savefig("fig8_delta_improvement.png")


print("\nAll figures saved to %s" % OUTPUT_DIR)
print("Files:")
for f in sorted(OUTPUT_DIR.glob("*.png")):
    print("  %s" % f.name)
