"""
Generate publication-quality plots for detailed_report.md.
All numbers match the report text exactly.
"""
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
PLOTS = BASE / "plots"
NOTEBOOKS = BASE / "notebooks"
HOLDOUT = BASE / "dataset" / "holdout"

PLOTS.mkdir(exist_ok=True)

# Load dimensionality robustness results
dim_results = json.load(open(NOTEBOOKS / "dimensionality_robustness_results.json"))

# Anonymisation mapping for holdout speakers
ANON_MAP = {
    "Bruce Willis": "HOLDOUT_001", "Gene Wilder": "HOLDOUT_002",
    "Tippi Hedren": "HOLDOUT_003", "Carol Burnett": "HOLDOUT_004",
    "Jane Goodall": "HOLDOUT_005", "Michael Caine": "HOLDOUT_006",
    "Rita Moreno": "HOLDOUT_007", "Willie Nelson": "HOLDOUT_008",
}
EXCLUDED = {"Rita Moreno", "Willie Nelson"}

def anon(name):
    return ANON_MAP.get(name, name)

# Load holdout data
try:
    holdout_tan = json.load(open(HOLDOUT / "holdout_text_acoustic_narrative_results.json"))
    pre_symp = holdout_tan["models"]["text_acoustic_narrative_no_after_symptoms"]
    holdout_speakers = [
        {**s, "speaker_name": anon(s["speaker_name"])}
        for s in pre_symp["speakers"] if s["speaker_name"] not in EXCLUDED
    ]
    print(f"Loaded holdout: {len(holdout_speakers)} speakers")
except Exception as e:
    print(f"Holdout load failed ({e}), using hardcoded values")
    holdout_speakers = [
        {"speaker_name": "HOLDOUT_001", "group": "dementia", "prob": 0.9985},
        {"speaker_name": "HOLDOUT_002", "group": "dementia", "prob": 0.973},
        {"speaker_name": "HOLDOUT_003", "group": "dementia", "prob": 0.964},
        {"speaker_name": "HOLDOUT_005", "group": "control", "prob": 0.052},
        {"speaker_name": "HOLDOUT_004", "group": "control", "prob": 0.500},
        {"speaker_name": "HOLDOUT_006", "group": "control", "prob": 0.614},
    ]

# ── Style ──
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "font.family": "sans-serif",
})
DEM = "#D32F2F"
CON = "#1565C0"
GREEN = "#2E7D32"
AMBER = "#F57F17"
GREY = "#9E9E9E"
PURPLE = "#7B1FA2"

# ============================================================
# Figure 1: Model Architecture Comparison (agentic dataset)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 4))

models = [
    ("MedASR + MedGemma text", 0.619, GREY),
    ("MedGemma vision (spectrograms)", 0.636, GREY),
    ("HEAR acoustic embeddings", 0.726, AMBER),
    ("Image + text multimodal", 0.840, AMBER),
    ("Text-only (Pyannote transcripts)", 0.904, CON),
    ("Text + acoustic narrative (pre-symptoms)", 0.911, GREEN),
]

names = [m[0] for m in models]
aucs = [m[1] for m in models]
colors = [m[2] for m in models]

bars = ax.barh(range(len(models)), aucs, color=colors, height=0.6,
               edgecolor="white", linewidth=0.5)

for i, (bar, auc) in enumerate(zip(bars, aucs)):
    ax.text(auc + 0.006, i, f"{auc:.3f}", va="center", ha="left",
            fontsize=10, fontweight="bold")

# Highlight final model
ax.barh(5, aucs[5], color=GREEN, height=0.6, edgecolor=GREEN,
        linewidth=2, linestyle="-")

ax.set_yticks(range(len(models)))
ax.set_yticklabels(names)
ax.set_xlim(0.5, 1.03)
ax.set_xlabel("AUC (speaker-grouped 5-fold CV)")
ax.set_title("Model Architecture Comparison \u2014 LLM-Assisted Dataset (188 pre-symptom speakers)")
ax.axvline(0.5, color="#BDBDBD", ls="--", lw=0.8)

ax.invert_yaxis()
fig.tight_layout()
fig.savefig(PLOTS / "fig_model_comparison.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("1/7 fig_model_comparison.png")


# ============================================================
# Figure 2: Confound Analysis
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

# Agentic
labels_a = ["Random\nembeddings", "Acoustic\nvariability", "Anti-leak\nimage",
            "Model-accessible\nconfound", "Metadata\nconfound", "Real model"]
vals_a = [0.500, 0.518, 0.576, 0.632, 0.687, 0.911]
cols_a = [GREY, GREY, GREY, AMBER, AMBER, GREEN]

axes[0].bar(range(len(labels_a)), vals_a, color=cols_a, width=0.65, edgecolor="white")
for i, v in enumerate(vals_a):
    axes[0].text(i, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
axes[0].set_xticks(range(len(labels_a)))
axes[0].set_xticklabels(labels_a, fontsize=8.5)
axes[0].set_ylim(0.4, 1.02)
axes[0].set_ylabel("AUC")
axes[0].set_title("LLM-Assisted Dataset")
axes[0].axhline(0.5, color="#BDBDBD", ls="--", lw=0.7)

# Annotate the gap
axes[0].annotate("", xy=(5, 0.911), xytext=(4, 0.687),
                 arrowprops=dict(arrowstyle="<->", color="black", lw=1.2))
axes[0].text(4.5, 0.80, "gap\n+0.224", ha="center", fontsize=9, fontweight="bold")

# DementiaNet
labels_d = ["Model-accessible\nconfound", "Metadata\nconfound", "Real model\n(text, pre-symp)"]
vals_d = [0.775, 0.806, 0.838]
cols_d = [AMBER, AMBER, CON]

axes[1].bar(range(len(labels_d)), vals_d, color=cols_d, width=0.55, edgecolor="white")
for i, v in enumerate(vals_d):
    axes[1].text(i, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
axes[1].set_xticks(range(len(labels_d)))
axes[1].set_xticklabels(labels_d, fontsize=8.5)
axes[1].set_ylim(0.4, 1.02)
axes[1].set_title("DementiaNet (human-curated)")
axes[1].axhline(0.5, color="#BDBDBD", ls="--", lw=0.7)

axes[1].annotate("", xy=(2, 0.838), xytext=(1, 0.806),
                 arrowprops=dict(arrowstyle="<->", color="black", lw=1.2))
axes[1].text(1.5, 0.822, "gap\n+0.032", ha="center", fontsize=9, fontweight="bold")

fig.suptitle("Confound Analysis: Is the Signal in the Speech?", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(PLOTS / "fig_confound_analysis.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("2/7 fig_confound_analysis.png")


# ============================================================
# Figure 3: Permutation Test
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

for i, name in enumerate(["Agentic", "DementiaNet"]):
    ax = axes[i]
    r = dim_results["permutation_test"][name]
    null_aucs = np.array(r["perm_aucs"])
    real_auc = r["real_auc"]

    ax.hist(null_aucs, bins=25, color="#B0BEC5", edgecolor="white", linewidth=0.5,
            alpha=0.9, label="Null distribution (200 shuffles)")
    ax.axvline(real_auc, color=DEM, lw=2.5, label=f"Real AUC = {real_auc:.3f}")
    ax.axvline(r["null_95th"], color=AMBER, lw=1.5, ls="--",
               label=f"95th percentile = {r['null_95th']:.3f}")

    ax.set_xlabel("AUC")
    ax.set_ylabel("Count")
    ax.set_title(f"{name} (p = {r['p_value']:.3f}, effect = +{r['effect_size']:.3f})")
    ax.legend(fontsize=8, loc="upper right")

fig.suptitle("Permutation Test \u2014 200 Label Shuffles", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(PLOTS / "fig_permutation_test.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("3/7 fig_permutation_test.png")


# ============================================================
# Figure 4: PCA Dimensionality Sweep
# ============================================================
fig, ax = plt.subplots(figsize=(8, 4.5))

for name, color, marker in [("Agentic", GREEN, "o"), ("DementiaNet", CON, "s")]:
    pca = dim_results["pca_sweep"][name]
    dims = []
    aucs = []
    for label in ["PCA-16", "PCA-32", "PCA-64", "PCA-128", "PCA-256", "PCA-512", "Full-5120"]:
        if label in pca:
            d = int(label.split("-")[1]) if "PCA" in label else 5120
            dims.append(d)
            aucs.append(pca[label]["auc"])
    ax.plot(dims, aucs, f"-{marker}", color=color, markersize=7, linewidth=2, label=name)

    # Annotate PCA-16
    if name == "Agentic":
        ax.annotate(f"PCA-16: {aucs[0]:.3f}", xy=(dims[0], aucs[0]),
                    xytext=(30, -25), textcoords="offset points", fontsize=9,
                    arrowprops=dict(arrowstyle="->", color=color, lw=0.8), color=color)

ax.set_xscale("log")
ax.set_xticks([16, 32, 64, 128, 256, 512, 5120])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xlabel("Number of dimensions")
ax.set_ylabel("AUC")
ax.set_title("PCA Dimensionality Sweep (Mean+Std features)")
ax.legend()
ax.set_ylim(0.75, 0.95)
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig(PLOTS / "fig_pca_sweep.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("4/7 fig_pca_sweep.png")


# ============================================================
# Figure 5: Learning Curve
# ============================================================
fig, ax = plt.subplots(figsize=(8, 4.5))

for name, color, marker in [("Agentic", GREEN, "o"), ("DementiaNet", CON, "s")]:
    curve = dim_results["learning_curve"][name]
    fracs = []
    aucs = []
    stds = []
    for frac_key in sorted(curve.keys(), key=float):
        v = curve[frac_key]
        if np.isnan(v["mean_auc"]):
            continue
        fracs.append(float(frac_key))
        aucs.append(v["mean_auc"])
        stds.append(v["std_auc"])

    fracs = np.array(fracs)
    aucs = np.array(aucs)
    stds = np.array(stds)

    ax.plot(fracs * 100, aucs, f"-{marker}", color=color, markersize=7, linewidth=2, label=name)
    ax.fill_between(fracs * 100, aucs - stds, aucs + stds, color=color, alpha=0.12)

ax.set_xlabel("Training set size (% of speakers)")
ax.set_ylabel("AUC")
ax.set_title("Learning Curve (10 random subsamples per fraction)")
ax.legend()
ax.set_ylim(0.55, 0.98)
ax.set_xlim(15, 105)
ax.grid(axis="y", alpha=0.3)
ax.axhline(0.5, color="#BDBDBD", ls="--", lw=0.7)

fig.tight_layout()
fig.savefig(PLOTS / "fig_learning_curve.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("5/7 fig_learning_curve.png")


# ============================================================
# Figure 6: Holdout Per-Speaker Scores
# ============================================================
fig, ax = plt.subplots(figsize=(8, 4))

speakers = sorted(holdout_speakers, key=lambda s: s["prob"])
names = [s["speaker_name"] for s in speakers]
probs = [s["prob"] for s in speakers]
colors = [DEM if s["group"] == "dementia" else CON for s in speakers]

bars = ax.barh(range(len(speakers)), probs, color=colors, height=0.55, edgecolor="white")

for i, (prob, spk) in enumerate(zip(probs, speakers)):
    label = f"{prob:.3f}"
    if prob > 0.5:
        ax.text(prob - 0.02, i, label, va="center", ha="right", fontsize=10,
                fontweight="bold", color="white")
    else:
        ax.text(prob + 0.015, i, label, va="center", ha="left", fontsize=10,
                fontweight="bold", color="black")

ax.set_yticks(range(len(speakers)))
ax.set_yticklabels(names, fontsize=11)
ax.set_xlim(0, 1.05)
ax.set_xlabel("Dementia probability")
ax.set_title("Holdout Evaluation \u2014 6 Fully Unseen Celebrity Speakers")
ax.axvline(0.5, color="black", ls="--", lw=1, alpha=0.5)
ax.text(0.51, -0.6, "threshold", fontsize=8, color="grey")

# Legend
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=DEM, label="Dementia (known)"),
                   Patch(color=CON, label="Control (known)")],
          loc="center right", fontsize=9)

fig.tight_layout()
fig.savefig(PLOTS / "fig_holdout_speakers.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("6/7 fig_holdout_speakers.png")


# ============================================================
# Figure 7: Core Metrics (regenerated with fold-mean AUC)
# ============================================================
fig, ax = plt.subplots(figsize=(6, 4))

metric_names = ["AUC", "F1", "Accuracy"]
vals = [0.911, 0.829, 0.851]
bar_colors = ["#5B8FF9", "#61DDAA", "#F6BD16"]

bars = ax.bar(metric_names, vals, color=bar_colors, width=0.55, edgecolor="white")
for i, v in enumerate(vals):
    ax.text(i, v + 0.018, f"{v:.3f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

ax.set_ylim(0.0, 1.0)
ax.set_title("Text + Acoustic Narrative (pre-symptoms)\n188 speakers, 5-fold speaker-grouped CV")
ax.set_ylabel("Score")

fig.tight_layout()
fig.savefig(PLOTS / "auc_f1_acc_pre_symptoms.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("7/7 auc_f1_acc_pre_symptoms.png")


print("\nAll plots saved to:", PLOTS)
