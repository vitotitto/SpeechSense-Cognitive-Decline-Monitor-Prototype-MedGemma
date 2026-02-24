"""
Dimensionality Robustness Analysis
===================================
Tests whether the model's performance is driven by genuine signal or
high-dimensional overfitting.

Experiments:
  1. Mean-only vs mean+std ablation (2560 vs 5120 dims)
  2. PCA sweep (16, 32, 64, 128, 256, 512 components)
  3. Regularisation C sweep (0.001 to 100)
  4. Learning curve (subsample speakers)
  5. Permutation test on real model (200 shuffles)
  6. PCA sweep on mean-only (2560 dims)

Run on both Agentic (pre-symptoms) and DementiaNet (pre-symptoms) datasets.
Uses lbfgs solver (converges properly on 5120-dim data).
"""
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.base import clone

RANDOM_STATE = 42
N_SPLITS = 5
np.random.seed(RANDOM_STATE)

# ── Paths (relative to repo root) ──
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENTIC_NPZ = os.path.join(BASE, "reproducibility", "multimodal", "outputs_text_acoustic_narrative_pre_symptoms", "multimodal_embeddings_agentic_manifest_text_acoustic_narrative.npz")
AGENTIC_MANIFEST = os.path.join(BASE, "dataset", "manifest.csv")

# DementiaNet paths (external dataset, not included in this repository)
DNET_EMB = os.environ.get("DNET_EMB_PATH", "dementianet_embeddings.pkl")
DNET_MANIFEST = os.environ.get("DNET_MANIFEST_PATH", "manifest_dementianet_clean.csv")

OUTPUT_PATH = os.path.join(BASE, "notebooks", "dimensionality_robustness_results.json")


def fprint(*args, **kwargs):
    """Print and flush immediately."""
    print(*args, **kwargs)
    sys.stdout.flush()


# ══════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════

def make_clf(C=1.0):
    """Default classifier: L2 LogReg with lbfgs (fast, converges properly)."""
    return LogisticRegression(
        C=C, class_weight="balanced", solver="lbfgs",
        penalty="l2", max_iter=5000, random_state=RANDOM_STATE
    )


def aggregate_speaker_mean_std(clip_embs, clip_labels, clip_speakers):
    """Speaker-level [mean, std] aggregation -> 2*D dims."""
    spk_data = defaultdict(lambda: {"embs": [], "label": None})
    for i in range(len(clip_speakers)):
        spk_data[clip_speakers[i]]["embs"].append(clip_embs[i])
        spk_data[clip_speakers[i]]["label"] = clip_labels[i]

    X_list, y_list, g_list = [], [], []
    for spk in sorted(spk_data.keys()):
        info = spk_data[spk]
        embs = np.stack(info["embs"], axis=0)
        mean = np.mean(embs, axis=0)
        std = np.std(embs, axis=0) if embs.shape[0] > 1 else np.zeros_like(mean)
        X_list.append(np.concatenate([mean, std]))
        y_list.append(info["label"])
        g_list.append(spk)

    return np.array(X_list, dtype=np.float32), np.array(y_list), np.array(g_list)


def aggregate_speaker_mean_only(clip_embs, clip_labels, clip_speakers):
    """Speaker-level mean-only aggregation -> D dims."""
    spk_data = defaultdict(lambda: {"embs": [], "label": None})
    for i in range(len(clip_speakers)):
        spk_data[clip_speakers[i]]["embs"].append(clip_embs[i])
        spk_data[clip_speakers[i]]["label"] = clip_labels[i]

    X_list, y_list, g_list = [], [], []
    for spk in sorted(spk_data.keys()):
        info = spk_data[spk]
        embs = np.stack(info["embs"], axis=0)
        mean = np.mean(embs, axis=0)
        X_list.append(mean)
        y_list.append(info["label"])
        g_list.append(spk)

    return np.array(X_list, dtype=np.float32), np.array(y_list), np.array(g_list)


def run_cv_simple(X, y, groups, clf_template=None, pca_dims=None):
    """Run GroupKFold CV, return mean AUC and fold AUCs."""
    if clf_template is None:
        clf_template = make_clf()
    n_splits = min(N_SPLITS, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    fold_aucs = []

    for tr_idx, te_idx in gkf.split(X, y, groups):
        Xtr, Xte = X[tr_idx].copy(), X[te_idx].copy()
        ytr, yte = y[tr_idx], y[te_idx]

        if pca_dims and pca_dims < Xtr.shape[1]:
            actual_dims = min(pca_dims, Xtr.shape[0] - 1)
            if actual_dims < 2:
                fold_aucs.append(0.5)
                continue
            pca = PCA(n_components=actual_dims, random_state=RANDOM_STATE)
            Xtr = pca.fit_transform(Xtr)
            Xte = pca.transform(Xte)

        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xte = sc.transform(Xte)
        Xtr = np.nan_to_num(Xtr, nan=0.0, posinf=0.0, neginf=0.0)
        Xte = np.nan_to_num(Xte, nan=0.0, posinf=0.0, neginf=0.0)

        clf = clone(clf_template)
        clf.fit(Xtr, ytr)
        p = clf.predict_proba(Xte)[:, 1]
        try:
            fold_aucs.append(roc_auc_score(yte, p))
        except ValueError:
            fold_aucs.append(0.5)

    return float(np.mean(fold_aucs)), [float(a) for a in fold_aucs]


# ══════════════════════════════════════════════════════════════
# Load datasets
# ══════════════════════════════════════════════════════════════
fprint("Loading datasets...")

# ── Agentic ──
data_ag = np.load(AGENTIC_NPZ, allow_pickle=True)
tan = data_ag["conditions"].item()["text_plus_acoustic_narrative"]
manifest_ag = pd.read_csv(AGENTIC_MANIFEST)

# Build manifest lookup
ag_map = {}
for _, row in manifest_ag.iterrows():
    stem = os.path.splitext(str(row["anon_filename"]))[0]
    ag_map[stem] = {
        "speaker": str(row["anon_speaker_id"]),
        "label": int(row["label"]),
    }

# Clip-level (exclude after_symptoms)
ag_clips_emb, ag_clips_labels, ag_clips_speakers = [], [], []
for clip_id in sorted(tan.keys()):
    if "_after_symptoms_" in clip_id.lower():
        continue
    info = ag_map.get(clip_id)
    if info is None:
        continue
    ag_clips_emb.append(np.asarray(tan[clip_id], dtype=np.float32))
    ag_clips_labels.append(info["label"])
    ag_clips_speakers.append(info["speaker"])

ag_clips_emb = np.array(ag_clips_emb)
ag_clips_labels = np.array(ag_clips_labels)
ag_clips_speakers = np.array(ag_clips_speakers)

# Speaker-level aggregation
X_ag_ms, y_ag, g_ag = aggregate_speaker_mean_std(ag_clips_emb, ag_clips_labels, ag_clips_speakers)
X_ag_m, _, _ = aggregate_speaker_mean_only(ag_clips_emb, ag_clips_labels, ag_clips_speakers)

n_ag = len(y_ag)
n_ag_dem = int(np.sum(y_ag == 1))
n_ag_con = int(np.sum(y_ag == 0))
fprint(f"Agentic (pre-symp): {n_ag} speakers ({n_ag_dem} dem, {n_ag_con} con), "
      f"{len(ag_clips_emb)} clips, mean+std={X_ag_ms.shape[1]} dims, mean-only={X_ag_m.shape[1]} dims")

# ── DementiaNet ──
with open(DNET_EMB, "rb") as f:
    mg_data = pickle.load(f)

mg_pre = {k: v for k, v in mg_data.items() if v["timepoint"] != "after_symptoms"}

dn_clips_emb = np.array([v["embedding"] for v in mg_pre.values()], dtype=np.float32)
dn_clips_labels = np.array([v["label"] for v in mg_pre.values()])
dn_clips_speakers = np.array([v["person"] for v in mg_pre.values()])

X_dn_ms, y_dn, g_dn = aggregate_speaker_mean_std(dn_clips_emb, dn_clips_labels, dn_clips_speakers)
X_dn_m, _, _ = aggregate_speaker_mean_only(dn_clips_emb, dn_clips_labels, dn_clips_speakers)

n_dn = len(y_dn)
n_dn_dem = int(np.sum(y_dn == 1))
n_dn_con = int(np.sum(y_dn == 0))
fprint(f"DementiaNet (pre-symp): {n_dn} speakers ({n_dn_dem} dem, {n_dn_con} con), "
      f"{len(dn_clips_emb)} clips, mean+std={X_dn_ms.shape[1]} dims, mean-only={X_dn_m.shape[1]} dims")


all_results = {}

# ══════════════════════════════════════════════════════════════
# EXPERIMENT 1: Mean-only vs Mean+Std ablation
# ══════════════════════════════════════════════════════════════
fprint(f"\n{'='*70}")
fprint("EXPERIMENT 1: Mean-only (2560) vs Mean+Std (5120)")
fprint(f"{'='*70}")

ablation_results = {}
for name, X_ms, X_m, y, g in [
    ("Agentic", X_ag_ms, X_ag_m, y_ag, g_ag),
    ("DementiaNet", X_dn_ms, X_dn_m, y_dn, g_dn),
]:
    fprint(f"\n  {name}:")
    auc_ms, folds_ms = run_cv_simple(X_ms, y, g)
    auc_m, folds_m = run_cv_simple(X_m, y, g)

    # Also with PCA-128
    auc_ms_pca, _ = run_cv_simple(X_ms, y, g, pca_dims=128)
    auc_m_pca, _ = run_cv_simple(X_m, y, g, pca_dims=128)

    fprint(f"    Mean+Std (5120 dims):     AUC = {auc_ms:.4f}  ratio = {X_ms.shape[1]}/{len(y)} = {X_ms.shape[1]/len(y):.1f}:1")
    fprint(f"    Mean-only (2560 dims):    AUC = {auc_m:.4f}  ratio = {X_m.shape[1]}/{len(y)} = {X_m.shape[1]/len(y):.1f}:1")
    fprint(f"    Mean+Std + PCA-128:       AUC = {auc_ms_pca:.4f}  ratio = 128/{len(y)} = {128/len(y):.2f}:1")
    fprint(f"    Mean-only + PCA-128:      AUC = {auc_m_pca:.4f}  ratio = 128/{len(y)} = {128/len(y):.2f}:1")
    fprint(f"    Delta (std adds):         {auc_ms - auc_m:+.4f}")
    fprint(f"    Delta (std adds, PCA):    {auc_ms_pca - auc_m_pca:+.4f}")

    ablation_results[name] = {
        "mean_std_full": {"auc": auc_ms, "dims": int(X_ms.shape[1]), "fold_aucs": folds_ms},
        "mean_only_full": {"auc": auc_m, "dims": int(X_m.shape[1]), "fold_aucs": folds_m},
        "mean_std_pca128": {"auc": auc_ms_pca, "dims": 128},
        "mean_only_pca128": {"auc": auc_m_pca, "dims": 128},
        "std_delta_full": auc_ms - auc_m,
        "std_delta_pca128": auc_ms_pca - auc_m_pca,
        "n_speakers": int(len(y)),
    }

all_results["mean_vs_mean_std"] = ablation_results


# ══════════════════════════════════════════════════════════════
# EXPERIMENT 2: PCA sweep (fine-grained)
# ══════════════════════════════════════════════════════════════
fprint(f"\n{'='*70}")
fprint("EXPERIMENT 2: PCA Sweep (16 to Full)")
fprint(f"{'='*70}")

pca_dims_list = [16, 32, 64, 128, 256, 512, None]  # None = full

pca_results = {}
for name, X, y, g in [
    ("Agentic", X_ag_ms, y_ag, g_ag),
    ("DementiaNet", X_dn_ms, y_dn, g_dn),
]:
    fprint(f"\n  {name} (mean+std, {X.shape[1]} dims, {len(y)} speakers):")
    sweep = {}
    for pca_d in pca_dims_list:
        label = f"PCA-{pca_d}" if pca_d else f"Full-{X.shape[1]}"
        effective_dims = pca_d if pca_d else X.shape[1]
        auc, folds = run_cv_simple(X, y, g, pca_dims=pca_d)
        ratio = effective_dims / len(y)
        fprint(f"    {label:12s}: AUC = {auc:.4f}  (dim:speaker = {effective_dims}:{len(y)} = {ratio:.1f}:1)  folds={[f'{a:.3f}' for a in folds]}")
        sweep[label] = {"auc": auc, "dims": effective_dims, "ratio": round(ratio, 2), "fold_aucs": folds}
    pca_results[name] = sweep

all_results["pca_sweep"] = pca_results


# ══════════════════════════════════════════════════════════════
# EXPERIMENT 3: Regularisation C sweep
# ══════════════════════════════════════════════════════════════
fprint(f"\n{'='*70}")
fprint("EXPERIMENT 3: Regularisation C Sweep")
fprint(f"{'='*70}")

C_values = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0]

c_results = {}
for name, X, y, g in [
    ("Agentic", X_ag_ms, y_ag, g_ag),
    ("DementiaNet", X_dn_ms, y_dn, g_dn),
]:
    fprint(f"\n  {name} (mean+std, {X.shape[1]} dims):")
    sweep = {}
    for C in C_values:
        clf = make_clf(C=C)
        auc, folds = run_cv_simple(X, y, g, clf_template=clf)
        fprint(f"    C={C:<8.3f}: AUC = {auc:.4f}")
        sweep[f"C={C}"] = {"auc": auc, "C": C, "fold_aucs": folds}
    c_results[name] = sweep

all_results["c_sweep"] = c_results


# ══════════════════════════════════════════════════════════════
# EXPERIMENT 4: Learning Curve (subsample speakers)
# ══════════════════════════════════════════════════════════════
fprint(f"\n{'='*70}")
fprint("EXPERIMENT 4: Learning Curve (subsample speakers)")
fprint(f"{'='*70}")

fractions = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
N_REPEATS = 10  # repeat subsampling for variance

learning_results = {}
for name, X, y, g in [
    ("Agentic", X_ag_ms, y_ag, g_ag),
    ("DementiaNet", X_dn_ms, y_dn, g_dn),
]:
    fprint(f"\n  {name} ({len(y)} speakers total):")
    curve = {}
    unique_speakers = np.unique(g)
    n_total = len(unique_speakers)

    for frac in fractions:
        n_use = max(10, int(n_total * frac))
        if n_use >= n_total:
            # Full dataset, single run
            auc, folds = run_cv_simple(X, y, g)
            curve[f"{frac:.1f}"] = {
                "n_speakers": int(n_total), "fraction": frac,
                "mean_auc": auc, "std_auc": 0.0,
                "all_aucs": [auc],
            }
            fprint(f"    {frac:.0%} ({n_total:3d} speakers): AUC = {auc:.4f}")
            continue

        repeat_aucs = []
        for rep in range(N_REPEATS):
            rng = np.random.RandomState(RANDOM_STATE + rep)
            # Stratified speaker sampling: keep class balance
            dem_spk = unique_speakers[y[np.array([np.where(g == s)[0][0] for s in unique_speakers])] == 1]
            con_spk = unique_speakers[y[np.array([np.where(g == s)[0][0] for s in unique_speakers])] == 0]

            n_dem_use = max(3, int(len(dem_spk) * frac))
            n_con_use = max(3, int(len(con_spk) * frac))

            sel_dem = rng.choice(dem_spk, size=min(n_dem_use, len(dem_spk)), replace=False)
            sel_con = rng.choice(con_spk, size=min(n_con_use, len(con_spk)), replace=False)
            sel_speakers = set(sel_dem) | set(sel_con)

            mask = np.array([s in sel_speakers for s in g])
            X_sub, y_sub, g_sub = X[mask], y[mask], g[mask]

            if len(np.unique(y_sub)) < 2 or len(np.unique(g_sub)) < N_SPLITS:
                continue

            auc_sub, _ = run_cv_simple(X_sub, y_sub, g_sub)
            repeat_aucs.append(auc_sub)

        if repeat_aucs:
            mean_auc = float(np.mean(repeat_aucs))
            std_auc = float(np.std(repeat_aucs))
            n_actual = int(n_use)
            curve[f"{frac:.1f}"] = {
                "n_speakers": n_actual, "fraction": frac,
                "mean_auc": mean_auc, "std_auc": std_auc,
                "all_aucs": [float(a) for a in repeat_aucs],
            }
            fprint(f"    {frac:.0%} ({n_actual:3d} speakers): AUC = {mean_auc:.4f} +/- {std_auc:.4f}  (n_reps={len(repeat_aucs)})")

    learning_results[name] = curve

all_results["learning_curve"] = learning_results


# ══════════════════════════════════════════════════════════════
# EXPERIMENT 5: Permutation Test on Real Model
# ══════════════════════════════════════════════════════════════
fprint(f"\n{'='*70}")
fprint("EXPERIMENT 5: Permutation Test (200 shuffles)")
fprint(f"{'='*70}")

N_PERM = 200

perm_results = {}
for name, X, y, g in [
    ("Agentic", X_ag_ms, y_ag, g_ag),
    ("DementiaNet", X_dn_ms, y_dn, g_dn),
]:
    fprint(f"\n  {name}:")
    # Real model AUC
    real_auc, real_folds = run_cv_simple(X, y, g)
    fprint(f"    Real model AUC: {real_auc:.4f}")

    # Permutation null distribution
    gkf = GroupKFold(n_splits=N_SPLITS)
    rng = np.random.RandomState(RANDOM_STATE)
    perm_aucs = []

    for perm_i in range(N_PERM):
        y_perm = rng.permutation(y)
        fold_aucs = []
        for tr_idx, te_idx in gkf.split(X, y_perm, g):
            sc = StandardScaler()
            Xtr = sc.fit_transform(X[tr_idx])
            Xte = sc.transform(X[te_idx])
            Xtr = np.nan_to_num(Xtr, nan=0.0)
            Xte = np.nan_to_num(Xte, nan=0.0)
            clf = make_clf()
            clf.fit(Xtr, y_perm[tr_idx])
            p = clf.predict_proba(Xte)[:, 1]
            try:
                fold_aucs.append(roc_auc_score(y_perm[te_idx], p))
            except ValueError:
                fold_aucs.append(0.5)
        perm_aucs.append(float(np.mean(fold_aucs)))

        if (perm_i + 1) % 20 == 0:
            fprint(f"      {perm_i + 1}/{N_PERM} permutations done...")

    p_value = float(np.mean([pa >= real_auc for pa in perm_aucs]))
    null_mean = float(np.mean(perm_aucs))
    null_std = float(np.std(perm_aucs))
    null_95 = float(np.percentile(perm_aucs, 95))

    fprint(f"    Null distribution: {null_mean:.4f} +/- {null_std:.4f} (95th pct: {null_95:.4f})")
    fprint(f"    p-value: {p_value:.4f}  {'SIGNIFICANT' if p_value < 0.05 else 'NOT significant'}")
    fprint(f"    Effect size (real - null): {real_auc - null_mean:+.4f}")

    perm_results[name] = {
        "real_auc": real_auc,
        "real_fold_aucs": real_folds,
        "null_mean": null_mean,
        "null_std": null_std,
        "null_95th": null_95,
        "p_value": p_value,
        "effect_size": real_auc - null_mean,
        "perm_aucs": perm_aucs,
    }

all_results["permutation_test"] = perm_results


# ══════════════════════════════════════════════════════════════
# EXPERIMENT 6: PCA sweep on mean-only (to separate the effects)
# ══════════════════════════════════════════════════════════════
fprint(f"\n{'='*70}")
fprint("EXPERIMENT 6: PCA Sweep on Mean-Only (2560 dims)")
fprint(f"{'='*70}")

pca_mean_only_results = {}
for name, X, y, g in [
    ("Agentic", X_ag_m, y_ag, g_ag),
    ("DementiaNet", X_dn_m, y_dn, g_dn),
]:
    fprint(f"\n  {name} (mean-only, {X.shape[1]} dims, {len(y)} speakers):")
    sweep = {}
    for pca_d in [16, 32, 64, 128, 256, None]:
        label = f"PCA-{pca_d}" if pca_d else f"Full-{X.shape[1]}"
        effective_dims = pca_d if pca_d else X.shape[1]
        auc, folds = run_cv_simple(X, y, g, pca_dims=pca_d)
        ratio = effective_dims / len(y)
        fprint(f"    {label:12s}: AUC = {auc:.4f}  (ratio = {ratio:.1f}:1)")
        sweep[label] = {"auc": auc, "dims": effective_dims, "ratio": round(ratio, 2), "fold_aucs": folds}
    pca_mean_only_results[name] = sweep

all_results["pca_sweep_mean_only"] = pca_mean_only_results


# ══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════
fprint(f"\n\n{'='*80}")
fprint("FINAL SUMMARY: Dimensionality Robustness")
fprint(f"{'='*80}")

fprint(f"\n1. MEAN-ONLY vs MEAN+STD:")
for name in ["Agentic", "DementiaNet"]:
    r = ablation_results[name]
    fprint(f"  {name}: mean+std={r['mean_std_full']['auc']:.4f}, mean-only={r['mean_only_full']['auc']:.4f}, "
          f"delta={r['std_delta_full']:+.4f}")

fprint(f"\n2. PCA SWEEP (mean+std):")
for name in ["Agentic", "DementiaNet"]:
    r = pca_results[name]
    aucs = [(k, v['auc']) for k, v in r.items()]
    fprint(f"  {name}: " + ", ".join(f"{k}={a:.3f}" for k, a in aucs))

fprint(f"\n3. C SWEEP:")
for name in ["Agentic", "DementiaNet"]:
    r = c_results[name]
    best = max(r.items(), key=lambda x: x[1]['auc'])
    fprint(f"  {name}: best C={best[1]['C']} (AUC={best[1]['auc']:.4f}), "
          f"C=1.0 AUC={r['C=1.0']['auc']:.4f}")

fprint(f"\n4. LEARNING CURVE (last 3 points):")
for name in ["Agentic", "DementiaNet"]:
    r = learning_results[name]
    for k in sorted(r.keys())[-3:]:
        v = r[k]
        fprint(f"  {name} {float(k):.0%}: {v['n_speakers']} speakers, AUC={v['mean_auc']:.4f} +/- {v['std_auc']:.4f}")

fprint(f"\n5. PERMUTATION TEST:")
for name in ["Agentic", "DementiaNet"]:
    r = perm_results[name]
    fprint(f"  {name}: real={r['real_auc']:.4f}, null={r['null_mean']:.4f}+/-{r['null_std']:.4f}, "
          f"p={r['p_value']:.4f}, effect={r['effect_size']:+.4f}")

fprint(f"\n6. PCA SWEEP (mean-only):")
for name in ["Agentic", "DementiaNet"]:
    r = pca_mean_only_results[name]
    aucs = [(k, v['auc']) for k, v in r.items()]
    fprint(f"  {name}: " + ", ".join(f"{k}={a:.3f}" for k, a in aucs))

# Save results
all_results["metadata"] = {
    "solver": "lbfgs",
    "max_iter": 5000,
    "n_permutations": N_PERM,
    "n_splits": N_SPLITS,
    "random_state": RANDOM_STATE,
    "agentic_speakers": n_ag,
    "dementianet_speakers": n_dn,
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(all_results, f, indent=2, default=str)
fprint(f"\nResults saved to: {OUTPUT_PATH}")
