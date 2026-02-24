"""
Generate Pre-Symptoms Plots
=================================
Reads the classification metrics JSON produced by the evaluate script,
runs OOF predictions for the confusion matrix, and generates:
  - AUC/F1/Accuracy bar chart
  - Speaker-level confusion matrix
  - OOF CSV

Usage:
    python code/generate_pre_symptoms_plots.py
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "dataset" / "manifest.csv"
TAN_NPZ = (
    ROOT
    / "reproducibility"
    / "multimodal"
    / "outputs_multimodal_agentic_manifest_text_acoustic_narrative"
    / "multimodal_embeddings_agentic_manifest_text_acoustic_narrative.npz"
)
METRICS_JSON = (
    ROOT
    / "reproducibility"
    / "multimodal"
    / "outputs_text_acoustic_narrative_pre_symptoms"
    / "classification_metrics_pre_symptoms.json"
)
OUT_REPRO_DIR = (
    ROOT
    / "reproducibility"
    / "multimodal"
    / "outputs_text_acoustic_narrative_pre_symptoms"
)
OUT_PLOTS_DIR = ROOT / "plots"

OOF_CSV = OUT_REPRO_DIR / "oof_text_acoustic_narrative_pre_symptoms.csv"
CM_JSON = OUT_REPRO_DIR / "cm_text_acoustic_narrative_pre_symptoms.json"
METRIC_PLOT = OUT_PLOTS_DIR / "auc_f1_acc_pre_symptoms.png"
CM_PLOT = OUT_PLOTS_DIR / "cm_text_acoustic_narrative_pre_symptoms.png"

RANDOM_STATE = 42
THRESHOLD = 0.5


def is_after_symptoms_clip(clip_name: str) -> bool:
    return "_after_symptoms_" in str(clip_name).lower()


def load_manifest_map():
    df = pd.read_csv(MANIFEST_PATH)
    m = {}
    for _, row in df.iterrows():
        stem = os.path.splitext(str(row["anon_filename"]))[0]
        m[stem] = {
            "label": int(row["label"]),
            "speaker_id": str(row["anon_speaker_id"]),
        }
    return m


def aggregate_speaker(X, y, g):
    by_spk = defaultdict(lambda: {"x": [], "y": None})
    for i, sid in enumerate(g):
        by_spk[sid]["x"].append(X[i])
        by_spk[sid]["y"] = int(y[i])

    Xs, ys, gs = [], [], []
    for sid, row in sorted(by_spk.items()):
        arr = np.stack(row["x"], axis=0)
        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0) if arr.shape[0] > 1 else np.zeros_like(mean)
        Xs.append(np.concatenate([mean, std]).astype(np.float32))
        ys.append(int(row["y"]))
        gs.append(sid)
    return np.stack(Xs, axis=0), np.asarray(ys, dtype=np.int64), np.asarray(gs)


def predict_probs_cv(X, y, groups):
    gkf = GroupKFold(n_splits=5)
    y_true_all, p_all, g_all, fold_all = [], [], [], []

    for fold_id, (tr, te) in enumerate(gkf.split(X, y, groups), 1):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]

        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)
        Xtr = np.nan_to_num(Xtr, nan=0.0, posinf=0.0, neginf=0.0)
        Xte = np.nan_to_num(Xte, nan=0.0, posinf=0.0, neginf=0.0)

        clf = LogisticRegression(
            max_iter=2000,
            C=1.0,
            class_weight="balanced",
            solver="saga",
            penalty="l2",
            random_state=RANDOM_STATE,
        )
        clf.fit(Xtr, ytr)
        p = clf.predict_proba(Xte)[:, 1]

        y_true_all.append(yte)
        p_all.append(p)
        g_all.append(groups[te])
        fold_all.append(np.full(len(te), fold_id, dtype=np.int64))

    return (
        np.concatenate(y_true_all),
        np.concatenate(p_all),
        np.concatenate(g_all),
        np.concatenate(fold_all),
    )


def plot_metrics(metrics_json_path: Path, out_path: Path):
    with open(metrics_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    m = data["text_plus_acoustic_narrative"]
    metric_names = ["AUC", "F1", "Accuracy"]
    vals = [m["auc"], m["f1"], m["accuracy"]]

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    bars = ax.bar(metric_names, vals, color=["#5B8FF9", "#61DDAA", "#F6BD16"])
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Text+Acoustic Narrative (no after-symptoms)\nSpeaker-level CV metrics")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(5.2, 4.8), constrained_layout=True)
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1], ["Control (0)", "Dementia (1)"])
    ax.set_yticks([0, 1], ["Control (0)", "Dementia (1)"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    max_v = max(cm.max(), 1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > max_v * 0.5 else "black"
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color=color, fontsize=11)

    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    OUT_REPRO_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Plot metrics bar chart
    plot_metrics(METRICS_JSON, METRIC_PLOT)

    # Generate OOF predictions for confusion matrix
    manifest_map = load_manifest_map()
    tan_pack = np.load(TAN_NPZ, allow_pickle=True)["conditions"].item()
    tan_clip = {k: np.asarray(v, dtype=np.float32) for k, v in tan_pack["text_plus_acoustic_narrative"].items()}

    all_clips = sorted(tan_clip.keys())
    filtered = [c for c in all_clips if not is_after_symptoms_clip(c)]
    n_excluded = len(all_clips) - len(filtered)

    X_list, y_list, g_list = [], [], []
    for clip_name in sorted(filtered):
        info = manifest_map.get(clip_name)
        if info is None:
            continue
        X_list.append(tan_clip[clip_name])
        y_list.append(int(info["label"]))
        g_list.append(str(info["speaker_id"]))

    X_clip = np.asarray(X_list, dtype=np.float32)
    y_clip = np.asarray(y_list, dtype=np.int64)
    g_clip = np.asarray(g_list)

    X_spk, y_spk, g_spk = aggregate_speaker(X_clip, y_clip, g_clip)

    y_true, p, speaker_ids, folds = predict_probs_cv(X_spk, y_spk, g_spk)
    y_pred = (p >= THRESHOLD).astype(np.int64)

    # Save OOF CSV
    oof_df = pd.DataFrame({
        "speaker_id": speaker_ids,
        "fold": folds,
        "y_true": y_true,
        "probability": p,
        "y_pred": y_pred,
    }).sort_values(["speaker_id"]).reset_index(drop=True)
    oof_df.to_csv(OOF_CSV, index=False)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_payload = {
        "threshold": THRESHOLD,
        "labels": ["control", "dementia"],
        "n_speakers": int(len(y_true)),
        "n_excluded_after_symptoms": int(n_excluded),
        "matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
    }
    with open(CM_JSON, "w", encoding="utf-8") as f:
        json.dump(cm_payload, f, indent=2)

    plot_confusion(
        cm,
        CM_PLOT,
        title="Text+Acoustic Narrative confusion matrix\nspeaker-level, after-symptoms excluded",
    )

    print(f"Saved: {METRIC_PLOT}")
    print(f"Saved: {CM_PLOT}")
    print(f"Saved: {OOF_CSV}")
    print(f"Saved: {CM_JSON}")


if __name__ == "__main__":
    main()
