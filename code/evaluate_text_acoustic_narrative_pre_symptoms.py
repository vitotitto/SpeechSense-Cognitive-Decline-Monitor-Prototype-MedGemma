"""
Evaluate Text+Acoustic Narrative (Pre-Symptoms)
=====================================================
5-fold GroupKFold cross-validation on speaker-level [mean, std] = 5120 dims.
Reports AUC, F1, accuracy. Saves metrics JSON and summary JSON.

Usage:
    python code/evaluate_text_acoustic_narrative_pre_symptoms.py
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "dataset" / "reference_manifests" / "manifest_agentic_clean.csv"
TAN_NPZ = (
    ROOT
    / "reproducibility"
    / "multimodal"
    / "outputs_multimodal_agentic_manifest_text_acoustic_narrative"
    / "multimodal_embeddings_agentic_manifest_text_acoustic_narrative.npz"
)

OUT_DIR = (
    ROOT
    / "reproducibility"
    / "multimodal"
    / "outputs_text_acoustic_narrative_pre_symptoms"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = OUT_DIR / "summary_pre_symptoms.json"
METRICS_PATH = OUT_DIR / "classification_metrics_pre_symptoms.json"

RANDOM_STATE = 42
THRESHOLD = 0.5


def is_after_symptoms_clip(clip_name: str) -> bool:
    return "_after_symptoms_" in str(clip_name).lower()


def _safe_auc(y_true, p):
    try:
        return float(roc_auc_score(y_true, p))
    except ValueError:
        return 0.5


def _metrics(y_true, p, thr=THRESHOLD):
    yp = (p >= thr).astype(np.int64)
    return {
        "auc": _safe_auc(y_true, p),
        "f1": float(f1_score(y_true, yp, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, yp)),
    }


def load_manifest_map():
    df = pd.read_csv(MANIFEST_PATH)
    mapping = {}
    for _, row in df.iterrows():
        stem = os.path.splitext(str(row["anon_filename"]))[0]
        mapping[stem] = {
            "speaker_id": str(row["anon_speaker_id"]),
            "label": int(row["label"]),
        }
    return mapping


def load_text_acoustic_clip_embeddings():
    pack = np.load(TAN_NPZ, allow_pickle=True)["conditions"].item()
    d = {k: np.asarray(v, dtype=np.float32) for k, v in pack["text_plus_acoustic_narrative"].items()}
    print(f"Loaded text+acoustic narrative clip embeddings: {len(d)}")
    return d


def clip_dict_to_arrays(clip_emb_dict, manifest_map):
    X, y, g, clips = [], [], [], []
    miss = 0
    for clip_name, emb in sorted(clip_emb_dict.items()):
        info = manifest_map.get(clip_name)
        if info is None:
            miss += 1
            continue
        X.append(np.asarray(emb, dtype=np.float32))
        y.append(int(info["label"]))
        g.append(str(info["speaker_id"]))
        clips.append(clip_name)
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64), np.asarray(g), clips, miss


def aggregate_speakers(X, y, groups):
    by_speaker = defaultdict(lambda: {"x": [], "y": None})
    for i, sid in enumerate(groups):
        by_speaker[sid]["x"].append(X[i])
        by_speaker[sid]["y"] = int(y[i])

    Xs, ys, gs = [], [], []
    for sid, row in sorted(by_speaker.items()):
        arr = np.stack(row["x"], axis=0)
        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0) if arr.shape[0] > 1 else np.zeros_like(mean)
        Xs.append(np.concatenate([mean, std]).astype(np.float32))
        ys.append(int(row["y"]))
        gs.append(sid)
    return np.stack(Xs, axis=0), np.asarray(ys, dtype=np.int64), np.asarray(gs)


def run_groupkfold_cv(X, y, groups):
    """5-fold GroupKFold CV returning per-fold AUCs and OOF predictions."""
    gkf = GroupKFold(n_splits=5)
    fold_aucs = []
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

        fold_aucs.append(_safe_auc(yte, p))
        y_true_all.append(yte)
        p_all.append(p)
        g_all.append(groups[te])
        fold_all.append(np.full(len(te), fold_id, dtype=np.int64))

    return {
        "mean_auc": float(np.mean(fold_aucs)),
        "std_auc": float(np.std(fold_aucs)),
        "fold_aucs": fold_aucs,
        "y_true": np.concatenate(y_true_all),
        "p": np.concatenate(p_all),
        "g": np.concatenate(g_all),
        "folds": np.concatenate(fold_all),
    }


def main():
    manifest_map = load_manifest_map()
    tan_clip = load_text_acoustic_clip_embeddings()

    # Filter after-symptoms clips
    all_clips = sorted(tan_clip.keys())
    filtered_clips = [c for c in all_clips if not is_after_symptoms_clip(c)]
    n_excluded = len(all_clips) - len(filtered_clips)
    tan_filtered = {k: tan_clip[k] for k in filtered_clips}

    print(f"Total clips: {len(all_clips)}")
    print(f"Excluded after-symptoms: {n_excluded}")
    print(f"Remaining clips: {len(filtered_clips)}")

    X_clip, y_clip, g_clip, clip_names, miss = clip_dict_to_arrays(tan_filtered, manifest_map)
    if miss:
        print(f"Clips missing from manifest: {miss}")

    X_spk, y_spk, g_spk = aggregate_speakers(X_clip, y_clip, g_clip)
    print(f"Speakers: {len(y_spk)} ({np.sum(y_spk == 1)} dementia, {np.sum(y_spk == 0)} control)")
    print(f"Speaker feature dim: {X_spk.shape[1]} (expected 5120)")

    # Run CV
    cv_results = run_groupkfold_cv(X_spk, y_spk, g_spk)
    oof_metrics = _metrics(cv_results["y_true"], cv_results["p"])

    print(f"\n5-fold GroupKFold CV:")
    print(f"  Mean AUC: {cv_results['mean_auc']:.4f} +/- {cv_results['std_auc']:.4f}")
    print(f"  Per-fold: {[f'{a:.4f}' for a in cv_results['fold_aucs']]}")
    print(f"  OOF F1: {oof_metrics['f1']:.4f}")
    print(f"  OOF Accuracy: {oof_metrics['accuracy']:.4f}")

    # Save summary JSON
    summary = {
        "dataset": "agentic_manifest_text",
        "condition": "text_acoustic_narrative_pre_symptoms",
        "n_clips_total": int(len(all_clips)),
        "n_clips_excluded_after_symptoms": int(n_excluded),
        "n_clips_used": int(len(clip_names)),
        "n_speakers": int(X_spk.shape[0]),
        "dims": {
            "clip_embedding_dim": 2560,
            "speaker_feature_dim": int(X_spk.shape[1]),
            "aggregation": "[clip_mean(2560), clip_std(2560)]",
        },
        "cv_results": {
            "mean_auc": cv_results["mean_auc"],
            "std_auc": cv_results["std_auc"],
            "fold_aucs": cv_results["fold_aucs"],
        },
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {SUMMARY_PATH}")

    # Save classification metrics JSON
    metrics_out = {
        "threshold": THRESHOLD,
        "n_speakers": int(len(y_spk)),
        "n_clips_used": int(len(clip_names)),
        "n_excluded_after_symptoms": int(n_excluded),
        "text_plus_acoustic_narrative": {
            "auc": oof_metrics["auc"],
            "f1": oof_metrics["f1"],
            "accuracy": oof_metrics["accuracy"],
        },
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"Saved: {METRICS_PATH}")


if __name__ == "__main__":
    main()
