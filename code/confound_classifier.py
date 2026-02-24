"""
Confound Classifier: Check for metadata leakage in SpeechSense dataset.

Trains LogisticRegression on ONLY metadata features (no embeddings) to predict
dementia label. If AUC >> 0.5, metadata alone predicts the outcome => confound.

Features per speaker:
  1. clip_count - number of clips per speaker
  2. mean_duration - average clip duration in seconds
  3. std_duration - std of clip durations
  4. temporal_category distribution - one-hot encoded proportions

Evaluation: GroupKFold (5-fold, speaker-grouped), matching the real model's CV.
"""

import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. Load manifest
# ============================================================
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
manifest_path = os.path.join(BASE, 'dataset', 'manifest.csv')
df = pd.read_csv(manifest_path)
print(f"Loaded manifest: {df.shape[0]} clips, {df['anon_speaker_id'].nunique()} speakers")
print(f"  Label distribution: {dict(df['label'].value_counts())}")
print()

# ============================================================
# 2. Exclude after_symptoms clips (to match pre-symptoms model)
# ============================================================
before_count = len(df)
df = df[df['temporal_label'] != 'after_symptoms'].copy()
after_count = len(df)
print(f"Excluded after_symptoms: {before_count} -> {after_count} clips ({before_count - after_count} removed)")
print(f"  Remaining speakers: {df['anon_speaker_id'].nunique()}")
print(f"  Label distribution: {dict(df['label'].value_counts())}")
print()

# ============================================================
# 3. Build speaker-level metadata features
# ============================================================
print("Building speaker-level metadata features...")
print()

# Basic aggregation per speaker
speaker_agg = df.groupby('anon_speaker_id').agg(
    label=('label', 'first'),
    clip_count=('clip_uid', 'count'),
    mean_duration=('duration_s', 'mean'),
    std_duration=('duration_s', 'std'),
    total_duration=('duration_s', 'sum'),
).reset_index()
speaker_agg['std_duration'] = speaker_agg['std_duration'].fillna(0)

# Temporal category distribution per speaker (one-hot proportions)
temporal_dummies = pd.get_dummies(df[['anon_speaker_id', 'temporal_label']].set_index('anon_speaker_id')['temporal_label'])
temporal_proportions = temporal_dummies.groupby(level=0).mean()
temporal_proportions.columns = [f'temporal_prop_{c}' for c in temporal_proportions.columns]
temporal_proportions = temporal_proportions.reset_index()

# Merge
speaker_df = speaker_agg.merge(temporal_proportions, on='anon_speaker_id', how='left')

print(f"Speaker-level dataset: {speaker_df.shape[0]} speakers, {speaker_df.shape[1]} columns")
print(f"Feature columns:")
feature_cols = [c for c in speaker_df.columns if c not in ['anon_speaker_id', 'label']]
for c in feature_cols:
    print(f"  {c}: mean={speaker_df[c].mean():.4f}, std={speaker_df[c].std():.4f}")
print()

# Show feature distributions by group
print("=== Feature distributions by label ===")
for c in feature_cols:
    ctrl = speaker_df.loc[speaker_df['label'] == 0, c]
    dem = speaker_df.loc[speaker_df['label'] == 1, c]
    print(f"  {c}:")
    print(f"    Control (n={len(ctrl)}): mean={ctrl.mean():.3f}, std={ctrl.std():.3f}")
    print(f"    Dementia (n={len(dem)}): mean={dem.mean():.3f}, std={dem.std():.3f}")
print()

# ============================================================
# 4. Prepare features and labels
# ============================================================
X = speaker_df[feature_cols].values
y = speaker_df['label'].values
groups = speaker_df['anon_speaker_id'].values

print(f"Feature matrix shape: {X.shape}")
print(f"Label distribution in final dataset: {dict(zip(*np.unique(y, return_counts=True)))}")
print()

# ============================================================
# 5. GroupKFold cross-validation (5-fold, speaker-grouped)
# ============================================================
print("=" * 60)
print("CONFOUND CLASSIFIER: GroupKFold (5-fold) Cross-Validation")
print("=" * 60)
print()

gkf = GroupKFold(n_splits=5)
fold_results = []

all_y_true = []
all_y_prob = []
all_y_pred = []

for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # Train LogisticRegression (matching the real model setup)
    clf = LogisticRegression(
        C=1.0,
        class_weight='balanced',
        solver='saga',
        penalty='l2',
        max_iter=5000,
        random_state=42,
    )
    clf.fit(X_train_s, y_train)
    
    y_prob = clf.predict_proba(X_test_s)[:, 1]
    y_pred = clf.predict(X_test_s)
    
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    
    n_train_0 = (y_train == 0).sum()
    n_train_1 = (y_train == 1).sum()
    n_test_0 = (y_test == 0).sum()
    n_test_1 = (y_test == 1).sum()
    
    print(f"Fold {fold_idx+1}: AUC={auc:.4f}, F1={f1:.4f}, Acc={acc:.4f}  "
          f"(train: {len(y_train)} [{n_train_0}/{n_train_1}], "
          f"test: {len(y_test)} [{n_test_0}/{n_test_1}])")
    
    fold_results.append({'auc': auc, 'f1': f1, 'acc': acc})
    all_y_true.extend(y_test)
    all_y_prob.extend(y_prob)
    all_y_pred.extend(y_pred)

print()

# ============================================================
# 6. Summary statistics
# ============================================================
mean_auc = np.mean([r['auc'] for r in fold_results])
std_auc = np.std([r['auc'] for r in fold_results])
mean_f1 = np.mean([r['f1'] for r in fold_results])
std_f1 = np.std([r['f1'] for r in fold_results])
mean_acc = np.mean([r['acc'] for r in fold_results])
std_acc = np.std([r['acc'] for r in fold_results])

# Overall metrics from pooled predictions
overall_auc = roc_auc_score(all_y_true, all_y_prob)
overall_f1 = f1_score(all_y_true, all_y_pred)
overall_acc = accuracy_score(all_y_true, all_y_pred)

print("=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"  Mean AUC:      {mean_auc:.4f} +/- {std_auc:.4f}")
print(f"  Mean F1:       {mean_f1:.4f} +/- {std_f1:.4f}")
print(f"  Mean Accuracy: {mean_acc:.4f} +/- {std_acc:.4f}")
print()
print(f"  Pooled AUC:      {overall_auc:.4f}")
print(f"  Pooled F1:       {overall_f1:.4f}")
print(f"  Pooled Accuracy: {overall_acc:.4f}")
print()

# Classification report on pooled predictions
print("Pooled Classification Report:")
print(classification_report(all_y_true, all_y_pred, target_names=['Control', 'Dementia']))

# ============================================================
# 7. Interpretation
# ============================================================
print("=" * 60)
print("INTERPRETATION")
print("=" * 60)

if mean_auc < 0.55:
    verdict = "NO LEAKAGE DETECTED"
    detail = (
        "Metadata features (clip count, duration, temporal category) are NOT\n"
        "predictive of dementia status. AUC is near chance (0.5).\n"
        "This means the real model's performance is driven by actual speech\n"
        "content features, not dataset artifacts."
    )
elif mean_auc < 0.65:
    verdict = "MILD CONCERN"
    detail = (
        "Metadata features show slight predictive power above chance.\n"
        "This warrants investigation but may not fully explain the real model's\n"
        "performance (which achieves AUC ~0.91)."
    )
elif mean_auc < 0.75:
    verdict = "MODERATE CONCERN"
    detail = (
        "Metadata features are moderately predictive of dementia status.\n"
        "Some of the real model's performance may be partially confounded\n"
        "by metadata/recording characteristics rather than speech content."
    )
else:
    verdict = "SERIOUS LEAKAGE CONCERN"
    detail = (
        "Metadata features are highly predictive of dementia status.\n"
        "The real model's performance may be substantially driven by\n"
        "dataset artifacts rather than genuine speech biomarkers."
    )

print(f"\n  VERDICT: {verdict}")
print(f"  (Mean AUC from metadata-only model = {mean_auc:.4f})")
print()
print(f"  {detail}")
print()

# ============================================================
# 8. Feature importance (from a full-dataset model)
# ============================================================
print("=" * 60)
print("FEATURE IMPORTANCE (full-dataset model coefficients)")
print("=" * 60)

scaler_full = StandardScaler()
X_full_s = scaler_full.fit_transform(X)
clf_full = LogisticRegression(
    C=1.0, class_weight='balanced', solver='saga', penalty='l2',
    max_iter=5000, random_state=42,
)
clf_full.fit(X_full_s, y)

coefs = clf_full.coef_[0]
importance = sorted(zip(feature_cols, coefs), key=lambda x: abs(x[1]), reverse=True)
print()
for feat, coef in importance:
    direction = "-> dementia" if coef > 0 else "-> control"
    print(f"  {feat:40s}  coef={coef:+.4f}  {direction}")
print()

# ============================================================
# 9. Sanity check: random baseline
# ============================================================
print("=" * 60)
print("SANITY CHECK: Random feature baseline")
print("=" * 60)

np.random.seed(42)
X_random = np.random.randn(X.shape[0], X.shape[1])

random_aucs = []
for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X_random, y, groups)):
    X_train_r, X_test_r = X_random[train_idx], X_random[test_idx]
    y_train_r, y_test_r = y[train_idx], y[test_idx]
    
    scaler_r = StandardScaler()
    X_train_rs = scaler_r.fit_transform(X_train_r)
    X_test_rs = scaler_r.transform(X_test_r)
    
    clf_r = LogisticRegression(C=1.0, class_weight='balanced', solver='saga',
                                penalty='l2', max_iter=5000, random_state=42)
    clf_r.fit(X_train_rs, y_train_r)
    
    y_prob_r = clf_r.predict_proba(X_test_rs)[:, 1]
    auc_r = roc_auc_score(y_test_r, y_prob_r)
    random_aucs.append(auc_r)

print(f"\n  Random baseline AUC: {np.mean(random_aucs):.4f} +/- {np.std(random_aucs):.4f}")
print(f"  Metadata model AUC:  {mean_auc:.4f} +/- {std_auc:.4f}")
print(f"  Real model AUC:      ~0.9112 (from memory)")
print()
print("Done.")
