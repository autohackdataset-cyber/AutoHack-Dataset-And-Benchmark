"""
ob1_train.py
============
Observation 1 — Per-attack performance of the baseline IDS (Random Forest)

Reproduces Table 8 and Figure 3 from the paper:
  - Table 8  : Per-attack Precision / Recall / F1-score / AUC of the RF
               baseline IDS, trained on the full preprocessed dataset.
  - Figure 3 : Confusion matrix of the Random Forest model (SubClass).

Mirrors ids.py training logic exactly:
  - Inputs: source/AutoHack2025/train_proc.csv, test_proc.csv
  - process_data() drops Timestamp column
  - feature_columns = columns - {Class, Label, Bus}
  - RandomForestClassifier(n_estimators=130, max_depth=30, n_jobs=18)
  - classification_report digits=4

Difference from ids.py:
  - The "support" column in the classification report is replaced with AUC
    (one-vs-rest, computed from predict_proba).

Outputs (saved under Result/Result_ob1/):
  - f1_auc_result.txt             : Precision / Recall / F1 / AUC table
  - confusion_matrix.csv          : raw confusion matrix (consumed by ob2)
  - confusion_matrix.png/jpg/pdf  : confusion matrix figure
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize

warnings.filterwarnings('ignore')


# =============================================================================
# Configuration
# =============================================================================
PROGRAM_PATH = os.path.dirname(os.path.abspath(__file__))
SOURCE_PATH = os.path.join(PROGRAM_PATH, ".." , "preprocess", "source", "AutoHack2025")
TRAIN_FILE = os.path.join(SOURCE_PATH, "train_proc.csv")
TEST_FILE  = os.path.join(SOURCE_PATH, "test_proc.csv")
OUTDIR     = os.path.join(PROGRAM_PATH, "..", "Result", "observation1")

os.makedirs(OUTDIR, exist_ok=True)

# RF hyperparameters — same as ids.py best_S
RF_N_ESTIMATORS = 130
RF_MAX_DEPTH    = 30
N_JOBS          = 18

# Label mapping for display
LABEL_MAP = {
    0: 'Normal',
    1: 'DoS',
    2: 'Spoofing',
    3: 'Replay',
    4: 'Fuzzing',
    5: 'UDS_Spoofing',
}
CLS_ORDER = ['Normal', 'DoS', 'Spoofing', 'Replay', 'Fuzzing', 'UDS_Spoofing']


# =============================================================================
# Helper — same process_data() as ids.py
# =============================================================================
def process_data(data):
    """Drop Timestamp column (mirrors ids.py.process_data)."""
    data.drop(['Timestamp'], axis=1, errors='ignore', inplace=True)
    return data


def save_fig(fig, basename):
    """Save figure as png, jpg, pdf."""
    for ext in ['png', 'jpg', 'pdf']:
        path = os.path.join(OUTDIR, f"{basename}.{ext}")
        fig.savefig(
            path, dpi=150, bbox_inches='tight',
            format='jpeg' if ext == 'jpg' else ext,
        )
        print(f"  saved -> {basename}.{ext}")


# =============================================================================
# 1. Load data (mirrors ids.py.define_file)
# =============================================================================
print("=" * 60)
print("1. Load data")
print("=" * 60)
print(f"  train: {TRAIN_FILE}")
print(f"  test : {TEST_FILE}")

df = process_data(pd.read_csv(TRAIN_FILE))
tf = process_data(pd.read_csv(TEST_FILE))

print(f"  train shape = {df.shape}")
print(f"  test  shape = {tf.shape}")

# feature_columns : same logic as ids.py
#   self.feature_columns = list(self.df.columns.difference(['Class','Label','Bus']))
feature_columns = list(df.columns.difference(['Class', 'Label', 'Bus']))
print(f"  features ({len(feature_columns)}): {feature_columns}")

train_x = df[feature_columns]
test_x  = tf[feature_columns]
train_y = df['Label']
test_y  = tf['Label']

print(f"\n  train Label value_counts:")
print(train_y.value_counts().sort_index().to_string())
print(f"\n  test Label value_counts:")
print(test_y.value_counts().sort_index().to_string())


# =============================================================================
# 2. Train RandomForest (mirrors ids.py.train_model_rf)
# =============================================================================
print("\n" + "=" * 60)
print("2. Train RandomForest")
print("=" * 60)
print(f"  n_estimators = {RF_N_ESTIMATORS}")
print(f"  max_depth    = {RF_MAX_DEPTH}")
print(f"  n_jobs       = {N_JOBS}")

clf_S = RandomForestClassifier(
    n_estimators=RF_N_ESTIMATORS,
    max_depth=RF_MAX_DEPTH,
    n_jobs=N_JOBS,
)
print("  Start training model S")
clf_S.fit(train_x, train_y)
print("  Done.")


# =============================================================================
# 3. Predict
# =============================================================================
print("\n" + "=" * 60)
print("3. Predict")
print("=" * 60)
predict_S = clf_S.predict(test_x)
predict_S_prob = clf_S.predict_proba(test_x)
print(f"  predictions shape = {predict_S.shape}")
print(f"  predict_proba shape = {predict_S_prob.shape}")

# Map integer labels to display strings (same as ids.py.save_report)
S_label = test_y.map(LABEL_MAP)
predict_S_labels = pd.Series(predict_S).map(LABEL_MAP)


# =============================================================================
# 4. Build F1 + AUC table (support replaced with AUC)
# =============================================================================
print("\n" + "=" * 60)
print("4. F1 + AUC table")
print("=" * 60)

# classification_report — same digits=4 as ids.py
report = classification_report(
    S_label, predict_S_labels,
    zero_division=0, digits=4, output_dict=True,
)

# AUC (one-vs-rest)
# clf_S.classes_ is the integer-label order used by predict_proba columns
classes_in_model = list(clf_S.classes_)
y_true_int = test_y.values.astype(int)
y_true_bin = label_binarize(y_true_int, classes=classes_in_model)

aucs = {}
for i, cls_int in enumerate(classes_in_model):
    cls_name = LABEL_MAP.get(int(cls_int), str(cls_int))
    try:
        if y_true_bin.shape[1] == 1:
            # Edge case: only 2 classes overall
            auc = roc_auc_score(y_true_bin, predict_S_prob[:, i])
        else:
            auc = roc_auc_score(y_true_bin[:, i], predict_S_prob[:, i])
    except Exception:
        auc = float('nan')
    aucs[cls_name] = auc

macro_auc = float(np.nanmean(list(aucs.values())))

# Build the printed/saved table
header = f"{'Attack Type':15s} {'Precision':>10} {'Recall':>10} {'F1-score':>10} {'AUC':>10}"
sep    = "-" * 60
lines  = [header, sep]

for cls in CLS_ORDER:
    if cls in report:
        r = report[cls]
        lines.append(
            f"{cls:15s} {r['precision']:10.4f} {r['recall']:10.4f} "
            f"{r['f1-score']:10.4f} {aucs.get(cls, float('nan')):10.4f}"
        )

lines.append(sep)
for avg in ['macro avg', 'weighted avg']:
    if avg in report:
        r = report[avg]
        label = 'Macro Avg' if avg == 'macro avg' else 'Weighted Avg'
        lines.append(
            f"{label:15s} {r['precision']:10.4f} {r['recall']:10.4f} "
            f"{r['f1-score']:10.4f} {macro_auc:10.4f}"
        )

result = "\n".join(lines)
print("\n" + result)

result_path = os.path.join(OUTDIR, "f1_auc_result.txt")
with open(result_path, "w", encoding="utf-8") as f:
    f.write(result)
print(f"\n  saved -> f1_auc_result.txt")


# =============================================================================
# 5. Confusion Matrix (mirrors ids.py.show_confusion_matrix SubClass part)
# =============================================================================
print("\n" + "=" * 60)
print("5. Confusion Matrix")
print("=" * 60)

# Use only classes that actually appear in the data
present = [c for c in CLS_ORDER if c in set(S_label) | set(predict_S_labels)]
cm = confusion_matrix(S_label, predict_S_labels, labels=present)

# Save confusion matrix as CSV (consumed by ob2_train.py as Figure 4 LEFT)
cm_df = pd.DataFrame(cm, index=present, columns=present)
cm_csv_path = os.path.join(OUTDIR, "confusion_matrix.csv")
cm_df.to_csv(cm_csv_path)
print(f"  saved -> confusion_matrix.csv (for Ob2 reuse)")

fig, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(
    cm, annot=True, fmt=",d", cmap="Blues",
    xticklabels=present, yticklabels=present,
    annot_kws={'size': 13, 'weight': 'bold'},
    linewidths=0.5, linecolor='gray', ax=ax,
)
ax.set_xlabel("Predicted Label", fontsize=19, labelpad=12)
ax.set_ylabel("True Label",      fontsize=19, labelpad=12)
ax.set_title("Confusion Matrix — SubClass",
             fontsize=21, fontweight='bold', pad=15)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right',
                   fontweight='bold', fontsize=19)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0,
                   fontweight='bold', fontsize=19)
plt.tight_layout()
save_fig(fig, "confusion_matrix")
plt.close(fig)


print("\n" + "=" * 60)
print(f"All outputs saved -> {OUTDIR}")
print("=" * 60)
