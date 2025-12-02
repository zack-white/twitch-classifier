import numpy as np
import joblib
import json
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

# === Load Data ===
print("🔍 Loading embeddings and labels...")
embeddings = np.load("node2vec_embeddings.npy")
y = np.load("y_labels.npy")
le = joblib.load("label_encoder.pkl")

with open("xgboost_node2vec_meta.json", "r") as f:
    meta = json.load(f)

class_names = meta["classes"]

# === Stage 1: Relabel for EN vs NON-EN ===
EN_label = class_names.index('EN')
y_stage1 = (y == EN_label).astype(int)  # 1 if EN, 0 if NOT EN

# === Train/Test Split ===
X_train, X_test, y_train_stage1, y_test_stage1, y_train_full, y_test_full = train_test_split(
    embeddings, y_stage1, y, test_size=0.2, random_state=42, stratify=y_stage1
)

# === Train Stage 1 Classifier (EN vs NON-EN) ===
print("\n🛠️ Training Stage 1: EN vs NON-EN classifier...")
clf_stage1 = xgb.XGBClassifier(
    objective="binary:logistic",
    max_depth=6,
    n_estimators=100,
    eval_metric="logloss",
    use_label_encoder=False,
    tree_method="hist"
)
clf_stage1.fit(X_train, y_train_stage1)

# === Predict Stage 1
y_pred_stage1 = clf_stage1.predict(X_test)

# === Prepare Data for Stage 2 (Only NON-EN)
# Only use truly NON-EN nodes for Stage 2 evaluation
stage2_mask = (y_pred_stage1 == 0) & (y_test_full != EN_label)
X_test_stage2 = X_test[stage2_mask]
y_test_stage2_true = y_test_full[stage2_mask]


# Filter out EN label
mask_non_en = y_train_stage1 == 0
X_train_stage2 = X_train[mask_non_en]
y_train_stage2 = y_train_full[mask_non_en]

# Remove EN from labels
non_en_labels = [label for i, label in enumerate(class_names) if i != EN_label]

# Reindex labels for Stage 2
label_map = {old_label: new_idx for new_idx, old_label in enumerate(sorted(set(y_train_stage2)))}
y_train_stage2_reindexed = np.array([label_map[label] for label in y_train_stage2])
y_test_stage2_true_reindexed = np.array([label_map[label] for label in y_test_stage2_true])

# === Train Stage 2 Classifier (among NON-EN groups)
print("\n🛠️ Training Stage 2: NON-EN classifier...")
clf_stage2 = xgb.XGBClassifier(
    objective="multi:softmax",
    num_class=len(non_en_labels),
    max_depth=6,
    n_estimators=100,
    eval_metric="mlogloss",
    use_label_encoder=False,
    tree_method="hist"
)
clf_stage2.fit(X_train_stage2, y_train_stage2_reindexed)

# === Predict Stage 2
y_pred_stage2 = clf_stage2.predict(X_test_stage2)

# === Evaluation
print("\n✅ Stage 1 (EN vs NON-EN) Evaluation:")
acc_stage1 = np.mean(y_pred_stage1 == y_test_stage1)
print(f"Accuracy: {acc_stage1:.4f}")
print(classification_report(y_test_stage1, y_pred_stage1, target_names=["NON-EN", "EN"]))

print("\n✅ Stage 2 (NON-EN Groups) Evaluation:")
acc_stage2 = np.mean(y_pred_stage2 == y_test_stage2_true_reindexed)
print(f"Accuracy (on NON-EN nodes): {acc_stage2:.4f}")
print(classification_report(
    y_test_stage2_true_reindexed,
    y_pred_stage2,
    target_names=[class_names[i] for i in range(len(class_names)) if i != EN_label]
))

# === Save models if needed
joblib.dump(clf_stage1, "stage1_EN_vs_NONEN.pkl")
joblib.dump(clf_stage2, "stage2_NONEN_groups.pkl")
print("\n💾 Models saved: Stage 1 and Stage 2 classifiers.")
