import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import xgboost as xgb

# === Load Everything ===
print("🔍 Loading embeddings and labels...")
embeddings = np.load("node2vec_embeddings.npy")
y_true = np.load("y_labels.npy")
le = joblib.load("label_encoder.pkl")
meta = json.load(open("xgboost_node2vec_meta.json"))
class_names = meta["classes"]

EN_label = class_names.index('EN')

# === Load Two-Stage Models
print("🛠️ Loading two-stage models...")
clf_stage1 = joblib.load("stage1_EN_vs_NONEN.pkl")
clf_stage2 = joblib.load("stage2_NONEN_groups.pkl")

# === Load Original One-Stage Model
print("🛠️ Loading one-stage XGBoost model...")
clf_one_stage = joblib.load("xgboost_node2vec_model.pkl")

# === Predict with One-Stage Classifier
print("🔮 Predicting with one-stage model...")
y_pred_one_stage = clf_one_stage.predict(embeddings)

# === Predict with Two-Stage Classifier
print("🔮 Predicting with two-stage model...")
y_pred_stage1 = clf_stage1.predict(embeddings)

# Initialize
y_pred_two_stage = np.full_like(y_true, fill_value=-1)

# Stage 1: EN vs NON-EN
y_pred_two_stage[y_pred_stage1 == 1] = EN_label

# Stage 2: among NON-EN
mask_non_en = (y_pred_stage1 == 0)
X_non_en = embeddings[mask_non_en]
y_pred_stage2 = clf_stage2.predict(X_non_en)

# Map Stage 2 labels
non_en_labels = [i for i in range(len(class_names)) if i != EN_label]
reverse_label_map = {idx: label for idx, label in enumerate(non_en_labels)}
y_pred_non_en_mapped = np.array([reverse_label_map[p] for p in y_pred_stage2])
y_pred_two_stage[mask_non_en] = y_pred_non_en_mapped

assert np.all(y_pred_two_stage != -1), "Error: Some nodes were not predicted."

# === Metrics
print("\n✅ One-Stage Model Classification Report:")
print(classification_report(y_true, y_pred_one_stage, target_names=class_names))

print("\n✅ Two-Stage Model Classification Report:")
print(classification_report(y_true, y_pred_two_stage, target_names=class_names))

# === Confusion Matrices
cm_one_stage = confusion_matrix(y_true, y_pred_one_stage)
cm_two_stage = confusion_matrix(y_true, y_pred_two_stage)

# === Plot Side-by-Side
print("\n🎨 Generating side-by-side confusion matrices...")

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

disp1 = ConfusionMatrixDisplay(confusion_matrix=cm_one_stage, display_labels=class_names)
disp1.plot(ax=axes[0], cmap="Blues", xticks_rotation=45, colorbar=False)
axes[0].set_title("One-Stage XGBoost")

disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_two_stage, display_labels=class_names)
disp2.plot(ax=axes[1], cmap="Greens", xticks_rotation=45, colorbar=False)
axes[1].set_title("Two-Stage Classifier")

plt.tight_layout()
plt.savefig("compare_confusion_matrices.png", dpi=300)
plt.show()

print("✅ Saved as compare_confusion_matrices.png")
