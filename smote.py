import numpy as np
import joblib
import json
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# === Load Data ===
print("🔍 Loading embeddings and labels...")
embeddings = np.load("node2vec_embeddings.npy")
y = np.load("y_labels.npy")
meta = json.load(open("xgboost_node2vec_meta.json"))
class_names = meta["classes"]

# === Apply SMOTE to Node2Vec Embeddings ===
print("🧬 Applying SMOTE to rebalance classes...")
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(embeddings, y)

# === Train/Test Split AFTER balancing
print("✂️ Splitting train/test...")
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
)

# === Train XGBoost on rebalanced data
print("🚀 Training XGBoost...")
clf = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=len(class_names),
    max_depth=6,
    n_estimators=100,
    eval_metric='mlogloss',
    tree_method="hist"
)
clf.fit(X_train, y_train)

# === Evaluate
print("\n✅ Evaluating...")
y_pred = clf.predict(X_test)

acc = np.mean(y_pred == y_test)
macro_f1 = f1_score(y_test, y_pred, average='macro')

print(f"✅ SMOTE XGBoost Accuracy: {acc:.4f}")
print(f"✅ SMOTE Macro F1 Score: {macro_f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# === Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(ax=ax, cmap="Purples", xticks_rotation=45)
plt.title("SMOTE-Enhanced XGBoost Confusion Matrix")
plt.tight_layout()
plt.savefig("xgboost_smote_confusion_matrix.png", dpi=300)
plt.show()

print("\n📸 Confusion matrix saved: xgboost_smote_confusion_matrix.png")
