import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
import os

# Try openTSNE if available, fallback to sklearn
try:
    from openTSNE import TSNE
    use_opentsne = True
except ImportError:
    from sklearn.manifold import TSNE
    use_opentsne = False

# === Config ===
MAX_PER_CLASS = 800  # adjust as needed

# === Load data ===
print("🔍 Loading embeddings and labels...")
X = np.load("node2vec_embeddings.npy")
y = np.load("y_labels.npy")
le = joblib.load("label_encoder.pkl")
class_names = le.classes_

# === Subsample balanced set ===
print(f"📦 Sampling up to {MAX_PER_CLASS} nodes per language group...")
np.random.seed(42)
sample_indices = []

for label in np.unique(y):
    idx = np.where(y == label)[0]
    n = min(MAX_PER_CLASS, len(idx))
    chosen = np.random.choice(idx, size=n, replace=False)
    sample_indices.extend(chosen)

X_sub = X[sample_indices]
y_sub = y[sample_indices]

# === Run t-SNE ===
print(f"🌀 Running t-SNE on {len(sample_indices)} nodes...")
if use_opentsne:
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        metric="cosine",
        n_jobs=os.cpu_count(),
        random_state=42,
    )
    X_2d = tsne.fit(X_sub)
else:
    from sklearn.manifold import TSNE
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        metric="cosine",
        random_state=42,
        n_iter=1000
    )
    X_2d = tsne.fit_transform(X_sub)

# === Plot ===
print("🎨 Plotting...")
plt.figure(figsize=(12, 10))
scatter = plt.scatter(
    X_2d[:, 0], X_2d[:, 1],
    c=y_sub, cmap="tab10", s=10, alpha=0.7
)

# Colorbar with language labels
cbar = plt.colorbar(scatter, ticks=np.arange(len(class_names)))
cbar.ax.set_yticklabels(class_names)
cbar.set_label("Language Group")

plt.title(f"Balanced t-SNE of Node2Vec Embeddings ({MAX_PER_CLASS} per group)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.grid(True)
plt.tight_layout()
plt.savefig("tsne_balanced_sample.png", dpi=300)
plt.show()

print("✅ Saved as tsne_balanced_sample.png")
