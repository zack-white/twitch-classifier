import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
from openTSNE import TSNE  # much faster than sklearn
import os
from matplotlib.colors import ListedColormap

# === Load embeddings and labels ===
print("🔍 Loading data...")
embeddings = np.load("node2vec_embeddings.npy")
y = np.load("y_labels.npy")
le = joblib.load("label_encoder.pkl")

with open("xgboost_node2vec_meta.json", "r") as f:
    meta = json.load(f)

num_classes = meta["num_classes"]
class_names = meta["classes"]

# === Subsample for speed (optional but recommended) ===
print("📦 Subsampling 800 nodes per class...")
np.random.seed(42)
indices = []

for label in np.unique(y):
    class_indices = np.where(y == label)[0]
    n = min(800, len(class_indices))
    chosen = np.random.choice(class_indices, size=n, replace=False)
    indices.extend(chosen)

embeddings_sub = embeddings[indices]
y_sub = y[indices]

# === Run openTSNE ===
print("🌀 Running t-SNE with openTSNE...")
tsne = TSNE(
    n_components=2,
    perplexity=30,
    metric="cosine",
    n_jobs=os.cpu_count(),
    random_state=42,
)
emb_2d = tsne.fit(embeddings_sub)

# === Plotting ===
print("🎨 Plotting t-SNE result...")
plt.figure(figsize=(10, 8))

# Define a fixed color map using tab10 (max 10 distinct colors)
color_map = plt.get_cmap("tab10")
colors = [color_map(i) for i in range(num_classes)]

# Plot each class with its own color
plt.figure(figsize=(10, 8))
for i in range(num_classes):
    idx = np.where(y_sub == i)[0]
    plt.scatter(
        emb_2d[idx, 0],
        emb_2d[idx, 1],
        c=[colors[i]],
        label=class_names[i],
        s=10,
        alpha=0.7
    )

# Add legend and labels
plt.legend(title="Language Group", loc="best", fontsize=9)
plt.title("t-SNE of Node2Vec Embeddings (Sampled by Language Group)")
plt.xlabel("t-SNE-1")
plt.ylabel("t-SNE-2")
plt.grid(True)
plt.tight_layout()
plt.savefig("node2vec_tsne_sampled.png", dpi=300)
plt.show()

print("✅ Done! Plot saved as node2vec_tsne_sampled.png")


# Colorbar with labels
cbar = plt.colorbar(scatter, ticks=range(num_classes))
cbar.ax.set_yticklabels(class_names)
cbar.set_label("Language Group")

plt.title("t-SNE of Node2Vec Embeddings (Sampled by Language Group)")
plt.xlabel("t-SNE-1")
plt.ylabel("t-SNE-2")
plt.grid(True)
plt.tight_layout()
plt.savefig("node2vec_tsne_sampled.png", dpi=300)
plt.show()

print("✅ Done! Plot saved as node2vec_tsne_sampled.png")
