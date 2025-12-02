import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import community as community_louvain 

# === Load language data ===
df = pd.read_csv("large_twitch_features.csv")

def group_language(lang):
    if lang == 'EN':
        return 'EN'
    elif lang in {'FR', 'ES', 'PT', 'IT'}:
        return 'ROMANCE'
    elif lang in {'DE', 'NL', 'NO', 'SV', 'DA'}:
        return 'GERMANIC'
    elif lang in {'RU', 'PL', 'CS', 'HU'}:
        return 'SLAVIC'
    elif lang in {'JA', 'KO', 'ZH', 'TH'}:
        return 'ASIAN'
    else:
        return 'OTHER'

df["lang_group"] = df["language"].apply(group_language)
le = LabelEncoder()
df["lang_group_label"] = le.fit_transform(df["lang_group"])
y_true = df["lang_group_label"].values
id_map = {id_: i for i, id_ in enumerate(df["numeric_id"])}

# === Load edges into NetworkX graph ===
edges_df = pd.read_csv("large_twitch_edges.csv")
edges_df = edges_df[
    edges_df["numeric_id_1"].isin(id_map) & edges_df["numeric_id_2"].isin(id_map)
]

G = nx.Graph()
G.add_nodes_from(range(len(df)))
G.add_edges_from([
    (id_map[row.numeric_id_1], id_map[row.numeric_id_2])
    for _, row in edges_df.iterrows()
])

# === Run Louvain Community Detection ===
print("🧩 Running Louvain...")
partition = community_louvain.best_partition(G)

# Convert partition dict to list
louvain_labels = np.array([partition[i] for i in range(len(df))])

# === Compare with language labels
print("\n📊 Comparing Louvain communities with language groups:")
nmi = normalized_mutual_info_score(y_true, louvain_labels)
ari = adjusted_rand_score(y_true, louvain_labels)

print(f"✅ Normalized Mutual Information (NMI): {nmi:.4f}")
print(f"✅ Adjusted Rand Index (ARI):          {ari:.4f}")
