import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json
import joblib

# === Load Node Features ===
print("🔍 Loading data...")
df = pd.read_csv('large_twitch_features.csv')

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

df['lang_group'] = df['language'].apply(group_language)
le = LabelEncoder()
df['lang_group_label'] = le.fit_transform(df['lang_group'])
y = torch.tensor(df['lang_group_label'].values, dtype=torch.long)
num_nodes = df.shape[0]

# Very simple input features: just views and lifetime (normalize them)
features = torch.tensor(df[['views', 'life_time']].values, dtype=torch.float32)
features = (features - features.mean(dim=0)) / features.std(dim=0)

# === Load Graph Structure ===
id_map = {id_: i for i, id_ in enumerate(df['numeric_id'])}
edges_df = pd.read_csv('large_twitch_edges.csv')
edges_df = edges_df[
    edges_df['numeric_id_1'].isin(id_map) & edges_df['numeric_id_2'].isin(id_map)
]

edge_index = torch.tensor([
    [id_map[src] for src in edges_df['numeric_id_1']],
    [id_map[dst] for dst in edges_df['numeric_id_2']]
], dtype=torch.long)

data = Data(x=features, edge_index=edge_index, y=y)

# === Train/Test Split (Indices Only)
train_idx, test_idx = train_test_split(
    np.arange(num_nodes),
    test_size=0.2,
    random_state=42,
    stratify=y.numpy()
)
train_idx = torch.tensor(train_idx, dtype=torch.long)
test_idx = torch.tensor(test_idx, dtype=torch.long)

# === Compute Class Weights
print("⚖️ Computing class weights...")
class_counts = torch.bincount(y)
class_weights = 1.0 / class_counts.float()
class_weights = class_weights * (len(class_counts) / class_weights.sum())  # Normalize

print("Class Weights:", class_weights.tolist())

# === Define GCN
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(in_channels=features.shape[1], hidden_channels=64, out_channels=len(le.classes_)).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

# === Training Loop
print("🚀 Starting training...")
for epoch in range(1, 201):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = loss_fn(out[train_idx], data.y[train_idx])
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        model.eval()
        _, pred = out.max(dim=1)
        train_acc = int((pred[train_idx] == data.y[train_idx]).sum()) / train_idx.size(0)
        test_acc = int((pred[test_idx] == data.y[test_idx]).sum()) / test_idx.size(0)
        print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

# === Final Evaluation
print("\n✅ Final Evaluation:")
model.eval()
_, pred = model(data.x, data.edge_index).max(dim=1)

from sklearn.metrics import classification_report

print(classification_report(
    data.y[test_idx].cpu().numpy(),
    pred[test_idx].cpu().numpy(),
    target_names=le.classes_
))
