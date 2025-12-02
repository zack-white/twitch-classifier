import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
import json

# === Load Data ===
df = pd.read_csv('large_twitch_features.csv')

# === New Feature Engineering ===
df['created_at'] = pd.to_datetime(df['created_at'])
df['updated_at'] = pd.to_datetime(df['updated_at'])
df['account_age'] = (df['updated_at'] - df['created_at']).dt.days
feature_cols = ['views', 'mature', 'life_time', 'affiliate', 'dead_account', 'account_age']

x = torch.tensor(df[feature_cols].values, dtype=torch.float)

# === Grouped Language Labels ===
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

# === Node ID Mapping ===
id_map = {id_: i for i, id_ in enumerate(df['numeric_id'])}

# === Load Edges ===
edges_df = pd.read_csv('large_twitch_edges.csv')
edges = edges_df[
    edges_df['numeric_id_1'].isin(id_map) & edges_df['numeric_id_2'].isin(id_map)
]
edge_index = torch.tensor([
    [id_map[src] for src in edges['numeric_id_1']],
    [id_map[dst] for dst in edges['numeric_id_2']]
], dtype=torch.long)

# === Train/Test Split ===
num_nodes = x.shape[0]
train_idx, test_idx = train_test_split(
    range(num_nodes), test_size=0.2, random_state=42, stratify=y
)
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[train_idx] = True
test_mask[test_idx] = True

data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, test_mask=test_mask)

# === GCN Model ===
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# === Setup ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(x.shape[1], 64, len(le.classes_)).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# === Class Weights for Imbalance ===
class_counts = torch.bincount(data.y[data.train_mask])
class_weights = 1.0 / class_counts.float()
class_weights = class_weights / class_weights.sum()
class_weights = class_weights.to(device)

# === Train and Evaluate ===
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask], weight=class_weights)
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate():
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)
        y_true = data.y[data.test_mask].cpu()
        y_pred = pred[data.test_mask].cpu()
        acc = (y_true == y_pred).float().mean().item()
        macro_f1 = f1_score(y_true, y_pred, average='macro')
    return acc, macro_f1, y_true, y_pred

# === Training Loop with Early Stopping ===
best_acc = 0
best_epoch = 0
best_model_state = None
patience = 20
patience_counter = 0
epochs = 200

for epoch in range(1, epochs + 1):
    loss = train()
    acc, macro_f1, _, _ = evaluate()

    if acc > best_acc:
        best_acc = acc
        best_epoch = epoch
        best_model_state = model.state_dict()
        patience_counter = 0
    else:
        patience_counter += 1

    print(f"Epoch {epoch:03d}, Loss: {loss:.4f}, Acc: {acc:.4f}, Macro F1: {macro_f1:.4f}")

    if patience_counter >= patience:
        print(f"\nEarly stopping at epoch {epoch}")
        break

# === Save Best Model and Params ===
torch.save(best_model_state, 'best_gnn_grouped.pt')
with open('gnn_grouped_hyperparams.json', 'w') as f:
    json.dump({
        'model': 'GCN',
        'features': feature_cols,
        'in_channels': x.shape[1],
        'hidden_channels': 64,
        'out_channels': len(le.classes_),
        'lr': 0.01,
        'weight_decay': 5e-4,
        'epochs_run': epoch,
        'best_epoch': best_epoch,
        'best_test_acc': best_acc,
        'early_stopped': patience_counter >= patience,
        'loss_function': 'weighted_cross_entropy',
        'language_groups': list(le.classes_)
    }, f, indent=4)

# === Final Report ===
print(f"\n✅ Best Epoch: {best_epoch}")
print(f"✅ Best Accuracy: {best_acc:.4f}")
_, _, y_true, y_pred = evaluate()
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=le.classes_))
# === Confusion Matrix ===