import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import Node2Vec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score
import xgboost as xgb
import joblib
import os
import json
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# === Load Node Features ===
try:
    df = pd.read_csv('large_twitch_features.csv')
    print("✅ Successfully loaded features data")
except Exception as e:
    print(f"❌ Error loading features: {e}")
    raise

# === Language Grouping ===
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

try:
    df['lang_group'] = df['language'].apply(group_language)
    le = LabelEncoder()
    df['lang_group_label'] = le.fit_transform(df['lang_group'])
    y = df['lang_group_label'].values
    num_nodes = df.shape[0]
    print("✅ Successfully processed language groups")
except Exception as e:
    print(f"❌ Error in language processing: {e}")
    raise

# === Build Graph from Edges ===
try:
    id_map = {id_: i for i, id_ in enumerate(df['numeric_id'])}
    edges_df = pd.read_csv('large_twitch_edges.csv')
    edges_df = edges_df[
        edges_df['numeric_id_1'].isin(id_map) & edges_df['numeric_id_2'].isin(id_map)
    ]
    edge_index = torch.tensor([
        [id_map[src] for src in edges_df['numeric_id_1']],
        [id_map[dst] for dst in edges_df['numeric_id_2']]
    ], dtype=torch.long)
    print("✅ Successfully built edge index")
except Exception as e:
    print(f"❌ Error building graph: {e}")
    raise

# === Node2Vec Training ===
try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Reduced parameters for stability
    node2vec = Node2Vec(
        edge_index=edge_index,
        embedding_dim=123,  # Reduced from 128
        walk_length=20,    # Reduced from 20
        context_size=10,    # Reduced from 10
        walks_per_node=10,  # Reduced from 10
        num_negative_samples=1,
        sparse=True
    ).to(device)

    loader = node2vec.loader(batch_size=64, shuffle=True)  # Reduced from 128
    optimizer = torch.optim.SparseAdam(list(node2vec.parameters()), lr=0.01)

    def train_node2vec():
        node2vec.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = node2vec.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    print("Training Node2Vec...")
    for epoch in range(1, 6):
        loss = train_node2vec()
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    # Get embeddings
    node2vec.eval()
    embeddings = node2vec.embedding.weight.detach().cpu().numpy()
    print("✅ Successfully trained Node2Vec and got embeddings")
except Exception as e:
    print(f"❌ Error in Node2Vec training: {e}")
    if torch.cuda.is_available():
        print("You might be running out of GPU memory. Try:")
        print("1. Reducing embedding_dim, walk_length, etc.")
        print("2. Using device='cpu'")
    raise

# === Train/Test Split ===
try:
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, y, test_size=0.2, random_state=42, stratify=y
    )
    print("✅ Successfully created train/test split")
except Exception as e:
    print(f"❌ Error in train/test split: {e}")
    raise

# === XGBoost Training ===
try:
    print("\nTraining XGBoost...")
    
    # Fixed XGBoost initialization (removed label encoder from params)
    clf = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(le.classes_),
        max_depth=6,
        n_estimators=100,
        eval_metric='mlogloss'
    )
    
    clf.fit(X_train, y_train)
    print("✅ Finished training XGBoost")
except Exception as e:
    print(f"❌ Error in XGBoost training: {e}")
    raise

# === Evaluation ===
try:
    print("Predicting...")
    y_pred = clf.predict(X_test)
    print("✅ Done predicting")
    
    acc = np.mean(y_pred == y_test)
    macro_f1 = f1_score(y_test, y_pred, average='macro')

    print(f"\nNode2Vec + XGBoost Accuracy: {acc:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
except Exception as e:
    print(f"❌ Error in evaluation: {e}")
    raise

# === Save Results ===
try:
    # Save the XGBoost model
    joblib.dump(clf, 'xgboost_node2vec_model.pkl')
    
    # Save metadata
    np.save('node2vec_embeddings.npy', embeddings)
    np.save('y_labels.npy', y)

    meta = {
        'classes': list(le.classes_),
        'num_classes': len(le.classes_),
        'embedding_dim': embeddings.shape[1]
    }
    with open('xgboost_node2vec_meta.json', 'w') as f:
        json.dump(meta, f, indent=4)

    # Save the LabelEncoder
    joblib.dump(le, 'label_encoder.pkl')
    print("✅ Successfully saved all models and metadata")
except Exception as e:
    print(f"❌ Error saving results: {e}")
    raise