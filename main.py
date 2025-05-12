from pathlib import Path
from nbformat import v4 as nbf

# Define the base dataset path (your local path)
base_path = "yelp_dataset"  # Point to directory, not a json file

# Set dataset path
BASE_PATH = f"{base_path}"
MAX_REVIEWS = 10000
EPOCHS = 5
LEARNING_RATE = 1e-4


# Setup & Imports
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# Text processing + Zagreb
import nltk
nltk.download('punkt')
nltk.download('stopwords')

def build_cooccurrence_graph(text, window_size=2):
    stop_words = set(stopwords.words('english'))
    words = [word.lower() for word in word_tokenize(text) if word.isalpha()]
    words = [w for w in words if w not in stop_words]
    G = nx.Graph()
    for i in range(len(words)):
        for j in range(i+1, min(i + window_size, len(words))):
            if words[i] != words[j]:
                G.add_edge(words[i], words[j])
    return G

def compute_zagreb_indices(G):
    degrees = dict(G.degree())
    edges = G.edges()
    non_edges = nx.non_edges(G)
    
    # First Zagreb Index: sum of squares of degrees of all vertices
    first_zagreb = sum(d**2 for d in degrees.values())
    
    # Second Zagreb Index: sum of products of degrees of adjacent vertices
    second_zagreb = sum(degrees[u] * degrees[v] for u, v in edges)
    
    # First Zagreb Co-Index: sum of degrees of non-adjacent vertices
    first_zagreb_co = sum(degrees[u] + degrees[v] for u, v in non_edges)
    
    # Second Zagreb Co-Index: sum of products of degrees of non-adjacent vertices
    second_zagreb_co = sum(degrees[u] * degrees[v] for u, v in non_edges)
    
    # Modified Zagreb Index (inverse degree sum)
    modified_zagreb = sum(1 / d for d in degrees.values() if d != 0)
    
    # Generalized Zagreb Index: sum of abs difference of degrees of adjacent vertices
    generalized_zagreb = sum(abs(degrees[u] - degrees[v]) for u, v in edges)
    
    # Hyper Zagreb Index: sum of squares of degree sums of adjacent vertices
    hyper_zagreb = sum((degrees[u] + degrees[v])**2 for u, v in edges)
    
    # Third Zagreb Index: sum of cubes of degrees
    third_zagreb = sum(d**3 for d in degrees.values())
    
    # Forgotten Index (F-Index): sum of cubes of degrees
    f_index = sum(d**3 for d in degrees.values())
    
    return [
        first_zagreb,          # First Zagreb Index
        second_zagreb,         # Second Zagreb Index
        first_zagreb_co,       # First Zagreb Co-Index
        second_zagreb_co,      # Second Zagreb Co-Index
        modified_zagreb,       # Modified Zagreb Index
        generalized_zagreb,    # Generalized Zagreb Index
        hyper_zagreb,          # Hyper Zagreb Index
        third_zagreb,          # Third Zagreb Index
        f_index                # Forgotten Index (F-Index)
    ]

# Load metadata
def load_user_data(user_file):
    user_dict = {}
    with open(user_file, 'r', encoding='utf-8') as f:
        for line in f:
            user = json.loads(line)
            user_dict[user['user_id']] = {
                'review_count': user.get('review_count', 0),
                'elite': 1 if user.get('elite') else 0,
                'average_stars': user.get('average_stars', 0.0),
                'funny_given': user.get('compliment_funny', 0)
            }
    return user_dict

def load_business_data(biz_file):
    biz_dict = {}
    with open(biz_file, 'r', encoding='utf-8') as f:
        for line in f:
            biz = json.loads(line)
            biz_dict[biz['business_id']] = {
                'stars': biz.get('stars', 0.0),
                'review_count': biz.get('review_count', 0)
            }
    return biz_dict

user_info = load_user_data(os.path.join(BASE_PATH, "user.json"))
biz_info = load_business_data(os.path.join(BASE_PATH, "business.json"))

# Build dataset
model = SentenceTransformer('all-MiniLM-L6-v2')
X, y = [], []

with open(os.path.join(BASE_PATH, "review.json"), 'r', encoding='utf-8') as f:
    for i, line in enumerate(tqdm(f, total=MAX_REVIEWS)):
        if i >= MAX_REVIEWS:
            break
        data = json.loads(line)
        text = data["text"]
        label = 1 if data["funny"] > 0 else 0

        G = build_cooccurrence_graph(text)
        zagreb_indices = compute_zagreb_indices(G)

        # For clarity, you could unpack them with names
        first_zagreb, second_zagreb, first_zagreb_co, second_zagreb_co, modified_zagreb, \
        generalized_zagreb, hyper_zagreb, third_zagreb, f_index = zagreb_indices

        embed = model.encode(text)

        user_meta = user_info.get(data["user_id"], {})
        biz_meta = biz_info.get(data["business_id"], {})

        extra = [
            user_meta.get("review_count", 0),
            user_meta.get("elite", 0),
            user_meta.get("average_stars", 0.0),
            user_meta.get("funny_given", 0),
            biz_meta.get("stars", 0.0),
            biz_meta.get("review_count", 0)
        ]

        features = np.concatenate((zagreb_indices, embed, extra))
        X.append(features)
        y.append(label)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

# MLP and training
class HumorMLP(nn.Module):
    def __init__(self, input_dim):
        super(HumorMLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.fc(x)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = HumorMLP(input_dim=X.shape[1]).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

X_train_t = torch.tensor(X_train).to(device)
y_train_t = torch.tensor(y_train).unsqueeze(1).to(device)
X_test_t = torch.tensor(X_test).to(device)

for epoch in range(EPOCHS):
    net.train()
    optimizer.zero_grad()
    outputs = net(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")

# Evaluation
net.eval()
with torch.no_grad():
    preds = net(X_test_t).cpu().numpy()
preds_bin = (preds > 0.5).astype(int)
print(classification_report(y_test, preds_bin))