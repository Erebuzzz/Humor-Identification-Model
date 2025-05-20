import os
import json
import numpy as np
import networkx as nx
from tqdm import tqdm
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet

# For embeddings
from gensim.models import Word2Vec
from transformers import BertTokenizer, BertModel
BERT_LOCAL_PATH = r"C:\Users\kshit\OneDrive\Documents\GitHub\Humor-Identification-Model\bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(BERT_LOCAL_PATH)
bert_model = BertModel.from_pretrained(BERT_LOCAL_PATH)

import torch

# For GloVe
def load_glove_embeddings(filepath, dim=100):
    embeddings = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

# For PyTorch MLP (with RMSprop)
import torch.nn as nn
import torch.optim as optim

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# ========== CONFIGURATION ==========
DATASET_PATH = r'D:\Humor Detection\yelp_dataset\yelp_academic_dataset_review.json'
MAX_SAMPLES = 15000
GLOVE_PATH = r'C:\Users\kshit\OneDrive\Documents\GitHub\Humor-Identification-Model\glove.6B.100d.txt'
BERT_MODEL_NAME = 'bert-base-uncased'
WORD2VEC_DIM = 100

# ========== LOAD EMBEDDINGS ==========
print("Loading GloVe embeddings...")
glove_embeddings = load_glove_embeddings(GLOVE_PATH, dim=100)

print("Loading BERT model and tokenizer...")
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
bert_model = BertModel.from_pretrained(BERT_MODEL_NAME)

# ========== DATA LOADING & BALANCING ==========
def load_balanced_yelp_reviews(path, max_samples=MAX_SAMPLES):
    humor, not_humor = [], []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            text = data.get('text', '')
            funny = data.get('funny', 0)
            if funny >= 3:
                humor.append((text, 1))
            else:
                not_humor.append((text, 0))
            if len(humor) >= max_samples // 2 and len(not_humor) >= max_samples // 2:
                break
    humor = humor[:max_samples // 2]
    not_humor = not_humor[:max_samples // 2]
    all_data = humor + not_humor
    np.random.shuffle(all_data)
    reviews, labels = zip(*all_data)
    return list(reviews), list(labels)

reviews, labels = load_balanced_yelp_reviews(DATASET_PATH)
print("Label distribution (overall):", Counter(labels))

# ========== PREPROCESSING ==========
stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [w for w in tokens if w.isalpha() and w not in stop_words]
    return tokens

tokenized_reviews = [preprocess(review) for review in tqdm(reviews, desc="Preprocessing")]

# ========== TRAIN WORD2VEC ==========
print("Training Word2Vec on dataset...")
w2v_model = Word2Vec(sentences=tokenized_reviews, vector_size=WORD2VEC_DIM, window=5, min_count=2, workers=4)

def get_word2vec_embedding(tokens, model, dim=100):
    vectors = [model.wv[w] for w in tokens if w in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(dim)

# ========== ZAGREB INDICES ==========
def compute_zagreb_indices(G, alpha=2):
    degrees = dict(G.degree())
    edges = list(G.edges())
    nodes = list(G.nodes())
    m1 = sum(degrees[u] + degrees[v] for u, v in edges)
    m2 = sum(degrees[u] * degrees[v] for u, v in edges)
    m1_co = sum(degrees[u] + degrees[v] for u in nodes for v in nodes if u != v and not G.has_edge(u, v))
    m2_co = sum(degrees[u] * degrees[v] for u in nodes for v in nodes if u != v and not G.has_edge(u, v))
    generalized = sum(degrees[u]*alpha + degrees[v]*alpha for u, v in edges)
    modified = sum(1/(degrees[u]*degrees[v]) if degrees[u]*degrees[v] != 0 else 0 for u, v in edges)
    third = sum(degrees[v]**3 for v in nodes)
    hyper = sum((degrees[u] + degrees[v])**2 for u, v in edges)
    f_index = sum(degrees[v]**3 for v in nodes)
    return [
        m1, m2, m1_co, m2_co, generalized, modified, third, hyper, f_index
    ]

# ========== WORDNET AMBIGUITY FEATURES ==========
def wordnet_ambiguity_features(tokens):
    synset_counts = [len(wordnet.synsets(w)) for w in tokens]
    if not synset_counts:
        return [0, 0, 0]
    mean_syn = np.mean(synset_counts)
    max_syn = np.max(synset_counts)
    gap_syn = max_syn - np.min(synset_counts)
    return [mean_syn, max_syn, gap_syn]

# ========== GLOVE EMBEDDING ==========
def get_glove_embedding(tokens, embeddings, dim=100):
    vectors = [embeddings[w] for w in tokens if w in embeddings]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(dim)

# ========== BERT EMBEDDING ==========
def get_bert_embedding(text):
    with torch.no_grad():
        encoded = tokenizer(text, return_tensors='pt', truncation=True, max_length=128, padding='max_length')
        output = bert_model(**encoded)
        return output.last_hidden_state[:,0,:].squeeze().numpy()

# ========== FEATURE EXTRACTION ==========
def build_cooccurrence_graph(tokens, window=2):
    G = nx.Graph()
    for i, word in enumerate(tokens):
        for j in range(i+1, min(i+window, len(tokens))):
            G.add_edge(word, tokens[j])
    return G

def extract_features(tokens, text):
    G = build_cooccurrence_graph(tokens)
    zagreb_feats = compute_zagreb_indices(G)
    ambiguity_feats = wordnet_ambiguity_features(tokens)
    length = len(tokens)
    exclam = sum(1 for t in tokens if t == '!')
    question = sum(1 for t in tokens if t == '?')
    glove_feat = get_glove_embedding(tokens, glove_embeddings)
    w2v_feat = get_word2vec_embedding(tokens, w2v_model)
    bert_feat = get_bert_embedding(text)
    return np.concatenate([zagreb_feats, ambiguity_feats, [length, exclam, question], glove_feat, w2v_feat, bert_feat])

features = np.array([
    extract_features(tokens, review)
    for tokens, review in tqdm(zip(tokenized_reviews, reviews), total=len(reviews), desc="Feature Extraction")
])

# ========== EXPORT ZAGREB INDICES FOR SELECTED REVIEWS ==========
zagreb_names = [
    "First Zagreb Index", "Second Zagreb Index",
    "First Zagreb Co-Index", "Second Zagreb Co-Index",
    "Generalized Zagreb Index", "Modified Zagreb Index",
    "Third Zagreb Index", "Hyper Zagreb Index", "Forgotten Index"
]

OUTPUT_FILE = "zagreb_indices_results.txt"
NUM_EXPORT = 10  # Number of reviews to export

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for idx, (tokens, review) in enumerate(zip(tokenized_reviews, reviews)):
        if idx >= NUM_EXPORT:
            break
        G = build_cooccurrence_graph(tokens)
        zagreb = compute_zagreb_indices(G)
        processed_text = review[:100].replace('\n', ' ')  # Handle newlines first
        f.write(f"Review #{idx+1}\n")
        f.write(f"Text Preview: {processed_text}\n")
        for name, value in zip(zagreb_names, zagreb):
            f.write(f"{name}: {value}\n")
        f.write("="*60 + "\n\n")

print(f"Exported Zagreb indices for {NUM_EXPORT} reviews to {OUTPUT_FILE}")

# ========== TRAIN-TEST SPLIT & SCALING ==========
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.3, random_state=42, stratify=labels
)
print("Train label distribution:", Counter(y_train))
print("Test label distribution:", Counter(y_test))

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ========== MODEL TRAINING ==========

# 1. SVM and NB (scikit-learn)
clf_nb = GaussianNB()
clf_svm = SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced')

# 2. MLP (PyTorch, supports Adam and RMSprop)
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024):  # Add input_dim parameter
        super().__init__()  # Initialize parent class
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # Use input_dim here
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.sigmoid(x)

def train_mlp(X_train, y_train, X_val, y_val, optimizer_name='adam', n_epochs=20, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)
    model = SimpleMLP(X_train.shape[1]).to(device)
    criterion = nn.BCELoss()
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=1e-3)
    for epoch in range(n_epochs):
        model.train()
        idx = np.random.permutation(len(X_train))
        for i in range(0, len(X_train), batch_size):
            batch_idx = idx[i:i+batch_size]
            batch_X = X_train[batch_idx]
            batch_y = y_train[batch_idx]
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        # Optionally print loss
    # Validation predictions
    model.eval()
    with torch.no_grad():
        y_pred = model(X_val).cpu().numpy().flatten()
    return (y_pred > 0.5).astype(int)

print("Training SVM and NB...")
clf_nb.fit(X_train, y_train)
clf_svm.fit(X_train, y_train)

print("Training MLP (Adam)...")
mlp_pred_adam = train_mlp(X_train, y_train, X_test, y_test, optimizer_name='adam')

print("Training MLP (RMSprop)...")
mlp_pred_rmsprop = train_mlp(X_train, y_train, X_test, y_test, optimizer_name='rmsprop')

# ========== ENSEMBLE (STACKING) ==========
estimators = [
    ('nb', clf_nb),
    ('svm', clf_svm)
    # MLP not included directly in scikit-learn stacking, but you can ensemble predictions manually
]
stacking = StackingClassifier(
    estimators=estimators,
    final_estimator=RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    passthrough=True
)
print("Training Stacking Ensemble...")
stacking.fit(X_train, y_train)
y_pred_stack = stacking.predict(X_test)

# ========== EVALUATION ==========
print("\n=== EVALUATION (SVM) ===")
y_pred_svm = clf_svm.predict(X_test)
print(classification_report(y_test, y_pred_svm, digits=4))

print("\n=== EVALUATION (NB) ===")
y_pred_nb = clf_nb.predict(X_test)
print(classification_report(y_test, y_pred_nb, digits=4))

print("\n=== EVALUATION (MLP Adam) ===")
print(classification_report(y_test, mlp_pred_adam, digits=4))

print("\n=== EVALUATION (MLP RMSprop) ===")
print(classification_report(y_test, mlp_pred_rmsprop, digits=4))

print("\n=== EVALUATION (Stacking Ensemble) ===")
print(classification_report(y_test, y_pred_stack, digits=4))
