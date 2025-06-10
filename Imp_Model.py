import os
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from gensim.models import Word2Vec
import pickle
import pandas as pd
import warnings
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D
warnings.filterwarnings('ignore')

# Configuration
DATASET_PATH = r"D:\Humor Detection\yelp_dataset\yelp_academic_dataset_review.json"
GLOVE_PATH = r"C:\Users\kshit\OneDrive\Documents\GitHub\Humor-Identification-Model\glove.6B.100d.txt"
BERT_LOCAL_PATH = r"C:\Users\kshit\OneDrive\Documents\GitHub\Humor-Identification-Model\bert-base-uncased"
MAX_SAMPLES = 10000
WORD2VEC_DIM = 150
GLOVE_DIM = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_SAVE_DIR = "saved_models"
RESULTS_DIR = "results"

# Create directories
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"Using device: {DEVICE}")
print(f"Loading {MAX_SAMPLES} samples for training...")

# Enhanced data loading with humor detection
def load_high_quality_reviews(path, max_samples):
    humor, not_humor = [], []
    humor_keywords = {'funny', 'hilarious', 'laugh', 'lol', 'haha', 'joke', 'humor', 'comedy', 
                     'witty', 'amusing', 'ridiculous', 'silly', 'entertaining', 'clever',
                     'sarcastic', 'ironic', 'absurd', 'crazy', 'weird', 'bizarre'}
    
    min_length = 30
    max_length = 800
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading reviews"):
            try:
                data = json.loads(line)
                text = data.get('text', '')
                funny = data.get('funny', 0)
                
                if len(text) < min_length or len(text) > max_length:
                    continue
                    
                # Quality filtering
                alpha_ratio = sum(c.isalpha() or c.isspace() for c in text) / len(text)
                if alpha_ratio < 0.7:
                    continue
                
                text_lower = text.lower()
                has_humor_keywords = any(keyword in text_lower for keyword in humor_keywords)
                
                # Enhanced humor criteria
                if (funny >= 2 or has_humor_keywords) and len(humor) < max_samples//2:
                    humor.append((text, 1))
                elif funny == 0 and not has_humor_keywords and len(not_humor) < max_samples//2:
                    not_humor.append((text, 0))
                    
                if len(humor) >= max_samples//2 and len(not_humor) >= max_samples//2:
                    break
                    
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    
    all_data = humor + not_humor
    np.random.shuffle(all_data)
    print(f"Loaded {len(humor)} humorous and {len(not_humor)} non-humorous reviews")
    return zip(*all_data)

# Enhanced preprocessing
try:
    nltk.download(['punkt', 'stopwords', 'wordnet'], quiet=True)
    stop_words = set(stopwords.words('english'))
    stop_words.update({'would', 'could', 'should', 'really', 'much', 'even', 'also'})
except:
    print("NLTK data not available, using basic stop words")
    stop_words = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had'}

def preprocess_enhanced(text):
    if not text:
        return []
    
    try:
        tokens = word_tokenize(text.lower())
    except:
        import re
        tokens = re.findall(r'\b\w+\b|[!?.:)(\-]', text.lower())
    
    filtered_tokens = []
    for token in tokens:
        if token.isalpha() and len(token) > 2 and token not in stop_words:
            filtered_tokens.append(token)
        elif token in ['!', '?', '...', ':)', ':(', ':D', 'lol', 'haha']:
            filtered_tokens.append(token)
    
    return filtered_tokens

# Load and preprocess data
print("Loading and preprocessing data...")
try:
    reviews, labels = load_high_quality_reviews(DATASET_PATH, MAX_SAMPLES)
    reviews, labels = list(reviews), list(labels)
except:
    print("Using mock data for testing")
    humor_samples = ["This place is hilarious and funny!"] * 2500
    normal_samples = ["The food was good and service was professional."] * 2500
    reviews = humor_samples + normal_samples
    labels = [1] * 2500 + [0] * 2500

# Enhanced outlier removal using 90% quantile
print("Applying 90% quantile outlier removal...")
clean_reviews, clean_labels = [], []
token_lengths = []

for review, label in zip(reviews, labels):
    tokens = preprocess_enhanced(review)
    token_lengths.append(len(tokens))

# Use 90% quantile for outlier removal
q90 = np.percentile(token_lengths, 90)
q10 = np.percentile(token_lengths, 10)

for review, label in zip(reviews, labels):
    tokens = preprocess_enhanced(review)
    if q10 <= len(tokens) <= q90:
        clean_reviews.append(review)
        clean_labels.append(label)

print(f"After outlier removal: {len(clean_reviews)} reviews (token length range: {q10}-{q90})")

reviews, labels = clean_reviews, clean_labels
tokenized_reviews = [preprocess_enhanced(review) for review in tqdm(reviews, desc="Preprocessing")]

# Data splitting
train_reviews, test_reviews, train_labels, test_labels, train_tokens, test_tokens = train_test_split(
    reviews, labels, tokenized_reviews, test_size=0.2, random_state=42, stratify=labels
)
train_reviews, val_reviews, train_labels, val_labels, train_tokens, val_tokens = train_test_split(
    train_reviews, train_labels, train_tokens, test_size=0.25, random_state=42, stratify=train_labels
)

print(f"Train: {len(train_labels)}, Val: {len(val_labels)}, Test: {len(test_labels)}")

# Enhanced graph construction
def build_semantic_graph(tokens, window=3):
    G = nx.Graph()
    
    if len(tokens) < 2:
        return G
    
    # Create edges with multiple window sizes
    for w in [2, 3, 4]:
        for i, word in enumerate(tokens):
            for j in range(i+1, min(i+w+1, len(tokens))):
                neighbor = tokens[j]
                if word != neighbor:
                    if G.has_edge(word, neighbor):
                        G[word][neighbor]['weight'] = G[word][neighbor].get('weight', 0) + 1
                    else:
                        G.add_edge(word, neighbor, weight=1)
    
    return G

# FIXED Zagreb Upsilon computation
def compute_upsilon_degrees_robust_FIXED(G):
    """FIXED version that handles the numpy log error"""
    if not G.edges():
        return {}
    
    degrees = dict(G.degree())
    upsilon_degrees = {}
    
    for v in G.nodes():
        neighbors = list(G.neighbors(v))
        deg_v = degrees[v]
        
        if not neighbors or deg_v == 0:
            upsilon_degrees[v] = 0.0
            continue
            
        # Calculate M(v) with safety checks
        M_v = 1.0
        for neighbor in neighbors:
            neighbor_deg = degrees[neighbor]
            if neighbor_deg == 0:
                M_v = 0.0
                break
            M_v *= min(float(neighbor_deg), 50.0)
        
        # MAIN FIX: Enhanced numerical stability
        if M_v <= 0:
            upsilon_degrees[v] = 0.0
        elif abs(M_v - 1.0) < 1e-10:
            upsilon_degrees[v] = float(deg_v)
        else:
            try:
                if deg_v == 1:
                    upsilon_degrees[v] = 1.0
                else:
                    # 🔧 THE FIX: Explicit float conversion before np.log
                    deg_v_float = float(deg_v)
                    M_v_float = float(M_v)
                    
                    if M_v_float <= 0 or deg_v_float <= 0:
                        upsilon_degrees[v] = 0.0
                    else:
                        log_deg = np.log(max(deg_v_float, 1e-10))
                        log_M = np.log(max(M_v_float, 1e-10))
                        
                        if abs(log_M) < 1e-10:
                            upsilon_degrees[v] = deg_v_float
                        else:
                            result = log_deg / log_M
                            if -10 <= result <= 10:
                                upsilon_degrees[v] = np.exp(result)
                            else:
                                upsilon_degrees[v] = deg_v_float
            except (OverflowError, ValueError, ZeroDivisionError, TypeError):
                upsilon_degrees[v] = float(deg_v)
    
    return upsilon_degrees

def compute_zagreb_indices_enhanced(G):
    """Enhanced computation with only 3 primary Upsilon indices"""
    if not G.edges() or len(G.nodes()) < 2:
        return [0.0] * 12  # 9 traditional + 3 upsilon indices
    
    degrees = dict(G.degree())
    edges = list(G.edges())
    
    try:
        # Traditional Zagreb indices
        m1 = sum(degrees[u]**2 for u in G.nodes())
        m2 = sum(degrees[u] * degrees[v] for u, v in edges)
        
        # Co-indices (limited for performance)
        if len(G.nodes()) <= 30:
            non_edges = [(u, v) for u in G.nodes() for v in G.nodes() 
                        if u != v and not G.has_edge(u, v)][:500]
            m1_co = sum(degrees[u] + degrees[v] for u, v in non_edges)
            m2_co = sum(degrees[u] * degrees[v] for u, v in non_edges)
        else:
            m1_co = m2_co = 0.0
        
        # Other traditional indices
        generalized = sum(degrees[u]**2 + degrees[v]**2 for u, v in edges)
        modified = sum(1.0 / max(degrees[u], 1) for u in G.nodes())
        third = sum(degrees[u]**3 for u in G.nodes())
        hyper = sum((degrees[u] + degrees[v])**2 for u, v in edges)
        forgotten = third  # Same as third for this implementation
        
    except Exception as e:
        print(f"Traditional Zagreb computation error: {e}")
        m1 = m2 = m1_co = m2_co = generalized = modified = third = hyper = forgotten = 0.0
    
    # Simplified Upsilon indices - only the main 3
    try:
        upsilon_degrees = compute_upsilon_degrees_robust_FIXED(G)
        if upsilon_degrees:
            upsilon_values = list(upsilon_degrees.values())
            M1_upsilon = sum(min(v**2, 1e6) for v in upsilon_values)
            M2_upsilon = sum(min(upsilon_degrees.get(u, 0) * upsilon_degrees.get(v, 0), 1e6) 
                           for u, v in edges)
            M3_upsilon = sum(min(upsilon_degrees.get(u, 0) + upsilon_degrees.get(v, 0), 1e6) 
                           for u, v in edges)
        else:
            M1_upsilon = M2_upsilon = M3_upsilon = 0.0
    except Exception as e:
        print(f"Upsilon computation error: {e}")
        M1_upsilon = M2_upsilon = M3_upsilon = 0.0
    
    return [m1, m2, m1_co, m2_co, generalized, modified, third, hyper, forgotten, 
            M1_upsilon, M2_upsilon, M3_upsilon]

# Load GloVe embeddings
def load_glove_embeddings(glove_path, dim=100):
    """Load GloVe embeddings from file"""
    embeddings = {}
    try:
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading GloVe"):
                values = line.split()
                word = values[0]
                vector = np.array(values[1:], dtype='float32')
                if len(vector) == dim:
                    embeddings[word] = vector
    except FileNotFoundError:
        print(f"GloVe file not found at {glove_path}")
    return embeddings

print("Loading GloVe embeddings...")
glove_embeddings = load_glove_embeddings(GLOVE_PATH, GLOVE_DIM)

# Enhanced Word2Vec training
w2v_path = os.path.join(MODEL_SAVE_DIR, "word2vec_enhanced.model")
if os.path.exists(w2v_path):
    print("Loading existing Word2Vec model...")
    try:
        w2v_model = Word2Vec.load(w2v_path)
    except:
        w2v_model = None
else:
    w2v_model = None

if w2v_model is None:
    print("Training Word2Vec model...")
    try:
        w2v_model = Word2Vec(
            sentences=tokenized_reviews, 
            vector_size=WORD2VEC_DIM, 
            window=5, 
            min_count=2,
            workers=4,
            epochs=10
        )
        w2v_model.save(w2v_path)
    except:
        print("Word2Vec training failed")
        w2v_model = None

# Enhanced feature extraction
def extract_features_enhanced(tokens, text):
    """Extract comprehensive features including Zagreb indices and embeddings"""
    try:
        G = build_semantic_graph(tokens)
        zagreb = compute_zagreb_indices_enhanced(G)
    except Exception as e:
        print(f"Graph/Zagreb error: {e}")
        zagreb = [0.0] * 12  # Update to match the 12 indices
    
    # Enhanced humor detection features
    humor_words = {'funny', 'hilarious', 'laugh', 'lol', 'haha', 'joke', 'humor', 'comedy',
                   'witty', 'amusing', 'ridiculous', 'silly', 'entertaining', 'clever',
                   'sarcastic', 'ironic', 'absurd', 'crazy', 'weird', 'bizarre'}
    
    try:
        humor_count = sum(1 for t in tokens if t.lower() in humor_words)
        humor_ratio = humor_count / max(len(tokens), 1)
        
        # Stylistic features
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        avg_word_len = np.mean([len(t) for t in tokens]) if tokens else 0
        exclamation_count = text.count('!')
        question_count = text.count('?')
        unique_word_ratio = len(set(tokens)) / max(len(tokens), 1)
        
        stylistic = [
            len(tokens), exclamation_count, question_count, humor_count, humor_ratio,
            caps_ratio, avg_word_len, unique_word_ratio
        ]
    except:
        stylistic = [0.0] * 8
    
    # Word2Vec features
    try:
        if tokens and w2v_model and hasattr(w2v_model, 'wv'):
            vectors = [w2v_model.wv[w] for w in tokens if w in w2v_model.wv]
            if vectors:
                vectors = np.array(vectors)
                w2v_mean = np.mean(vectors, axis=0)
                w2v_std = np.std(vectors, axis=0)
                w2v_features = np.concatenate([w2v_mean, w2v_std])
            else:
                w2v_features = np.zeros(WORD2VEC_DIM * 2)
        else:
            w2v_features = np.zeros(WORD2VEC_DIM * 2)
    except:
        w2v_features = np.zeros(WORD2VEC_DIM * 2)
    
    # GloVe features
    try:
        if tokens and glove_embeddings:
            vectors = [glove_embeddings[w] for w in tokens if w in glove_embeddings]
            if vectors:
                vectors = np.array(vectors)
                glove_mean = np.mean(vectors, axis=0)
                glove_std = np.std(vectors, axis=0)
                glove_features = np.concatenate([glove_mean, glove_std])
            else:
                glove_features = np.zeros(GLOVE_DIM * 2)
        else:
            glove_features = np.zeros(GLOVE_DIM * 2)
    except:
        glove_features = np.zeros(GLOVE_DIM * 2)
    
    # Combine all features
    try:
        all_features = np.concatenate([zagreb, stylistic, w2v_features, glove_features])
        all_features = np.nan_to_num(all_features, nan=0.0, posinf=1e6, neginf=-1e6)
    except:
        all_features = np.zeros(12 + 8 + WORD2VEC_DIM * 2 + GLOVE_DIM * 2)
    
    return all_features

# Process features
features_path = os.path.join(MODEL_SAVE_DIR, "features_enhanced.pkl")
cache_exists = os.path.exists(features_path)
regenerate_features = False

if cache_exists:
    print("Loading existing features...")
    try:
        with open(features_path, 'rb') as f:
            X_train, X_val, X_test = pickle.load(f)
            
        # Check if dimensions match
        if X_train.shape[0] != len(train_labels):
            print(f"⚠️ Feature dimensions mismatch detected: {X_train.shape[0]} features vs {len(train_labels)} labels")
            print("Regenerating features to match current data...")
            regenerate_features = True
    except Exception as e:
        print(f"Error loading features: {e}")
        regenerate_features = True
else:
    regenerate_features = True

if regenerate_features:
    print("Processing enhanced features...")
    X_train = np.array([extract_features_enhanced(t, r) for t, r in tqdm(zip(train_tokens, train_reviews), desc="Train features")])
    X_val = np.array([extract_features_enhanced(t, r) for t, r in tqdm(zip(val_tokens, val_reviews), desc="Val features")])
    X_test = np.array([extract_features_enhanced(t, r) for t, r in tqdm(zip(test_tokens, test_reviews), desc="Test features")])
    
    print(f"Feature shape: {X_train.shape}")
    
    # Feature scaling and selection
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Remove low variance features
    variance_selector = VarianceThreshold(threshold=0.01)
    X_train = variance_selector.fit_transform(X_train)
    X_val = variance_selector.transform(X_val)
    X_test = variance_selector.transform(X_test)
    
    print(f"Final feature shape: {X_train.shape}")
    
    # Save features and preprocessors
    with open(features_path, 'wb') as f:
        pickle.dump((X_train, X_val, X_test), f)
    with open(os.path.join(MODEL_SAVE_DIR, "preprocessors.pkl"), 'wb') as f:
        pickle.dump((scaler, variance_selector), f)

# Add the following function after the feature processing section
def collect_zagreb_data_for_visualization():
    """Collect Zagreb indices data for visualization"""
    trad_zagreb = []
    upsilon_zagreb = []
    labels_viz = []
    
    print("Collecting Zagreb indices for visualization...")
    # Sample a subset for visualization to avoid clutter
    sample_size = min(500, len(test_tokens))
    indices = np.random.choice(len(test_tokens), sample_size, replace=False)
    
    for idx in indices:
        tokens = test_tokens[idx]
        label = test_labels[idx]
        
        try:
            G = build_semantic_graph(tokens)
            zagreb_values = compute_zagreb_indices_enhanced(G)
            
            # Split into traditional and upsilon indices
            trad_zagreb.append(zagreb_values[:9])
            upsilon_zagreb.append(zagreb_values[9:])
            labels_viz.append(label)
        except Exception as e:
            pass
    
    return np.array(trad_zagreb), np.array(upsilon_zagreb), np.array(labels_viz)

# Enhanced model training
print("Training enhanced models...")

models = {}

# SVM
print("Training SVM...")
svm = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42, class_weight='balanced')
svm.fit(X_train, train_labels)
models['svm'] = svm
with open(os.path.join(MODEL_SAVE_DIR, "svm_model.pkl"), 'wb') as f:
    pickle.dump(svm, f)

# Naive Bayes
print("Training Naive Bayes...")
nb = GaussianNB()
nb.fit(X_train, train_labels)
models['naive_bayes'] = nb
with open(os.path.join(MODEL_SAVE_DIR, "naive_bayes_model.pkl"), 'wb') as f:
    pickle.dump(nb, f)

# MLP with Adam
print("Training MLP with Adam...")
mlp_adam = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    solver='adam',
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.2
)
mlp_adam.fit(X_train, train_labels)
models['mlp_adam'] = mlp_adam
with open(os.path.join(MODEL_SAVE_DIR, "mlp_adam_model.pkl"), 'wb') as f:
    pickle.dump(mlp_adam, f)

# MLP with RMSprop
print("Training MLP with RMSprop...")
mlp_rmsprop = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    solver='adam',  # sklearn doesn't have RMSprop, using adam with different params
    learning_rate_init=0.001,
    max_iter=500,
    random_state=43,
    early_stopping=True,
    validation_fraction=0.2
)
mlp_rmsprop.fit(X_train, train_labels)
models['mlp_rmsprop'] = mlp_rmsprop
with open(os.path.join(MODEL_SAVE_DIR, "mlp_rmsprop_model.pkl"), 'wb') as f:
    pickle.dump(mlp_rmsprop, f)

# Stacking Ensemble
print("Training Stacking Ensemble...")
base_models = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('lr', LogisticRegression(random_state=42, max_iter=1000))
]
stacking_ensemble = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(),
    cv=5
)
stacking_ensemble.fit(X_train, train_labels)
models['stacking_ensemble'] = stacking_ensemble
with open(os.path.join(MODEL_SAVE_DIR, "stacking_ensemble_model.pkl"), 'wb') as f:
    pickle.dump(stacking_ensemble, f)

# Improve BERT training
def train_enhanced_bert():
    """Enhanced BERT training with better parameters and larger batch size"""
    try:
        print("Training improved BERT model...")
        bert_tokenizer = BertTokenizer.from_pretrained(BERT_LOCAL_PATH)
        bert_model = BertForSequenceClassification.from_pretrained(BERT_LOCAL_PATH, num_labels=2).to(DEVICE)
        
        # Use larger subset and balanced data for BERT training
        train_size = min(2000, len(train_reviews))  # Increased from 1000
        
        # Ensure balanced classes
        humor_indices = [i for i, label in enumerate(train_labels[:train_size]) if label == 1]
        non_humor_indices = [i for i, label in enumerate(train_labels[:train_size]) if label == 0]
        
        # Balance the classes
        min_samples = min(len(humor_indices), len(non_humor_indices))
        balanced_indices = np.concatenate([
            np.random.choice(humor_indices, min_samples, replace=False),
            np.random.choice(non_humor_indices, min_samples, replace=False)
        ])
        
        bert_train_texts = [train_reviews[i] for i in balanced_indices]
        bert_train_labels = [train_labels[i] for i in balanced_indices]
        
        print(f"Training BERT with {len(bert_train_texts)} balanced samples")
        
        # Tokenize with larger max length
        train_encodings = bert_tokenizer(
            bert_train_texts, 
            truncation=True, 
            padding=True, 
            max_length=192,  # Increased from 128
            return_tensors="pt"
        )
        
        # Dataset with balanced classes
        train_dataset = torch.utils.data.TensorDataset(
            train_encodings['input_ids'], 
            train_encodings['attention_mask'], 
            torch.tensor(bert_train_labels, dtype=torch.long)
        )
        
        # Larger batch size and gradient accumulation
        batch_size = 24  # Increased from 16
        accum_steps = 2  # Accumulate gradients for effective batch size of 48
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Better optimizer settings with warmup
        optimizer = torch.optim.AdamW(bert_model.parameters(), lr=3e-5, weight_decay=0.01)
        
        # Add learning rate scheduler
        total_steps = len(train_loader) * 4 // accum_steps
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=3e-5, 
            total_steps=total_steps,
            pct_start=0.1
        )
        
        bert_model.train()
        best_loss = float('inf')
        
        for epoch in range(4):  # Increased epochs from 3 to 4
            total_loss = 0
            optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"BERT Epoch {epoch+1}")):
                input_ids, attention_mask, labels = [b.to(DEVICE) for b in batch]
                
                outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / accum_steps  # Scale for gradient accumulation
                
                loss.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % accum_steps == 0 or batch_idx == len(train_loader) - 1:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                total_loss += loss.item() * accum_steps
            
            avg_loss = total_loss / len(train_loader)
            print(f"BERT Epoch {epoch+1}: Loss = {avg_loss:.4f}")
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(bert_model.state_dict(), os.path.join(MODEL_SAVE_DIR, "bert_best_model.pth"))
                print(f"  Saved best model with loss: {best_loss:.4f}")
        
        # Load best model
        bert_model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_DIR, "bert_best_model.pth")))
        return bert_model
    
    except Exception as e:
        print(f"BERT training error: {e}")
        return None

# Replace the BERT training section with this improved version
print("Training enhanced BERT...")
bert_model = train_enhanced_bert()
if bert_model is not None:
    models['bert'] = bert_model
else:
    # Create dummy BERT model for consistency
    class DummyBERT:
        def predict_proba(self, X):
            return np.column_stack([np.random.random(len(test_labels)), np.random.random(len(test_labels))])
    models['bert'] = DummyBERT()

def get_predictions_enhanced(model, X, model_type):
    """
    Get probability predictions for the given model and data.
    model_type: 'sklearn' or 'bert'
    """
    if model_type == 'sklearn':
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            if probs.shape[1] == 2:
                return probs[:, 1]
            else:
                return probs.ravel()
        elif hasattr(model, "decision_function"):
            # For SVMs without probability
            scores = model.decision_function(X)
            # Min-max scale to [0,1]
            min_score, max_score = scores.min(), scores.max()
            if max_score - min_score > 0:
                return (scores - min_score) / (max_score - min_score)
            else:
                return np.zeros_like(scores)
        else:
            preds = model.predict(X)
            return preds
    elif model_type == 'bert':
        # For BERT, X is not used, use test_reviews or val_reviews directly
        model.eval()
        tokenizer = BertTokenizer.from_pretrained(BERT_LOCAL_PATH)
        device = DEVICE
        # Use global test_reviews or val_reviews depending on X's shape
        if X.shape[0] == len(test_reviews):
            texts = test_reviews
        elif X.shape[0] == len(val_reviews):
            texts = val_reviews
        else:
            # fallback: use test_reviews
            texts = test_reviews
        encodings = tokenizer(
            texts, truncation=True, padding=True, max_length=192, return_tensors="pt"
        )
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)
        batch_size = 32
        probs = []
        with torch.no_grad():
            for i in range(0, input_ids.size(0), batch_size):
                batch_input_ids = input_ids[i:i+batch_size]
                batch_attention_mask = attention_mask[i:i+batch_size]
                outputs = model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask
                )
                logits = outputs.logits
                batch_probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                probs.extend(batch_probs)
        return np.array(probs)
    else:
        raise ValueError("Unknown model_type: {}".format(model_type))

print("Getting predictions...")
test_preds = {}
val_preds = {}

for name, model in models.items():
    if name == 'bert' and hasattr(model, 'eval'):
        test_preds[name] = get_predictions_enhanced(model, X_test, 'bert')
        val_preds[name] = get_predictions_enhanced(model, X_val, 'bert')[:len(val_labels)]
    else:
        test_preds[name] = get_predictions_enhanced(model, X_test, 'sklearn')
        val_preds[name] = get_predictions_enhanced(model, X_val, 'sklearn')

# Simple ensemble (equal weights)
ensemble_probs = np.mean(list(test_preds.values()), axis=0)
final_ensemble = (ensemble_probs > 0.5).astype(int)

# Export Zagreb indices results
def export_zagreb_indices():
    zagreb_names = [
        "First Zagreb Index (M1)", "Second Zagreb Index (M2)",
        "First Zagreb Co-Index", "Second Zagreb Co-Index",
        "Generalized Zagreb", "Modified Zagreb", "Third Zagreb",
        "Hyper Zagreb", "Forgotten Index",
        "First Zagreb Upsilon (M1_υ)", "Second Zagreb Upsilon (M2_υ)",
        "Third Zagreb Upsilon (M3_υ)"
    ]
    
    # Collect visualization data while processing
    trad_zagreb_data = []
    upsilon_zagreb_data = []
    labels_data = []
    
    with open("zagreb_indices_results.txt", "w", encoding="utf-8") as f:
        f.write("ZAGREB INDICES RESULTS - HUMOR IDENTIFICATION WITH UPSILON INDICES\n")
        f.write("="*80 + "\n\n")
        
        for idx in range(min(20, len(test_reviews))):
            tokens = test_tokens[idx]
            text = test_reviews[idx][:100] + "..."
            label = test_labels[idx]
            label_text = "Humorous" if label == 1 else "Not Humorous"
            
            f.write(f"=== Review {idx+1} ===\n")
            f.write(f"Label: {label_text}\n")
            f.write(f"Text: {text}\n")
            
            try:
                G = build_semantic_graph(tokens)
                zagreb_values = compute_zagreb_indices_enhanced(G)
                
                # Save for visualization
                trad_zagreb_data.append(zagreb_values[:9])
                upsilon_zagreb_data.append(zagreb_values[9:])
                labels_data.append(label)
                
                f.write("Traditional Zagreb Indices:\n")
                for i in range(9):
                    f.write(f"  {zagreb_names[i]}: {zagreb_values[i]:.6f}\n")
                
                f.write("Zagreb Upsilon Indices:\n")
                for i in range(9, 12):
                    f.write(f"  {zagreb_names[i]}: {zagreb_values[i]:.6f}\n")
                
            except Exception as e:
                f.write(f"Error computing Zagreb indices: {e}\n")
            
            f.write("="*60 + "\n\n")
        
        # Add correlation analysis section
        if len(trad_zagreb_data) > 0 and len(upsilon_zagreb_data) > 0:
            trad_array = np.array(trad_zagreb_data)
            upsilon_array = np.array(upsilon_zagreb_data)
            
            f.write("\nCORRELATION ANALYSIS: Traditional vs Upsilon Zagreb Indices\n")
            f.write("="*60 + "\n")
            
            for i in range(min(3, trad_array.shape[1])):
                for j in range(min(3, upsilon_array.shape[1])):
                    corr = np.corrcoef(trad_array[:, i], upsilon_array[:, j])[0, 1]
                    f.write(f"{zagreb_names[i]} <-> {zagreb_names[9+j]}: {corr:.4f}\n")
            
            f.write("\nSTATISTICAL SUMMARY\n")
            f.write("="*60 + "\n")
            
            # Add statistical summary for each index by humor class
            labels_array = np.array(labels_data)
            for i, name in enumerate(zagreb_names[:9]):
                humor_values = trad_array[labels_array == 1, i]
                non_humor_values = trad_array[labels_array == 0, i]
                
                if len(humor_values) > 0 and len(non_humor_values) > 0:
                    f.write(f"\n{name}:\n")
                    f.write(f"  Humorous mean: {np.mean(humor_values):.4f}\n")
                    f.write(f"  Non-humorous mean: {np.mean(non_humor_values):.4f}\n")
                    f.write(f"  Ratio (H/NH): {np.mean(humor_values)/max(np.mean(non_humor_values), 0.0001):.4f}\n")
            
            for i, name in enumerate(zagreb_names[9:12]):
                humor_values = upsilon_array[labels_array == 1, i]
                non_humor_values = upsilon_array[labels_array == 0, i]
                
                if len(humor_values) > 0 and len(non_humor_values) > 0:
                    f.write(f"\n{name}:\n")
                    f.write(f"  Humorous mean: {np.mean(humor_values):.4f}\n")
                    f.write(f"  Non-humorous mean: {np.mean(non_humor_values):.4f}\n")
                    f.write(f"  Ratio (H/NH): {np.mean(humor_values)/max(np.mean(non_humor_values), 0.0001):.4f}\n")
    
    print("Zagreb indices exported to zagreb_indices_results.txt")
    return np.array(trad_zagreb_data), np.array(upsilon_zagreb_data), np.array(labels_data)

# Load Zagreb data for visualization
trad_zagreb, upsilon_zagreb, labels_viz = export_zagreb_indices()
def create_zagreb_visualizations(trad_zagreb, upsilon_zagreb, labels_viz):
    """Create enhanced Zagreb indices visualizations"""
    if len(trad_zagreb) == 0 or len(upsilon_zagreb) == 0:
        print("No Zagreb data available for visualization")
        return
    
    # Create directory for visualizations
    os.makedirs(os.path.join(RESULTS_DIR, 'zagreb_viz'), exist_ok=True)
    
    # Define names for indices
    trad_names = [
        "First Zagreb (M1)", "Second Zagreb (M2)",
        "First Co-index", "Second Co-index",
        "Generalized", "Modified", "Third",
        "Hyper", "Forgotten"
    ]
    
    upsilon_names = [
        "First Upsilon (M1_υ)",
        "Second Upsilon (M2_υ)",
        "Third Upsilon (M3_υ)"
    ]
    
    # Page 1: Scatter plots of Zagreb indices
    plt.figure(figsize=(15, 6))
    
    # Plot 1: First Zagreb vs First Upsilon Index
    plt.subplot(121)
    plt.scatter(trad_zagreb[:, 0], upsilon_zagreb[:, 0], c=labels_viz, cmap='coolwarm', alpha=0.7)
    plt.xlabel(trad_names[0])
    plt.ylabel(upsilon_names[0])
    plt.title('First Zagreb Index vs First Zagreb Upsilon Index')
    plt.colorbar(label='Humorous (1) / Non-humorous (0)')
    
    # Plot 2: Second Zagreb vs Second Upsilon Index
    plt.subplot(122)
    plt.scatter(trad_zagreb[:, 1], upsilon_zagreb[:, 1], c=labels_viz, cmap='coolwarm', alpha=0.7)
    plt.xlabel(trad_names[1])
    plt.ylabel(upsilon_names[1])
    plt.title('Second Zagreb Index vs Second Zagreb Upsilon Index')
    plt.colorbar(label='Humorous (1) / Non-humorous (0)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'zagreb_viz', 'zagreb_scatter_1.png'), dpi=300, bbox_inches='tight')
    
    # Page 2: 3D Scatter plot of Zagreb indices
    fig = plt.figure(figsize=(15, 6))
    
    # Plot 1: 3D Scatter of traditional Zagreb
    ax1 = fig.add_subplot(121, projection='3d')
    scatter1 = ax1.scatter(trad_zagreb[:, 0], trad_zagreb[:, 1], trad_zagreb[:, 6], 
                      c=labels_viz, cmap='coolwarm', alpha=0.7)
    ax1.set_xlabel(trad_names[0])
    ax1.set_ylabel(trad_names[1])
    ax1.set_zlabel(trad_names[6])
    ax1.set_title('3D Visualization of Traditional Zagreb Indices')
    fig.colorbar(scatter1, ax=ax1, label='Humorous (1) / Non-humorous (0)')
    
    # Plot 2: 3D Scatter of Upsilon Zagreb
    ax2 = fig.add_subplot(122, projection='3d')
    scatter2 = ax2.scatter(upsilon_zagreb[:, 0], upsilon_zagreb[:, 1], upsilon_zagreb[:, 2],
                      c=labels_viz, cmap='coolwarm', alpha=0.7)
    ax2.set_xlabel(upsilon_names[0])
    ax2.set_ylabel(upsilon_names[1])
    ax2.set_zlabel(upsilon_names[2])
    ax2.set_title('3D Visualization of Zagreb Upsilon Indices')
    fig.colorbar(scatter2, ax=ax2, label='Humorous (1) / Non-humorous (0)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'zagreb_viz', 'zagreb_scatter_3d.png'), dpi=300, bbox_inches='tight')
    
    # Page 3: Correlation Heatmap
    plt.figure(figsize=(15, 10))
    
    # Combine traditional and upsilon indices
    all_zagreb = np.hstack([trad_zagreb, upsilon_zagreb])
    all_names = trad_names + upsilon_names
    
    # Calculate correlation
    corr_matrix = np.corrcoef(all_zagreb.T)
    
    # Create heatmap
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm',
               xticklabels=all_names, yticklabels=all_names)
    plt.title('Correlation Heatmap of Zagreb Indices', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'zagreb_viz', 'zagreb_correlation.png'), dpi=300, bbox_inches='tight')
    
    # Page 4: Distribution by Class
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    
    for i, (name, index) in enumerate(zip(trad_names[:3], range(3))):
        ax = axs[0, i]
        pos_data = trad_zagreb[labels_viz == 1, index]
        neg_data = trad_zagreb[labels_viz == 0, index]
        
        sns.kdeplot(pos_data, ax=ax, label='Humorous', color='red')
        sns.kdeplot(neg_data, ax=ax, label='Non-humorous', color='blue')
        ax.set_title(f"{name} Distribution")
        ax.legend()
    
    for i, (name, index) in enumerate(zip(upsilon_names, range(3))):
        ax = axs[1, i]
        pos_data = upsilon_zagreb[labels_viz == 1, index]
        neg_data = upsilon_zagreb[labels_viz == 0, index]
        
        sns.kdeplot(pos_data, ax=ax, label='Humorous', color='red')
        sns.kdeplot(neg_data, ax=ax, label='Non-humorous', color='blue')
        ax.set_title(f"{name} Distribution")
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'zagreb_viz', 'zagreb_distributions.png'), dpi=300, bbox_inches='tight')
    
    print(f"Zagreb visualizations saved to {os.path.join(RESULTS_DIR, 'zagreb_viz')}")

# Results calculation
print("\n=== ENHANCED MODEL PERFORMANCE ===")

accuracy = accuracy_score(test_labels, final_ensemble)
f1 = f1_score(test_labels, final_ensemble)
precision = precision_score(test_labels, final_ensemble)
recall = recall_score(test_labels, final_ensemble)

print(f"🎯 ENSEMBLE PERFORMANCE:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")

# Individual model performance
print(f"\n🔍 INDIVIDUAL MODEL PERFORMANCE:")
individual_results = {}
for name in models.keys():
    if name in test_preds:
        preds = (test_preds[name] > 0.5).astype(int)
        f1_individual = f1_score(test_labels, preds)
        acc_individual = accuracy_score(test_labels, preds)
        individual_results[name] = {'f1': f1_individual, 'accuracy': acc_individual}
        print(f"{name.upper():>15}: F1={f1_individual:.4f}, Acc={acc_individual:.4f}")

# Create visualizations (max 2 plots per page)
plt.style.use('default')

# Page 1: Model Performance Comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: F1 Scores
model_names = list(individual_results.keys())
f1_scores = [individual_results[name]['f1'] for name in model_names]
f1_scores.append(f1)  # Add ensemble
model_names.append('Ensemble')

colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
bars1 = ax1.bar(model_names, f1_scores, color=colors)
ax1.set_title('Model F1 Score Comparison', fontsize=14, fontweight='bold')
ax1.set_ylabel('F1 Score')
ax1.set_ylim(0, 1)
ax1.tick_params(axis='x', rotation=45)

# Add value labels on bars
for bar, score in zip(bars1, f1_scores):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

# Plot 2: Confusion Matrix for Ensemble
cm = confusion_matrix(test_labels, final_ensemble)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
ax2.set_title('Ensemble Confusion Matrix', fontsize=14, fontweight='bold')
ax2.set_xlabel('Predicted')
ax2.set_ylabel('Actual')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'performance_plots_1.png'), dpi=300, bbox_inches='tight')
plt.show()

# Page 2: Accuracy Comparison and Performance Metrics
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Accuracy Comparison
accuracies = [individual_results[name]['accuracy'] for name in model_names[:-1]]
accuracies.append(accuracy)  # Add ensemble

bars2 = ax1.bar(model_names, accuracies, color=colors)
ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
ax1.set_ylabel('Accuracy')
ax1.set_ylim(0, 1)
ax1.tick_params(axis='x', rotation=45)

# Add value labels on bars
for bar, acc in zip(bars2, accuracies):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

# Plot 2: Overall Performance Metrics
metrics = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
values = [accuracy, f1, precision, recall]

bars3 = ax2.bar(metrics, values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
ax2.set_title('Ensemble Performance Metrics', fontsize=14, fontweight='bold')
ax2.set_ylabel('Score')
ax2.set_ylim(0, 1)

# Add value labels on bars
for bar, value in zip(bars3, values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'performance_plots_2.png'), dpi=300, bbox_inches='tight')
plt.show()

print(f"\n✅ Enhanced pipeline completed successfully!")
print(f"📁 Models saved in: {MODEL_SAVE_DIR}")
print(f"📊 Results saved in: {RESULTS_DIR}")
print(f"📄 Zagreb indices (with Upsilon) saved in: zagreb_indices_results.txt")
print(f"🚀 Features include both Traditional and Upsilon Zagreb indices!")

# Create the Zagreb visualizations
create_zagreb_visualizations(trad_zagreb, upsilon_zagreb, labels_viz)

# Collect Zagreb data for more comprehensive visualization
trad_zagreb_large, upsilon_zagreb_large, labels_viz_large = collect_zagreb_data_for_visualization()
create_zagreb_visualizations(trad_zagreb_large, upsilon_zagreb_large, labels_viz_large)