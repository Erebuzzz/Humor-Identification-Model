## Section 1: Environment Setup & Imports

import pandas as pd
import numpy as np
import json
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import nltk
import re
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import os

# Create results directory
os.makedirs("results", exist_ok=True)

print("Loading NLP resources...")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load spaCy model (medium-sized English model with word vectors)
print("Loading spaCy model with word vectors...")
try:
    nlp = spacy.load("en_core_web_md")
    print("SpaCy model loaded successfully")
except:
    print("SpaCy model not found. Installing...")
    import subprocess
    subprocess.call("python -m spacy download en_core_web_md", shell=True)
    nlp = spacy.load("en_core_web_md")
    print("SpaCy model installed and loaded")

## Section 2: Extract and Read Yelp Review JSON

def load_yelp_reviews(path, limit=20000):
    data = []
    print(f"Loading data from {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, total=limit)):
            if i >= limit:
                break
            try:
                review = json.loads(line)
                data.append({
                    'review_id': review['review_id'],
                    'text': review['text'],
                    'funny': review['funny']
                })
            except:
                continue
    df = pd.DataFrame(data)
    df['label'] = df['funny'].apply(lambda x: 1 if x > 0 else 0)
    
    # Print class distribution
    print(f"Original class distribution: {df['label'].value_counts().to_dict()}")
    return df

reviews_df = load_yelp_reviews('D:\Humor Detection\yelp_dataset\yelp_academic_dataset_review.json', limit=10000)

# Balance the dataset
humor_df = reviews_df[reviews_df['label'] == 1]
not_humor_df = reviews_df[reviews_df['label'] == 0].sample(n=humor_df.shape[0], random_state=42)
reviews_df = pd.concat([humor_df, not_humor_df]).sample(frac=1, random_state=42)
print(f"Balanced class distribution: {reviews_df['label'].value_counts().to_dict()}")

## Section 3: Text Preprocessing

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Keep some punctuation that might be relevant for humor
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s!?.,]', '', text)
    tokens = nltk.word_tokenize(text)
    # Keep humor-indicator words even if they are stopwords
    humor_indicators = {'lol', 'haha', 'funny', 'hilarious', 'laugh'}
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words or w in humor_indicators]
    return tokens

print("Preprocessing text...")
reviews_df['tokens'] = reviews_df['text'].apply(preprocess_text)

## Section 4: Graph Construction & Zagreb Indices

def compute_zagreb_indices(tokens, k=3):
    """Compute all 9 Zagreb indices with proper normalization and error handling."""
    if not tokens or len(tokens) < 2:
        return [0] * 9
        
    try:
        G = nx.Graph()
        G.add_nodes_from(tokens)
        for i in range(len(tokens)-1):
            for j in range(i+1, min(i+k+1, len(tokens))):
                if G.has_edge(tokens[i], tokens[j]):
                    G[tokens[i]][tokens[j]]['weight'] += 1
                else:
                    G.add_edge(tokens[i], tokens[j], weight=1)

        degrees = dict(G.degree())
        
        if not G.edges() or not degrees:
            return [0] * 9
            
        # Calculate Zagreb indices
        M1 = sum([deg**2 for deg in degrees.values()])
        M2 = sum([degrees[u]*degrees[v] for u,v in G.edges()])
        
        # For large graphs, sample non-edges to prevent excessive computation
        if len(G.nodes()) > 30:
            nodes = list(G.nodes())
            sampled = np.random.choice(nodes, min(30, len(nodes)), replace=False)
            M1_co = sum([degrees[u] + degrees[v] for u in sampled for v in sampled 
                      if u != v and not G.has_edge(u, v)])
            M2_co = sum([degrees[u]*degrees[v] for u in sampled for v in sampled 
                      if u != v and not G.has_edge(u, v)])
            # Scale to approximate full graph
            scaling = (len(nodes)**2) / (len(sampled)**2)
            M1_co *= scaling
            M2_co *= scaling
        else:
            M1_co = sum([degrees[u] + degrees[v] for u in G.nodes() for v in G.nodes() 
                       if u != v and not G.has_edge(u, v)])
            M2_co = sum([degrees[u]*degrees[v] for u in G.nodes() for v in G.nodes() 
                       if u != v and not G.has_edge(u, v)])
        
        generalized_zagreb = sum([deg**k for deg in degrees.values()])
        modified_zagreb = sum([1/(degrees[u]*degrees[v]) if degrees[u]*degrees[v] != 0 else 0 
                              for u,v in G.edges()])
        M3 = sum([abs(degrees[u]-degrees[v]) for u,v in G.edges()])
        HM = sum([(degrees[u]+degrees[v])**2 for u,v in G.edges()])
        F = sum([deg**3 for deg in degrees.values()])
        
        # Normalize by graph size to prevent scale issues
        n_nodes = max(1, len(G.nodes()))
        n_edges = max(1, len(G.edges()))
        
        # Use log transformation to handle extreme values
        indices = [
            np.log1p(M1/n_nodes),
            np.log1p(M2/n_edges),
            np.log1p(M1_co/(n_nodes**2)),
            np.log1p(M2_co/(n_nodes**2)),
            np.log1p(generalized_zagreb/n_nodes),
            np.log1p(modified_zagreb/n_edges),
            np.log1p(M3/n_edges),
            np.log1p(HM/n_edges),
            np.log1p(F/n_nodes)
        ]
        
        # Replace any invalid values with zeros
        indices = [0 if np.isnan(x) or np.isinf(x) else x for x in indices]
        
        return indices
    except Exception as e:
        print(f"Error in compute_zagreb_indices: {e}")
        return [0] * 9

## Section 5: Word Embedding using spaCy

def get_embedding_vector(tokens):
    """Get word embeddings using spaCy."""
    if not tokens:
        return np.zeros(300)  # spaCy's vectors are 300-dimensional
    
    vectors = []
    for token in tokens:
        # Get vector from spaCy's vocabulary
        token_obj = nlp.vocab[token]
        if token_obj.has_vector:
            vectors.append(token_obj.vector)
    
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(300)

## Section 6: Feature Engineering (Fusion)

print("Extracting features...")
features = []
labels = []

for idx, row in tqdm(reviews_df.iterrows(), total=reviews_df.shape[0]):
    # Get Zagreb indices
    zagreb = compute_zagreb_indices(row['tokens'])
    
    # Get word embeddings
    emb = get_embedding_vector(row['tokens'])
    
    # Extract additional humor-specific features
    exclamation_count = row['text'].count('!')
    question_count = row['text'].count('?')
    laughter_words = {'haha', 'hahaha', 'lol', 'rofl', 'lmao', 'funny', 'hilarious'}
    laughter_count = sum(1 for token in row['tokens'] if token.lower() in laughter_words)
    caps_ratio = sum(1 for c in row['text'] if c.isupper()) / max(len(row['text']), 1)
    
    humor_features = [exclamation_count, question_count, laughter_count, caps_ratio]
    
    # Combine all features
    fused = np.concatenate([zagreb, emb, humor_features])
    features.append(fused)
    labels.append(row['label'])

X = np.array(features)
y = np.array(labels)

print(f"Feature vector shape: {X.shape}")

## Section 7: Model Training & Evaluation

# Scale features for better model performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print("Training models...")
svc = SVC(probability=True, C=1.0, kernel='rbf', gamma='scale', random_state=42)
mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, alpha=0.0001, random_state=42)
rf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)

estimators = [('svc', svc), ('mlp', mlp), ('rf', rf)]
ensemble = StackingClassifier(estimators=estimators, final_estimator=RandomForestClassifier(n_estimators=100), cv=5)

ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)
y_pred_proba = ensemble.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("\n====== RESULTS =======")
print(f"Accuracy:  {accuracy:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

## Section 8: Visualization & Zagreb Indices Importance

# Zagreb indices names
zagreb_names = [
    'M1 (First Zagreb)', 
    'M2 (Second Zagreb)', 
    'M1_co (First Zagreb Co-index)', 
    'M2_co (Second Zagreb Co-index)',
    'Gen_Zagreb (Generalized Zagreb)', 
    'Mod_Zagreb (Modified Zagreb)', 
    'M3 (Third Zagreb)', 
    'HM (Hyper Zagreb)', 
    'F (Forgotten Index)'
]

# Analyze Zagreb indices importance using the RF component of the ensemble
rf_model = [est[1] for est in ensemble.estimators_ if est[0] == 'rf'][0]
feature_importances = rf_model.feature_importances_
zagreb_importances = feature_importances[:9]  # First 9 features are Zagreb indices

# Sort indices by importance
sorted_idx = np.argsort(zagreb_importances)
sorted_names = [zagreb_names[i] for i in sorted_idx]
sorted_importances = zagreb_importances[sorted_idx]

# Create visualizations
plt.figure(figsize=(12, 8))
plt.barh(sorted_names, sorted_importances, color='teal')
for i, v in enumerate(sorted_importances):
    plt.text(v + 0.01, i, f"{v:.4f}")
plt.xlabel('Feature Importance')
plt.title('Zagreb Indices Importance for Humor Detection')
plt.tight_layout()
plt.savefig(os.path.join("results", 'zagreb_importance.png'))

# Performance metrics visualization
metrics = {
    'Accuracy': accuracy,
    'F1-Score': f1,
    'Precision': precision,
    'Recall': recall
}

plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
ax = sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), palette="viridis")
plt.title("Evaluation Metrics for Humor Detection using Zagreb Indices + Embeddings", fontsize=15)
plt.ylim(0, 1.0)

for i, v in enumerate(metrics.values()):
    ax.text(i, v + 0.02, f"{v:.4f}", ha='center', fontsize=12)
    
plt.tight_layout()
plt.savefig(os.path.join("results", 'metrics.png'))

# Confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Not Humor', 'Humor'], 
            yticklabels=['Not Humor', 'Humor'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join("results", 'confusion_matrix.png'))

# Save summary
top_indices = [zagreb_names[i] for i in sorted_idx[-3:]][::-1]
top_importances = sorted_importances[-3:][::-1]

summary = f"""
HUMOR DETECTION USING ZAGREB INDICES - SUMMARY
----------------------------------------------
DATASET: {len(reviews_df)} reviews ({sum(reviews_df['label'])} humorous, {len(reviews_df) - sum(reviews_df['label'])} non-humorous)

PERFORMANCE METRICS:
- Accuracy:  {accuracy:.4f}
- Precision: {precision:.4f}
- Recall:    {recall:.4f}
- F1 Score:  {f1:.4f}

TOP 3 MOST IMPORTANT ZAGREB INDICES:
1. {top_indices[0]}: {top_importances[0]:.4f}
2. {top_indices[1]}: {top_importances[1]:.4f}
3. {top_indices[2]}: {top_importances[2]:.4f}
"""

print(summary)

with open(os.path.join("results", 'results_summary.txt'), 'w') as f:
    f.write(summary)

print(f"Pipeline Completed Successfully. Results saved to 'results' directory.")
