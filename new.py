import torch.nn.functional as F

# First, make sure you've imported all necessary libraries
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_recall_curve, f1_score, confusion_matrix
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Constants
MAX_REVIEWS = 10000
EPOCHS = 8  # Increased epochs
LEARNING_RATE = 3e-4  # Slightly higher learning rate
BASE_PATH = r"D:\Humor Detection\yelp_dataset"
BATCH_SIZE = 128
SEED = 42

# Set random seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

def build_cooccurrence_graph(text, window_size=3):  # Increased window size
    """Build a co-occurrence graph from text."""
    # Simple tokenization fallback in case NLTK fails
    words = []
    
    try:
        # Use NLTK for better tokenization
        words = [word.lower() for word in word_tokenize(text) if word.isalpha()]
    except Exception:
        # Fallback to simple tokenization
        words = [word.lower() for word in text.split() if word.isalpha()]
    
    # Remove stopwords
    try:
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if w not in stop_words]
    except Exception:
        pass  # Continue without stopword removal if it fails
    
    # Build the graph
    G = nx.Graph()
    
    # Add nodes first
    for word in words:
        G.add_node(word)
    
    # Add edges based on co-occurrence within window
    for i in range(len(words)):
        for j in range(i+1, min(i + window_size + 1, len(words))):
            if words[i] != words[j]:
                if G.has_edge(words[i], words[j]):
                    # Increment weight if edge exists
                    G[words[i]][words[j]]['weight'] += 1
                else:
                    # Create new edge with weight 1
                    G.add_edge(words[i], words[j], weight=1)
    
    return G

def compute_zagreb_indices(G):
    """Compute various Zagreb indices for a graph."""
    # Handle empty graph case
    if len(G) == 0:
        zeros = [0.0] * 9
        return zeros, dict(zip([
            'first_zagreb', 'second_zagreb', 'first_zagreb_co', 'second_zagreb_co',
            'modified_zagreb', 'generalized_zagreb', 'hyper_zagreb',
            'third_zagreb', 'f_index'
        ], zeros))
    
    degrees = dict(G.degree(weight='weight'))  # Use weighted degrees
    edges = G.edges()
    non_edges = nx.non_edges(G)
    
    # Calculate Zagreb indices
    first_zagreb = sum(d**2 for d in degrees.values())
    second_zagreb = sum(degrees[u] * degrees[v] for u, v in edges)
    first_zagreb_co = sum(degrees[u] + degrees[v] for u, v in non_edges)
    second_zagreb_co = sum(degrees[u] * degrees[v] for u, v in non_edges)
    modified_zagreb = sum(1 / (d + 1e-10) for d in degrees.values())  # Avoid division by zero
    generalized_zagreb = sum(abs(degrees[u] - degrees[v]) for u, v in edges)
    hyper_zagreb = sum((degrees[u] + degrees[v])**2 for u, v in edges)
    third_zagreb = sum(d**3 for d in degrees.values())
    f_index = sum(d**3 for d in degrees.values())
    
    # Create a dictionary for easier access
    indices = {
        'first_zagreb': first_zagreb,
        'second_zagreb': second_zagreb, 
        'first_zagreb_co': first_zagreb_co,
        'second_zagreb_co': second_zagreb_co,
        'modified_zagreb': modified_zagreb,
        'generalized_zagreb': generalized_zagreb,
        'hyper_zagreb': hyper_zagreb,
        'third_zagreb': third_zagreb,
        'f_index': f_index
    }
    
    # Normalize the indices to prevent exploding values
    for key in indices:
        indices[key] = np.clip(indices[key], -1e6, 1e6)
        
    return list(indices.values()), indices

def analyze_zagreb_importance(model, feature_names=None):
    """Analyze the importance of Zagreb indices for humor detection."""
    if feature_names is None:
        zagreb_names = [
            'first_zagreb', 'second_zagreb', 'first_zagreb_co', 'second_zagreb_co',
            'modified_zagreb', 'generalized_zagreb', 'hyper_zagreb',
            'third_zagreb', 'f_index'
        ]
        feature_names = zagreb_names + ['embed_' + str(i) for i in range(384)] + ['meta_' + str(i) for i in range(6)]
    
    weights = model.fc1.weight.data.cpu().numpy() if isinstance(model, ImprovedHumorMLP) else model.fc[0].weight.data.cpu().numpy()
    importance = np.abs(weights).mean(axis=0)
    
    zagreb_importance = [(name, imp) for name, imp in zip(feature_names[:9], importance[:9])]
    zagreb_importance.sort(key=lambda x: x[1], reverse=True)
    
    return zagreb_importance

def extract_features(text, user_meta, biz_meta, model):
    """Extract features from text and metadata for humor detection."""
    G = build_cooccurrence_graph(text)
    zagreb_values, zagreb_dict = compute_zagreb_indices(G)
    
    # Handle errors in sentence embedding
    try:
        embed = model.encode(text)
    except Exception:
        # If embedding fails, provide zeros
        embed = np.zeros(384)
    
    # Extract user and business metadata features
    extra = [
        np.log1p(user_meta.get("review_count", 0)),  # Log transform to reduce skew
        user_meta.get("elite", 0),
        user_meta.get("average_stars", 0.0),
        np.log1p(user_meta.get("funny_given", 0)),  # Log transform
        biz_meta.get("stars", 0.0),
        np.log1p(biz_meta.get("review_count", 0))  # Log transform
    ]
    
    # Additional text features
    text_length = min(len(text) / 1000, 5.0)  # Normalized text length
    word_count = min(len(text.split()) / 100, 5.0)  # Normalized word count
    
    # Compute basic linguistic features
    exclamation_count = min(text.count('!') / 10, 5.0)
    question_count = min(text.count('?') / 5, 5.0)
    capital_ratio = sum(1 for c in text if c.isupper()) / (len(text) + 1)
    
    # Additional features
    extra_features = [text_length, word_count, exclamation_count, question_count, capital_ratio]
    
    # Concatenate all features
    features = np.concatenate((zagreb_values, embed, extra, extra_features))
    
    return features, zagreb_dict

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

class ImprovedHumorMLP(nn.Module):
    """Improved neural network for humor detection."""
    def __init__(self, input_dim):
        super(ImprovedHumorMLP, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(input_dim)
        self.fc1 = nn.Linear(input_dim, 256)  # Wider network
        self.dropout1 = nn.Dropout(0.4)  # Increased dropout
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.batch_norm3 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.dropout3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = self.batch_norm1(x)
        x = F.leaky_relu(self.fc1(x), 0.1)  # Using LeakyReLU
        x = self.dropout1(x)
        x = self.batch_norm2(x)
        x = F.leaky_relu(self.fc2(x), 0.1)
        x = self.dropout2(x)
        x = self.batch_norm3(x)
        x = F.leaky_relu(self.fc3(x), 0.1)
        x = self.dropout3(x)
        x = torch.sigmoid(self.fc4(x))
        return x

def load_user_data(user_file):
    """Load user metadata from Yelp dataset."""
    user_dict = {}
    with open(user_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                user = json.loads(line)
                user_dict[user['user_id']] = {
                    'review_count': user.get('review_count', 0),
                    'elite': 1 if user.get('elite') else 0,
                    'average_stars': user.get('average_stars', 0.0),
                    'funny_given': user.get('compliment_funny', 0)
                }
            except json.JSONDecodeError:
                continue  # Skip malformed lines
    
    return user_dict

def load_business_data(biz_file):
    """Load business metadata from Yelp dataset."""
    biz_dict = {}
    with open(biz_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                biz = json.loads(line)
                biz_dict[biz['business_id']] = {
                    'stars': biz.get('stars', 0.0),
                    'review_count': biz.get('review_count', 0)
                }
            except json.JSONDecodeError:
                continue  # Skip malformed lines
    
    return biz_dict

def load_and_process_data():
    """Load and process Yelp data for humor detection."""
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    X, y = [], []
    
    # File paths
    user_file = os.path.join(BASE_PATH, "yelp_academic_dataset_user.json")
    business_file = os.path.join(BASE_PATH, "yelp_academic_dataset_business.json")
    review_file = os.path.join(BASE_PATH, "yelp_academic_dataset_review.json")
    
    print(f"Loading user data from {user_file}")
    user_info = load_user_data(user_file)
    
    print(f"Loading business data from {business_file}")
    biz_info = load_business_data(business_file)
    
    print(f"Processing reviews from {review_file}")
    
    # Keep track of humor/non-humor counts to balance
    humor_count, non_humor_count = 0, 0
    humor_target = MAX_REVIEWS // 2
    non_humor_target = MAX_REVIEWS // 2
    
    with open(review_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f)):
            if humor_count >= humor_target and non_humor_count >= non_humor_target:
                break
                
            try:
                data = json.loads(line)
                text = data["text"]
                is_humor = data.get("funny", 0) > 0
                
                # Balance dataset collection
                if is_humor and humor_count < humor_target:
                    label = 1
                    humor_count += 1
                elif not is_humor and non_humor_count < non_humor_target:
                    label = 0
                    non_humor_count += 1
                else:
                    continue  # Skip this review to maintain balance
                
                features, _ = extract_features(
                    text, 
                    user_info.get(data.get("user_id", ""), {}), 
                    biz_info.get(data.get("business_id", ""), {}),
                    sentence_model
                )
                
                X.append(features)
                y.append(label)
                
                # Print progress
                if (humor_count + non_humor_count) % 1000 == 0:
                    print(f"Collected {humor_count} humorous and {non_humor_count} non-humorous reviews")
            
            except Exception as e:
                print(f"Error processing review: {str(e)}")
                continue
    
    print(f"Final dataset: {humor_count} humorous and {non_humor_count} non-humorous reviews")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def find_best_threshold(y_true, y_scores):
    """Find the optimal classification threshold."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
    best_threshold = thresholds[np.argmax(f1_scores)]
    return best_threshold

def visualize_results(y_test, y_pred, zagreb_importance):
    """Create visualizations of model performance and feature importance."""
    # Create a directory for visualizations
    os.makedirs("visualizations", exist_ok=True)
    
    # Plot Zagreb indices importance
    plt.figure(figsize=(10, 6))
    names = [name for name, _ in zagreb_importance]
    values = [value for _, value in zagreb_importance]
    plt.barh(names, values, color='skyblue')
    plt.xlabel('Importance')
    plt.title('Zagreb Indices Importance for Humor Detection')
    plt.tight_layout()
    plt.savefig('visualizations/zagreb_importance.png')
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Humor', 'Humor'],
                yticklabels=['Not Humor', 'Humor'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('visualizations/confusion_matrix.png')
    
    print("Visualizations saved to 'visualizations' directory.")

def main():
    """Main execution function."""
    print("Loading and processing data...")
    X, y = load_and_process_data()
    
    # Check for class balance
    unique, counts = np.unique(y, return_counts=True)
    print(f"Class distribution: {dict(zip(unique, counts))}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    
    # Apply SMOTE to oversample minority class
    print("Applying SMOTE to balance training data...")
    sm = SMOTE(random_state=SEED)
    X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)
    
    # Check class distribution after SMOTE
    unique, counts = np.unique(y_train_resampled, return_counts=True)
    print(f"Training class distribution after SMOTE: {dict(zip(unique, counts))}")
    
    # Convert to tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    X_train_t = torch.tensor(X_train_resampled).to(device)
    y_train_t = torch.tensor(y_train_resampled).unsqueeze(1).to(device)
    X_test_t = torch.tensor(X_test).to(device)
    y_test_t = torch.tensor(y_test).unsqueeze(1).to(device)
    
    # Create data loader with larger batch size
    train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize model
    input_dim = X.shape[1]
    print(f"Feature dimension: {input_dim}")
    net = ImprovedHumorMLP(input_dim=input_dim).to(device)
    
    # Use Focal Loss with BCELoss for imbalanced data
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)  # Added weight decay
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    
    # Training loop
    best_f1 = 0
    best_model_state = None
    
    print("\nStarting training...")
    for epoch in range(EPOCHS):
        net.train()
        total_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss/len(train_loader)
        
        # Validation after each epoch
        net.eval()
        with torch.no_grad():
            val_preds = net(X_test_t).cpu().numpy()
            
        # Find optimal threshold
        threshold = find_best_threshold(y_test, val_preds)
        val_preds_bin = (val_preds > threshold).astype(int)
        
        # Calculate metrics
        val_f1 = f1_score(y_test, val_preds_bin)
        val_auc = roc_auc_score(y_test, val_preds)
        
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}, Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}, Threshold: {threshold:.4f}")
        
        # Save best model based on F1 score
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_state = net.state_dict().copy()
            print(f"New best model with F1: {best_f1:.4f}")
        
        scheduler.step(avg_loss)
    
    # Load best model for final evaluation
    if best_model_state is not None:
        net.load_state_dict(best_model_state)
        
    # Final evaluation
    print("\nFinal Evaluation:")
    net.eval()
    with torch.no_grad():
        preds = net(X_test_t).cpu().numpy()
        
    # Find optimal threshold
    best_threshold = find_best_threshold(y_test, preds)
    print(f"Optimal threshold: {best_threshold:.4f}")
    
    # Apply optimal threshold
    preds_bin = (preds > best_threshold).astype(int)
    
    # Print metrics
    auc = roc_auc_score(y_test, preds)
    accuracy = accuracy_score(y_test, preds_bin)
    f1 = f1_score(y_test, preds_bin)
    
    print(f"ROC-AUC: {auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(classification_report(y_test, preds_bin))
    
    # Analyze feature importance
    zagreb_importance = analyze_zagreb_importance(net)
    print("\nZagreb Indices Importance for Humor Detection:")
    for name, importance in zagreb_importance:
        print(f"{name}: {importance:.6f}")
    
    # Create visualizations
    from sklearn.metrics import confusion_matrix
    visualize_results(y_test, preds_bin, zagreb_importance)
    
    print("\nModel training complete!")

if __name__ == "__main__":
    main()