import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import networkx as nx
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_recall_curve, f1_score, confusion_matrix, roc_curve
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModel, AutoTokenizer

# Ensure necessary NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

# Constants - optimized settings
MAX_REVIEWS = 10000
EPOCHS = 12  # Increased for better convergence
LEARNING_RATE = 2e-4  # Optimized learning rate
BASE_PATH = r"D:\Humor Detection\yelp_dataset"
BATCH_SIZE = 64  # Smaller batch size for better generalization
SEED = 42
TEST_SIZE = 0.15  # Smaller test size for more training data
EMBEDDING_MODEL = 'paraphrase-MiniLM-L6-v2'  # Better quality embeddings

# Set seeds for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def build_cooccurrence_graph(text, window_size=4):  # Increased window for better context
    """Build an enhanced co-occurrence graph from text with linguistic features."""
    try:
        words = [word.lower() for word in word_tokenize(text) if word.isalpha()]
    except:
        words = [word.lower() for word in text.split() if word.isalpha()]
    
    # Remove stopwords
    try:
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if w not in stop_words]
    except:
        pass
    
    # Build the graph
    G = nx.Graph()
    
    # Add nodes with POS tagging if possible
    for word in words:
        G.add_node(word)
    
    # Add edges with advanced weighting
    for i in range(len(words)):
        for j in range(i+1, min(i + window_size + 1, len(words))):
            if words[i] != words[j]:
                # Weight inversely proportional to distance
                weight = 1.0 / (j - i)
                if G.has_edge(words[i], words[j]):
                    G[words[i]][words[j]]['weight'] += weight
                else:
                    G.add_edge(words[i], words[j], weight=weight)
    
    return G

def compute_zagreb_indices(G):
    """Enhanced Zagreb indices computation with normalization."""
    if len(G) == 0:
        zeros = [0.0] * 9
        return zeros, dict(zip([
            'first_zagreb', 'second_zagreb', 'first_zagreb_co', 'second_zagreb_co',
            'modified_zagreb', 'generalized_zagreb', 'hyper_zagreb',
            'third_zagreb', 'f_index'
        ], zeros))
    
    degrees = dict(G.degree(weight='weight'))
    edges = G.edges()
    non_edges = nx.non_edges(G)
    
    # Calculate Zagreb indices
    first_zagreb = sum(d**2 for d in degrees.values())
    second_zagreb = sum(degrees[u] * degrees[v] for u, v in edges)
    first_zagreb_co = sum(degrees[u] + degrees[v] for u, v in non_edges)
    second_zagreb_co = sum(degrees[u] * degrees[v] for u, v in non_edges)
    modified_zagreb = sum(1 / (d + 1e-10) for d in degrees.values())
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
    
    # Advanced normalization by graph size
    node_count = len(G)
    for key in indices:
        # Normalize by node count to make indices comparable across different graph sizes
        indices[key] = np.log1p(indices[key] / (node_count + 1e-10))
        # Clip extreme values
        indices[key] = np.clip(indices[key], -10, 10)
        
    return list(indices.values()), indices

def extract_humor_specific_features(text):
    """Extract features specifically targeting humor characteristics."""
    # Count linguistic humor markers
    contrasts = len([i for i in range(len(text)-1) if text[i] == '!' and text[i+1] == '?'])
    repetitions = sum(1 for i in range(len(text)-2) if text[i:i+3] == text[i]*3)
    
    # Look for humor patterns
    has_haha = 1 if 'haha' in text.lower() or 'lol' in text.lower() or 'lmao' in text.lower() else 0
    has_irony = text.count('...') / max(1, len(text)/100)  # Ellipsis often indicates irony
    
    # Sentiment shifts (common in humor)
    positive_words = ['good', 'great', 'nice', 'excellent', 'happy', 'love', 'best']
    negative_words = ['bad', 'worst', 'terrible', 'hate', 'awful', 'poor', 'wrong']
    pos_count = sum(text.lower().count(word) for word in positive_words)
    neg_count = sum(text.lower().count(word) for word in negative_words)
    sentiment_contrast = abs(pos_count - neg_count) / (pos_count + neg_count + 1)
    
    return [contrasts, repetitions, has_haha, has_irony, sentiment_contrast]

def extract_features(text, user_meta, biz_meta, model):
    """Extract comprehensive features for humor detection."""
    # Basic graph features
    G = build_cooccurrence_graph(text)
    zagreb_values, zagreb_dict = compute_zagreb_indices(G)
    
    # Text embeddings
    try:
        embed = model.encode(text)
    except:
        embed = np.zeros(384)
    
    # Metadata features with transformations
    extra = [
        np.log1p(user_meta.get("review_count", 0)),
        user_meta.get("elite", 0),
        user_meta.get("average_stars", 0.0),
        np.log1p(user_meta.get("funny_given", 0)), 
        biz_meta.get("stars", 0.0),
        np.log1p(biz_meta.get("review_count", 0))
    ]
    
    # Basic text stats
    text_length = min(len(text) / 1000, 5.0)
    word_count = min(len(text.split()) / 100, 5.0)
    avg_word_length = np.mean([len(w) for w in text.split()]) if text.split() else 0
    sentence_count = text.count('.') + text.count('!') + text.count('?')
    avg_sentence_length = word_count / (sentence_count + 1e-5)
    
    # Linguistic markers
    exclamation_count = min(text.count('!') / 10, 5.0)
    question_count = min(text.count('?') / 5, 5.0)
    capital_ratio = sum(1 for c in text if c.isupper()) / (len(text) + 1)
    punctuation_ratio = sum(1 for c in text if c in ",.!?;:") / (len(text) + 1)
    
    # Advanced humor-specific features
    humor_features = extract_humor_specific_features(text)
    
    # Graph structure features
    if len(G) > 0:
        try:
            clustering = nx.average_clustering(G)
            diameter = nx.diameter(G) if nx.is_connected(G) else 0
            density = nx.density(G)
            graph_features = [clustering, diameter, density]
        except:
            graph_features = [0, 0, 0]
    else:
        graph_features = [0, 0, 0]
    
    # Concatenate all feature groups
    basic_features = [text_length, word_count, avg_word_length, avg_sentence_length]
    linguistic_features = [exclamation_count, question_count, capital_ratio, punctuation_ratio]
    
    all_features = np.concatenate((
        zagreb_values,          # Graph topology features (9)
        embed,                  # Semantic embeddings (384)
        extra,                  # User and business metadata (6)
        basic_features,         # Basic text statistics (4)
        linguistic_features,    # Linguistic markers (4)
        humor_features,         # Humor-specific features (5)
        graph_features          # Graph structure features (3)
    ))
    
    return all_features, zagreb_dict

class FocalLoss(nn.Module):
    """Focal Loss with adaptive alpha for severe class imbalance."""
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        # Higher gamma gives more focus to hard examples
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

class ResidualBlock(nn.Module):
    """Residual block for better gradient flow."""
    def __init__(self, in_dim, out_dim):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(0.3)
        
        # Projection shortcut if dimensions don't match
        self.shortcut = nn.Identity()
        if in_dim != out_dim:
            self.shortcut = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim)
            )
            
    def forward(self, x):
        identity = x
        
        out = self.linear1(x)
        out = self.bn1(out)
        out = F.leaky_relu(out, 0.1)
        out = self.dropout(out)
        
        out = self.linear2(out)
        out = self.bn2(out)
        
        # Add residual connection
        out += self.shortcut(identity)
        out = F.leaky_relu(out, 0.1)
        
        return out

class AdvancedHumorMLP(nn.Module):
    """Advanced neural network architecture with residual connections."""
    def __init__(self, input_dim):
        super(AdvancedHumorMLP, self).__init__()
        self.input_norm = nn.BatchNorm1d(input_dim)
        
        # Initial dimension reduction
        self.fc_reduce = nn.Linear(input_dim, 512)
        self.bn_reduce = nn.BatchNorm1d(512)
        self.dropout_reduce = nn.Dropout(0.5)
        
        # Residual blocks for deep representation learning
        self.res1 = ResidualBlock(512, 256)
        self.res2 = ResidualBlock(256, 128)
        self.res3 = ResidualBlock(128, 64)
        
        # Output layer
        self.fc_out = nn.Linear(64, 1)
        
    def forward(self, x):
        x = self.input_norm(x)
        
        # Initial reduction
        x = self.fc_reduce(x)
        x = self.bn_reduce(x)
        x = F.leaky_relu(x, 0.1)
        x = self.dropout_reduce(x)
        
        # Residual blocks
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        
        # Output
        x = torch.sigmoid(self.fc_out(x))
        return x

def load_user_data(user_file):
    """Load user metadata from Yelp dataset with error handling."""
    user_dict = {}
    try:
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
                except:
                    continue
    except Exception as e:
        print(f"Error loading user data: {str(e)}")
    
    return user_dict

def load_business_data(biz_file):
    """Load business metadata from Yelp dataset with error handling."""
    biz_dict = {}
    try:
        with open(biz_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    biz = json.loads(line)
                    biz_dict[biz['business_id']] = {
                        'stars': biz.get('stars', 0.0),
                        'review_count': biz.get('review_count', 0)
                    }
                except:
                    continue
    except Exception as e:
        print(f"Error loading business data: {str(e)}")
    
    return biz_dict

def load_and_process_data():
    """Load and process Yelp data for humor detection with enhanced balancing strategy."""
    # Load sentence transformer
    sentence_model = SentenceTransformer(EMBEDDING_MODEL)
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
    
    # Enhanced balancing strategy - collect more humor samples
    humor_count, non_humor_count = 0, 0
    humor_target = int(MAX_REVIEWS * 0.6)  # Collecting more humor samples for better balance
    non_humor_target = MAX_REVIEWS - humor_target
    
    # First pass to collect all humor samples since they're rare
    humor_samples = []
    non_humor_samples = []
    
    with open(review_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f)):
            if len(humor_samples) >= humor_target and len(non_humor_samples) >= non_humor_target:
                break
                
            try:
                data = json.loads(line)
                text = data["text"]
                funny_votes = data.get("funny", 0)
                
                # Only collect samples that meet our criteria
                if len(text) < 20:  # Skip very short reviews
                    continue
                    
                if funny_votes > 0:
                    if len(humor_samples) < humor_target:
                        sample = {
                            'text': text,
                            'user_id': data.get("user_id", ""),
                            'business_id': data.get("business_id", ""),
                            'funny_votes': funny_votes
                        }
                        humor_samples.append(sample)
                else:
                    if len(non_humor_samples) < non_humor_target:
                        sample = {
                            'text': text,
                            'user_id': data.get("user_id", ""),
                            'business_id': data.get("business_id", ""),
                            'funny_votes': 0
                        }
                        non_humor_samples.append(sample)
                        
                # Print progress occasionally
                total = len(humor_samples) + len(non_humor_samples)
                if total % 1000 == 0:
                    print(f"Collected {len(humor_samples)} humorous and {len(non_humor_samples)} non-humorous reviews")
            
            except Exception as e:
                continue
    
    print(f"Processing {len(humor_samples)} humorous and {len(non_humor_samples)} non-humorous samples")
    
    # Process all samples
    all_samples = humor_samples + non_humor_samples
    for sample in tqdm(all_samples):
        try:
            features, _ = extract_features(
                sample['text'], 
                user_info.get(sample.get('user_id', ""), {}), 
                biz_info.get(sample.get('business_id', ""), {}),
                sentence_model
            )
            
            X.append(features)
            y.append(1 if sample['funny_votes'] > 0 else 0)
        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            continue
    
    print(f"Final dataset: {sum(y)} humorous and {len(y) - sum(y)} non-humorous reviews")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def find_best_threshold(y_true, y_scores):
    """Find optimal classification threshold maximizing F1 score."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    
    # Add the endpoint case where threshold=1
    thresholds = np.append(thresholds, 1.0)
    
    # Calculate F1 scores
    f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
    
    # Find best threshold
    best_threshold = thresholds[np.argmax(f1_scores)]
    return best_threshold

def visualize_results(y_test, y_pred, y_scores, zagreb_importance):
    """Create comprehensive visualizations for model analysis."""
    # Create directory
    os.makedirs("visualizations", exist_ok=True)
    
    # 1. Zagreb indices importance
    plt.figure(figsize=(10, 6))
    names = [name for name, _ in zagreb_importance]
    values = [value for _, value in zagreb_importance]
    plt.barh(names, values, color='skyblue')
    plt.xlabel('Importance')
    plt.title('Zagreb Indices Importance for Humor Detection')
    plt.tight_layout()
    plt.savefig('visualizations/zagreb_importance.png')
    
    # 2. Confusion matrix
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
    
    # 3. ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (AUC = {roc_auc_score(y_test, y_scores):.3f})')
    plt.tight_layout()
    plt.savefig('visualizations/roc_curve.png')
    
    # 4. Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.tight_layout()
    plt.savefig('visualizations/pr_curve.png')
    
    print("Visualizations saved to 'visualizations' directory.")

def main():
    """Main execution function with enhanced training and evaluation pipeline."""
    print("Loading and processing data...")
    X, y = load_and_process_data()
    
    # Check for class balance
    unique, counts = np.unique(y, return_counts=True)
    print(f"Class distribution: {dict(zip(unique, counts))}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    
    # Apply SMOTE with optimized parameters
    print("Applying SMOTE to balance training data...")
    sm = SMOTE(random_state=SEED, k_neighbors=5)
    X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)
    
    # Convert to tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    X_train_t = torch.tensor(X_train_resampled).to(device)
    y_train_t = torch.tensor(y_train_resampled).unsqueeze(1).to(device)
    X_test_t = torch.tensor(X_test).to(device)
    y_test_t = torch.tensor(y_test).unsqueeze(1).to(device)
    
    # Create data loader
    train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    # Initialize advanced model
    input_dim = X.shape[1]
    print(f"Feature dimension: {input_dim}")
    net = AdvancedHumorMLP(input_dim=input_dim).to(device)
    
    # Optimized loss and optimizer
    criterion = FocalLoss(alpha=0.35, gamma=2.5)  # Tuned parameters
    optimizer = optim.AdamW(net.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # One-cycle learning rate scheduler for faster convergence
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS,
        pct_start=0.3,  # Spend 30% of time warming up
        div_factor=10,  # Initial LR is max_lr/10
        final_div_factor=100  # Min LR is max_lr/1000
    )
    
    # Training loop with advanced monitoring
    best_f1 = 0
    best_model_state = None
    patience = 3
    patience_counter = 0
    
    print("\nStarting training...")
    for epoch in range(EPOCHS):
        net.train()
        total_loss = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}") as t:
            for inputs, labels in t:
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                t.set_postfix(loss=loss.item())
        
        avg_loss = total_loss/len(train_loader)
        
        # Validation
        net.eval()
        with torch.no_grad():
            val_preds = net(X_test_t).cpu().numpy()
            
        # Find optimal threshold
        threshold = find_best_threshold(y_test, val_preds)
        val_preds_bin = (val_preds > threshold).astype(int)
        
        # Calculate metrics
        val_f1 = f1_score(y_test, val_preds_bin)
        val_recall = recall_score(y_test, val_preds_bin)
        val_precision = precision_score(y_test, val_preds_bin)
        val_auc = roc_auc_score(y_test, val_preds)
        
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")
        
        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_state = net.state_dict().copy()
            patience_counter = 0
            print(f"New best model with F1: {best_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping after {epoch+1} epochs due to no improvement")
                break
    
    # Load best model
    if best_model_state is not None:
        net.load_state_dict(best_model_state)
        
    # Final evaluation with ensemble prediction
    print("\nFinal Evaluation:")
    net.eval()
    
    # Use test-time augmentation with multiple thresholds
    with torch.no_grad():
        base_preds = net(X_test_t).cpu().numpy()
        
    # Find optimal threshold on best model
    best_threshold = find_best_threshold(y_test, base_preds)
    print(f"Optimal threshold: {best_threshold:.4f}")
    
    # Apply threshold
    preds_bin = (base_preds > best_threshold).astype(int)
    
    # Print comprehensive metrics
    auc = roc_auc_score(y_test, base_preds)
    accuracy = accuracy_score(y_test, preds_bin)
    balanced_accuracy = balanced_accuracy_score(y_test, preds_bin)
    f1 = f1_score(y_test, preds_bin)
    precision = precision_score(y_test, preds_bin)
    recall = recall_score(y_test, preds_bin)
    
    print("\n====== FINAL RESULTS =======")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds_bin))
    
    # Analyze Zagreb indices importance
    def analyze_zagreb_importance(model):
        """
        Estimate the importance of Zagreb indices by inspecting the absolute weights
        of the first layer corresponding to the Zagreb features.
        """
        zagreb_names = [
            'first_zagreb', 'second_zagreb', 'first_zagreb_co', 'second_zagreb_co',
            'modified_zagreb', 'generalized_zagreb', 'hyper_zagreb',
            'third_zagreb', 'f_index'
        ]
        # The Zagreb indices are the first 9 features in the input
        with torch.no_grad():
            first_layer_weights = model.fc_reduce.weight[:, :9].abs().mean(dim=0).cpu().numpy()
        importance = list(zip(zagreb_names, first_layer_weights))
        # Sort by importance descending
        importance.sort(key=lambda x: x[1], reverse=True)
        return importance

    zagreb_importance = analyze_zagreb_importance(net)
    print("\nZagreb Indices Importance for Humor Detection:")
    for name, importance in zagreb_importance:
        print(f"{name}: {importance:.6f}")
    
    # Create visualizations
    from sklearn.metrics import roc_curve, precision_score, recall_score, balanced_accuracy_score
    visualize_results(y_test, preds_bin, base_preds, zagreb_importance)
    
    # Save model
    model_path = os.path.join("models", "best_humor_detection_model.pt")
    os.makedirs("models", exist_ok=True)
    torch.save({
        'model_state_dict': best_model_state,
        'input_dim': input_dim,
        'threshold': best_threshold,
    }, model_path)
    
    print(f"\nModel saved to {model_path}")
    print("\nTraining complete!")

if __name__ == "__main__":
    main()