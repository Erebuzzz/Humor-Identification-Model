import streamlit as st
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import BertTokenizer, BertForSequenceClassification
import os
from PIL import Image
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Humor Detection",
    page_icon="😂",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem !important;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #333;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .humor-positive {
        background-color: rgba(76, 175, 80, 0.2);
        border-left: 5px solid #4CAF50;
    }
    .humor-negative {
        background-color: rgba(244, 67, 54, 0.2);
        border-left: 5px solid #F44336;
    }
    .highlight {
        font-weight: 500;
        color: #1E88E5;
    }
    .model-card {
        background-color: #f9f9f9;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #ddd;
    }
    .viz-container {
        background-color: #ffffff;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid #eee;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# Initialize NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))
    stop_words.update({'would', 'could', 'should', 'really', 'much', 'even', 'also'})
except:
    stop_words = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had'}

# Load the models and preprocessors
@st.cache_resource
def load_models():
    models = {}
    try:
        with open("saved_models/svm_model.pkl", 'rb') as f:
            models["svm"] = pickle.load(f)
        with open("saved_models/naive_bayes_model.pkl", 'rb') as f:
            models["naive_bayes"] = pickle.load(f)
        with open("saved_models/mlp_adam_model.pkl", 'rb') as f:
            models["mlp_adam"] = pickle.load(f)
        with open("saved_models/mlp_rmsprop_model.pkl", 'rb') as f:
            models["mlp_rmsprop"] = pickle.load(f)
        with open("saved_models/stacking_ensemble_model.pkl", 'rb') as f:
            models["stacking"] = pickle.load(f)
        
        # Load BERT model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        bert_model = BertForSequenceClassification.from_pretrained("saved_models/bert-base-uncased", 
                                                                  num_labels=2)
        bert_model.load_state_dict(torch.load("saved_models/bert_best_model.pth", 
                                             map_location=device))
        bert_model.to(device)
        bert_model.eval()
        models["bert"] = bert_model
        
        # Load tokenizer
        tokenizer = BertTokenizer.from_pretrained("saved_models/bert-base-uncased")
        
        # Load word2vec
        from gensim.models import Word2Vec
        w2v_model = Word2Vec.load("saved_models/word2vec_enhanced.model")
        
        # Load preprocessors
        with open("saved_models/preprocessors.pkl", 'rb') as f:
            scaler, variance_selector = pickle.load(f)
            
        # Load GloVe embeddings
        glove_embeddings = {}
        try:
            with open("saved_models/glove.6B.100d.txt", 'r', encoding='utf-8') as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    vector = np.array(values[1:], dtype='float32')
                    glove_embeddings[word] = vector
        except:
            print("GloVe embeddings not found.")
            
        return models, tokenizer, w2v_model, scaler, variance_selector, glove_embeddings, device
    except Exception as e:
        st.error(f"Error loading models: {e}")
        # Return dummy models for demo
        return {}, None, None, None, None, {}, "cpu"

# Text preprocessing
def preprocess_text(text):
    """Preprocess text for feature extraction"""
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

# Graph building and Zagreb indices computation
def build_semantic_graph(tokens, window=3):
    """Build semantic graph from tokens"""
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

def compute_upsilon_degrees_robust(G):
    """Compute Upsilon degrees for graph nodes"""
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
        
        if M_v <= 0:
            upsilon_degrees[v] = 0.0
        elif abs(M_v - 1.0) < 1e-10:
            upsilon_degrees[v] = float(deg_v)
        else:
            try:
                if deg_v == 1:
                    upsilon_degrees[v] = 1.0
                else:
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
            except:
                upsilon_degrees[v] = float(deg_v)
    
    return upsilon_degrees

def compute_zagreb_indices(G):
    """Compute both traditional and Upsilon Zagreb indices"""
    if not G.edges() or len(G.nodes()) < 2:
        return [0.0] * 12  # 9 traditional + 3 upsilon indices
    
    try:
        degrees = dict(G.degree())
        edges = list(G.edges())
        
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
        forgotten = third
        
        # Upsilon indices
        upsilon_degrees = compute_upsilon_degrees_robust(G)
        if upsilon_degrees:
            upsilon_values = list(upsilon_degrees.values())
            M1_upsilon = sum(min(v**2, 1e6) for v in upsilon_values)
            M2_upsilon = sum(min(upsilon_degrees.get(u, 0) * upsilon_degrees.get(v, 0), 1e6) 
                           for u, v in edges)
            M3_upsilon = sum(min(upsilon_degrees.get(u, 0) + upsilon_degrees.get(v, 0), 1e6) 
                           for u, v in edges)
        else:
            M1_upsilon = M2_upsilon = M3_upsilon = 0.0
            
        return [m1, m2, m1_co, m2_co, generalized, modified, third, hyper, forgotten, 
                M1_upsilon, M2_upsilon, M3_upsilon]
    except:
        return [0.0] * 12

def extract_features(tokens, text, w2v_model, glove_embeddings):
    """Extract features from text for model input"""
    # Graph-based features
    try:
        G = build_semantic_graph(tokens)
        zagreb = compute_zagreb_indices(G)
    except:
        zagreb = [0.0] * 12
    
    # Stylistic features
    humor_words = {'funny', 'hilarious', 'laugh', 'lol', 'haha', 'joke', 'humor', 'comedy',
                   'witty', 'amusing', 'ridiculous', 'silly', 'entertaining', 'clever',
                   'sarcastic', 'ironic', 'absurd', 'crazy', 'weird', 'bizarre'}
    
    try:
        humor_count = sum(1 for t in tokens if t.lower() in humor_words)
        humor_ratio = humor_count / max(len(tokens), 1)
        
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
    WORD2VEC_DIM = 150
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
    GLOVE_DIM = 100
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
    
    return all_features, G, zagreb

def get_prediction(text, models, tokenizer, w2v_model, scaler, variance_selector, glove_embeddings, device):
    """Get predictions from all models"""
    tokens = preprocess_text(text)
    features, graph, zagreb = extract_features(tokens, text, w2v_model, glove_embeddings)
    
    # Apply preprocessing transforms
    if scaler is not None and variance_selector is not None:
        features = scaler.transform(features.reshape(1, -1))
        features = variance_selector.transform(features)
    else:
        features = features.reshape(1, -1)
    
    predictions = {}
    
    # Get predictions from sklearn models
    for name, model in models.items():
        if name != "bert":
            try:
                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(features)[0, 1]
                    pred = int(prob > 0.5)
                    predictions[name] = {"prediction": pred, "probability": prob}
                else:
                    pred = model.predict(features)[0]
                    predictions[name] = {"prediction": pred, "probability": float(pred)}
            except Exception as e:
                predictions[name] = {"prediction": 0, "probability": 0.0, "error": str(e)}
    
    # Get prediction from BERT
    if "bert" in models and tokenizer is not None:
        try:
            bert_model = models["bert"]
            encoding = tokenizer(text, return_tensors='pt', max_length=192, 
                               padding='max_length', truncation=True)
            
            with torch.no_grad():
                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)
                outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                prob = probs[0, 1].item()
                
            predictions["bert"] = {"prediction": pred, "probability": prob}
        except Exception as e:
            predictions["bert"] = {"prediction": 0, "probability": 0.0, "error": str(e)}
    
    # Ensemble prediction (average probabilities)
    try:
        probs = [p["probability"] for p in predictions.values() if "probability" in p]
        if probs:
            ensemble_prob = sum(probs) / len(probs)
            ensemble_pred = int(ensemble_prob > 0.5)
            predictions["ensemble"] = {"prediction": ensemble_pred, "probability": ensemble_prob}
    except:
        predictions["ensemble"] = {"prediction": 0, "probability": 0.0}
    
    return predictions, graph, zagreb

def visualize_graph(G, upsilon_degrees=None):
    """Create a visualization of the semantic graph"""
    if not G.nodes():
        return None
    
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    
    node_size = [20 + 10 * G.degree(node) for node in G.nodes()]
    
    if upsilon_degrees:
        node_color = [upsilon_degrees.get(node, 0) for node in G.nodes()]
        plt.title("Semantic Graph with Upsilon Degrees")
        
        # Fix: Connect the scatter plot to the colorbar by storing the scatter result
        scatter = nx.draw_networkx(G, pos, with_labels=True, node_size=node_size, 
                node_color=node_color, cmap=plt.cm.viridis, font_size=8, 
                edge_color='gray', alpha=0.8)
                
        # Create a mappable object with the same colormap
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(
            vmin=min(node_color) if node_color else 0,
            vmax=max(node_color) if node_color else 1
        ))
        sm.set_array([])
        
        # Add colorbar to the current axes
        plt.colorbar(sm, ax=plt.gca(), label="Upsilon Degree")
    else:
        plt.title("Semantic Graph with Node Degrees")
        nx.draw(G, pos, with_labels=True, node_size=node_size, 
                node_color='skyblue', font_size=8, edge_color='gray', alpha=0.8)
    
    plt.tight_layout()
    return plt

def visualize_zagreb_metrics(zagreb):
    """Create a visual representation of Zagreb indices"""
    if all(v == 0 for v in zagreb):
        return None
    
    # Define names for indices
    trad_names = [
        "First Zagreb (M1)", 
        "Second Zagreb (M2)",
        "First Co-index", 
        "Second Co-index",
        "Generalized", 
        "Modified", 
        "Third",
        "Hyper", 
        "Forgotten"
    ]
    
    upsilon_names = [
        "First Upsilon (M1_υ)",
        "Second Upsilon (M2_υ)",
        "Third Upsilon (M3_υ)"
    ]
    
    # Create two bar charts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot traditional Zagreb indices
    trad_vals = zagreb[:9]
    # Normalize for better visualization
    trad_vals_norm = [v / max(max(trad_vals), 1) for v in trad_vals]
    ax1.bar(trad_names, trad_vals_norm, color='skyblue')
    ax1.set_title("Traditional Zagreb Indices (Normalized)", fontsize=14)
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot Upsilon Zagreb indices
    upsilon_vals = zagreb[9:]
    ax2.bar(upsilon_names, upsilon_vals, color='lightgreen')
    ax2.set_title("Upsilon Zagreb Indices", fontsize=14)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return plt

# Load cached images
def load_cached_images():
    images = {}
    image_paths = {
        'zagreb_scatter': 'results/zagreb_viz/zagreb_scatter_1.png',
        'zagreb_3d': 'results/zagreb_viz/zagreb_scatter_3d.png',
        'zagreb_corr': 'results/zagreb_viz/zagreb_correlation.png',
        'zagreb_dist': 'results/zagreb_viz/zagreb_distributions.png',
        'performance_1': 'results/performance_plots_1.png',
        'performance_2': 'results/performance_plots_2.png',
    }
    
    for name, path in image_paths.items():
        try:
            if os.path.exists(path):
                images[name] = Image.open(path)
        except:
            pass
            
    return images

# Load example texts
def load_example_texts():
    examples = [
        "This restaurant is hilarious! The waiter tripped and dumped spaghetti all over my new pants. Best birthday ever, lol!",
        "I can't believe how terrible this place is. The food was cold, service was slow, and the prices were outrageous.",
        "The comedy show was absolutely fantastic. I haven't laughed this hard in years - my sides hurt!",
        "The movie was decent but nothing special. Standard plot with predictable twists and average acting.",
        "Our server was AMAZING! She remembered my allergies from 6 months ago and brought me a special birthday dessert. What a gem!"
    ]
    return examples

# Main function
def main():
    st.markdown("<h1 class='main-header'>Humor Identification Model</h1>", unsafe_allow_html=True)
    st.markdown("""
    This application demonstrates a computational humor detection model using graph-based Zagreb indices 
    combined with machine learning and BERT.
    """)
    
    # Load models and preprocessors
    models, tokenizer, w2v_model, scaler, variance_selector, glove_embeddings, device = load_models()
    
    # Load example texts
    examples = load_example_texts()
    
    # Load cached images
    images = load_cached_images()
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["📝 Humor Detection", "📊 Model Performance", "📈 Zagreb Visualizations"])
    
    with tab1:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # User input section
            st.markdown("<h3 class='sub-header'>Test Your Text</h3>", unsafe_allow_html=True)
            
            # Example selector
            example_option = st.selectbox(
                "Select an example or enter your own text below:",
                ["-- Select an example --"] + examples
            )
            
            # Text input area
            if example_option != "-- Select an example --":
                user_text = st.text_area("Enter or edit text:", value=example_option, height=150)
            else:
                user_text = st.text_area("Enter text:", height=150, placeholder="Type your text here...")
            
        with col2:
            st.markdown("<h3 class='sub-header'>Options</h3>", unsafe_allow_html=True)
            
            show_graph = st.checkbox("Show semantic graph", value=True)
            show_zagreb = st.checkbox("Show Zagreb indices", value=True)
            show_all_models = st.checkbox("Show all model predictions", value=True)
        
        # Analysis button
        if st.button("Analyze Text", type="primary"):
            if not user_text:
                st.warning("Please enter some text to analyze.")
            else:
                with st.spinner("Analyzing text..."):
                    # Get predictions
                    predictions, graph, zagreb = get_prediction(
                        user_text, models, tokenizer, w2v_model, 
                        scaler, variance_selector, glove_embeddings, device
                    )
                    
                    # Display ensemble result
                    if "ensemble" in predictions:
                        result = predictions["ensemble"]["prediction"]
                        prob = predictions["ensemble"]["probability"]
                        
                        if result == 1:
                            st.markdown(f"""
                            <div class="result-box humor-positive">
                                <h3>This text is likely humorous! 😂</h3>
                                <p>Confidence: {prob:.2%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="result-box humor-negative">
                                <h3>This text is likely not humorous 😐</h3>
                                <p>Confidence: {(1-prob):.2%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Show all model predictions
                    if show_all_models:
                        st.markdown("<h3 class='sub-header'>Individual Model Predictions</h3>", 
                                   unsafe_allow_html=True)
                        
                        model_cols = st.columns(3)
                        
                        for i, (name, result) in enumerate(predictions.items()):
                            if name != "ensemble":
                                with model_cols[i % 3]:
                                    pred = result["prediction"]
                                    prob = result["probability"]
                                    
                                    st.markdown(f"""
                                    <div class="model-card">
                                        <h4>{name.upper()}</h4>
                                        <p>Prediction: {"😂 Humorous" if pred == 1 else "😐 Not humorous"}</p>
                                        <p>Confidence: {max(prob, 1-prob):.2%}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                    
                    # Display visualizations side by side
                    if show_graph or show_zagreb:
                        st.markdown("<h3 class='sub-header'>Text Analysis Visualizations</h3>", 
                                   unsafe_allow_html=True)
                        
                        viz_cols = st.columns(2)
                        
                        # Graph visualization
                        if show_graph and len(graph.nodes()) > 0:
                            with viz_cols[0]:
                                st.markdown('<div class="viz-container">', unsafe_allow_html=True)
                                st.subheader("Semantic Graph")
                                
                                upsilon_degrees = compute_upsilon_degrees_robust(graph)
                                graph_plt = visualize_graph(graph, upsilon_degrees)
                                if graph_plt:
                                    st.pyplot(graph_plt)
                                
                                # Display basic graph metrics
                                st.markdown(f"""
                                **Graph Metrics:**
                                - Nodes: {len(graph.nodes())}
                                - Edges: {len(graph.edges())}
                                - Density: {nx.density(graph):.4f}
                                """)
                                st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Zagreb indices visualization
                        if show_zagreb:
                            with viz_cols[1]:
                                st.markdown('<div class="viz-container">', unsafe_allow_html=True)
                                st.subheader("Zagreb Indices")
                                
                                zagreb_plt = visualize_zagreb_metrics(zagreb)
                                if zagreb_plt:
                                    st.pyplot(zagreb_plt)
                                
                                # Display key Zagreb values
                                st.markdown(f"""
                                **Key Zagreb Values:**
                                - First Zagreb (M1): {zagreb[0]:.2f}
                                - Second Zagreb (M2): {zagreb[1]:.2f}
                                - First Upsilon (M1_υ): {zagreb[9]:.2f}
                                - Second Upsilon (M2_υ): {zagreb[10]:.2f}
                                """)
                                st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown("<h3 class='sub-header'>Model Performance</h3>", unsafe_allow_html=True)
        
        # Display model performance metrics
        metrics_df = pd.DataFrame({
            'Model': ['SVM', 'Naive Bayes', 'MLP (Adam)', 'MLP (RMSprop)', 'Stacking', 'BERT', 'Ensemble'],
            'F1 Score': [0.7889, 0.7910, 0.7777, 0.7811, 0.8060, 0.8000, 0.8313],
            'Accuracy': [0.7980, 0.7875, 0.7850, 0.7906, 0.8160, 0.8055, 0.8383],
            'Precision': [0.8140, 0.7950, 0.8012, 0.8135, 0.8430, 0.8360, 0.8748],
            'Recall': [0.7650, 0.7872, 0.7555, 0.7510, 0.7720, 0.7670, 0.7919]
        })
        
        st.dataframe(metrics_df.style.highlight_max(axis=0), use_container_width=True)
        
        # Display performance plots
        performance_cols = st.columns(2)
        
        with performance_cols[0]:
            if 'performance_1' in images:
                st.image(images['performance_1'], caption="Model F1 Scores and Confusion Matrix", use_column_width=True)
            else:
                st.info("Performance visualization not available.")
        
        with performance_cols[1]:
            if 'performance_2' in images:
                st.image(images['performance_2'], caption="Model Accuracy and Metrics", use_column_width=True)
            else:
                st.info("Metrics visualization not available.")
    
    with tab3:
        st.markdown("<h3 class='sub-header'>Zagreb Indices Visualizations</h3>", unsafe_allow_html=True)
        st.markdown("""
        Zagreb indices are graph-based metrics that capture structural properties of semantic networks.
        These visualizations show the relationship between traditional Zagreb indices and their Upsilon
        variants, and how they differ between humorous and non-humorous texts.
        """)
        
        # Zagreb scatterplot
        if 'zagreb_scatter' in images:
            st.image(images['zagreb_scatter'], caption="Zagreb Indices Scatter Plot", use_column_width=True)
        
        # 3D visualization
        if 'zagreb_3d' in images:
            st.image(images['zagreb_3d'], caption="3D Visualization of Zagreb Indices", use_column_width=True)
        
        viz_cols = st.columns(2)
        
        # Correlation heatmap
        with viz_cols[0]:
            if 'zagreb_corr' in images:
                st.image(images['zagreb_corr'], caption="Zagreb Indices Correlation Heatmap", use_column_width=True)
            
        # Distribution plot
        with viz_cols[1]:
            if 'zagreb_dist' in images:
                st.image(images['zagreb_dist'], caption="Zagreb Indices Distribution by Class", use_column_width=True)
    
    # About section in sidebar
    st.sidebar.markdown("<h3 class='sub-header'>About</h3>", unsafe_allow_html=True)
    st.sidebar.markdown("""
    This application demonstrates a computational humor detection model using:
    
    - **Graph-based features**: Zagreb indices and Upsilon variants
    - **Machine learning**: Ensemble of SVM, Naive Bayes, MLPs
    - **Deep learning**: Fine-tuned BERT model
    - **Word embeddings**: Word2Vec and GloVe
    
    The model achieves 83.8% accuracy in detecting humor in text.
    """)
    
    # Display model architecture
    with st.sidebar.expander("Model Architecture"):
        st.markdown("""
        The system employs an ensemble of:
        1. SVM with RBF kernel
        2. Gaussian Naive Bayes
        3. MLP with Adam optimizer
        4. MLP with RMSprop optimizer
        5. Stacking Ensemble
        6. Fine-tuned BERT model
        
        Final prediction combines all model outputs.
        """)
    
    # Display features explanation
    with st.sidebar.expander("Features Explained"):
        st.markdown("""
        **Zagreb Indices**:
        - First Zagreb: Sum of squares of vertex degrees
        - Second Zagreb: Sum of products of adjacent vertex degrees
        - Upsilon variants: Modified versions with weighted importance
        
        **Other Features**:
        - Stylistic markers (capitalization, punctuation)
        - Humor-specific word presence
        - Embedding-based semantic features
        """)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    Developed using:
    - Streamlit
    - PyTorch & Transformers
    - NetworkX & Scikit-learn
    """)

if __name__ == "__main__":
    main()