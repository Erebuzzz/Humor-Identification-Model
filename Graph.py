import json
import networkx as nx
from pyvis.network import Network
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import os

nltk.download('punkt')
nltk.download('stopwords')

# === CONFIGURATION ===
DATASET_PATH = r'D:\Humor Detection\yelp_dataset\yelp_academic_dataset_review.json'
NUM_REVIEWS = 5  # Number of reviews to visualize (change as needed)
OUTPUT_DIR = "dynamic_graphs"

# === DATA LOADING & PREPROCESSING ===
def load_reviews(path, num_reviews=NUM_REVIEWS):
    reviews = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_reviews:
                break
            data = json.loads(line)
            text = data.get('text', '')
            reviews.append(text)
    return reviews

stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [w for w in tokens if w.isalpha() and w not in stop_words]
    return tokens

reviews = load_reviews(DATASET_PATH)
tokenized_reviews = [preprocess(review) for review in reviews]

# === GRAPH CONSTRUCTION ===
def build_cooccurrence_graph(tokens, window=2):
    G = nx.Graph()
    for i, word in enumerate(tokens):
        for j in range(i+1, min(i+window, len(tokens))):
            G.add_edge(word, tokens[j])
    return G

# === DYNAMIC VISUALIZATION WITH PYVIS ===
def visualize_dynamic_graph(tokens, html_filename="dynamic_graph.html"):
    G = build_cooccurrence_graph(tokens)
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", notebook=False)
    net.from_nx(G)
    net.show_buttons(filter_=['physics'])
    net.show(html_filename, notebook=False)
    print(f"Dynamic graph saved as {html_filename}. Open it in your browser to interact.")

# === MAIN LOOP: VISUALIZE MULTIPLE REVIEWS ===
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

for idx, tokens in enumerate(tokenized_reviews):
    print(f"\nReview {idx} tokens: {tokens}")
    html_file = os.path.join(OUTPUT_DIR, f"dynamic_graph_review_{idx}.html")
    visualize_dynamic_graph(tokens, html_filename=html_file)
