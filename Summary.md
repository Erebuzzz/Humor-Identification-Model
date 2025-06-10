# Humor Identification Model: Using Graph-based Zagreb Indices with BERT

## Project Summary

This project develops an advanced computational model to automatically identify humor in text using a novel combination of graph-based features, specifically Zagreb indices and their Upsilon variants, together with state-of-the-art language models. The model demonstrates excellent performance, achieving over 83% accuracy in humor detection on a balanced dataset of Yelp reviews.

## Key Features

- **Novel Application of Zagreb Indices**: Utilizes both traditional Zagreb indices and their Upsilon variants to capture the structural properties of text represented as semantic graphs
- **Enhanced BERT Integration**: Implements a fine-tuned BERT model with balanced class training and gradient accumulation
- **Ensemble Learning**: Combines multiple machine learning approaches (SVM, Naive Bayes, MLPs, BERT) for optimal performance
- **Comprehensive Visualization**: Provides detailed visualizations of the relationship between graph indices and humor

## Dataset

The project uses the Yelp Academic Dataset, from which 10,000 reviews were selected (5,000 humorous and 5,000 non-humorous). After preprocessing and outlier removal, 8,068 reviews were used with the following splits:
- Training set: 4,840 reviews
- Validation set: 1,614 reviews
- Test set: 1,614 reviews

## Methodology

### Text Preprocessing
- Custom tokenization with stopword removal
- Outlier removal using 10th and 90th percentiles of token lengths
- Embedding generation using Word2Vec and GloVe

### Feature Engineering
- **Graph-Based Features**: Conversion of text to semantic graphs with multiple window sizes
- **Zagreb Indices**: 
  - 9 Traditional Zagreb indices (First Zagreb, Second Zagreb, co-indices, etc.)
  - 3 Upsilon Zagreb indices capturing deeper structural properties
- **Stylistic Features**: Capitalization, punctuation, word length, humor-related words
- **Embedding Features**: Statistical aggregations of word embeddings

### Model Architecture
The system employs an ensemble of various classifiers:
1. Support Vector Machine (SVM) with RBF kernel
2. Gaussian Naive Bayes
3. Multi-Layer Perceptron with Adam optimizer
4. Multi-Layer Perceptron with RMSprop optimizer
5. Stacking Ensemble (Random Forest, Gradient Boosting, Logistic Regression)
6. Fine-tuned BERT model

### BERT Enhancement
The BERT model was significantly improved with:
- Balanced class training (1,996 samples equally distributed)
- Increased sequence length (192 tokens vs standard 128)
- Larger effective batch size through gradient accumulation
- OneCycleLR scheduling
- Multiple training epochs with best model selection

## Results

### Overall Performance

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 83.83% |
| F1 Score  | 83.13% |
| Precision | 87.48% |
| Recall    | 79.19% |

### Individual Model Performance

| Model             | F1 Score | Accuracy |
|-------------------|----------|----------|
| SVM               | 78.89%   | 79.80%   |
| Naive Bayes       | 79.10%   | 78.75%   |
| MLP (Adam)        | 77.77%   | 78.50%   |
| MLP (RMSprop)     | 78.11%   | 79.06%   |
| Stacking Ensemble | 80.60%   | 81.60%   |
| BERT              | 80.00%   | 80.55%   |
| **Final Ensemble**| **83.13%**| **83.83%**|

## Visualizations

### Model Performance Comparison

![Model Performance Comparison](results/performance_plots_1.png)

The left chart shows F1 scores across all models, with the ensemble outperforming individual models. The right chart displays the confusion matrix for the ensemble model.

### Accuracy and Performance Metrics

![Accuracy and Performance Metrics](results/performance_plots_2.png)

The left chart compares accuracy across all models, while the right chart displays the four primary performance metrics for the ensemble model.

### Zagreb Indices Visualization

![Zagreb Scatter Plot](results/zagreb_viz/zagreb_scatter_1.png)

Scatter plots showing the relationship between traditional and Upsilon Zagreb indices, colored by humor class (red = humorous, blue = non-humorous).

### 3D Visualization of Zagreb Indices

![3D Zagreb Visualization](results/zagreb_viz/zagreb_scatter_3d.png)

Three-dimensional visualization of traditional Zagreb indices (left) and Upsilon Zagreb indices (right), revealing clustering patterns of humorous vs. non-humorous texts.

### Correlation Heatmap

![Zagreb Correlation Heatmap](results/zagreb_viz/zagreb_correlation.png)

Correlation analysis between different Zagreb indices, showing strong correlations between certain traditional and Upsilon indices.

### Distribution by Class

![Zagreb Distributions](results/zagreb_viz/zagreb_distributions.png)

Distribution plots comparing the values of traditional and Upsilon Zagreb indices between humorous and non-humorous texts.

## Key Findings

1. **Zagreb Indices are Effective for Humor Detection**: Both traditional and Upsilon Zagreb indices show discriminative power in distinguishing humorous from non-humorous text.

2. **Upsilon Variants Add Value**: The Upsilon variants of Zagreb indices capture structural properties not present in traditional indices, particularly for humor identification.

3. **Ensemble Approach Outperforms Single Models**: The combination of different modeling approaches yields superior performance compared to any individual model.

4. **BERT Benefits from Balanced Training**: The enhanced BERT model with balanced class training shows significant improvement over standard implementation.

5. **Structural Properties Matter**: The correlation between graph structural properties and humor suggests that the semantic structure of text is an important factor in humor identification.

## Technical Implementation

### Zagreb Indices
The project implements both traditional Zagreb indices and their Upsilon variants:

```python
def compute_zagreb_indices_enhanced(G):
    """Enhanced computation with only 3 primary Upsilon indices"""
    if not G.edges() or len(G.nodes()) < 2:
        return [0.0] * 12  # 9 traditional + 3 upsilon indices
    
    degrees = dict(G.degree())
    edges = list(G.edges())
    
    # Traditional Zagreb indices
    m1 = sum(degrees[u]**2 for u in G.nodes())
    m2 = sum(degrees[u] * degrees[v] for u, v in edges)
    
    # Co-indices
    m1_co = sum(degrees[u] + degrees[v] for u, v in non_edges)
    m2_co = sum(degrees[u] * degrees[v] for u, v in non_edges)
    
    # Other traditional indices
    generalized = sum(degrees[u]**2 + degrees[v]**2 for u, v in edges)
    modified = sum(1.0 / max(degrees[u], 1) for u in G.nodes())
    third = sum(degrees[u]**3 for u in G.nodes())
    hyper = sum((degrees[u] + degrees[v])**2 for u, v in edges)
    forgotten = third
    
    # Upsilon indices
    upsilon_degrees = compute_upsilon_degrees_robust_FIXED(G)
    upsilon_values = list(upsilon_degrees.values())
    M1_upsilon = sum(v**2 for v in upsilon_values)
    M2_upsilon = sum(upsilon_degrees[u] * upsilon_degrees[v] for u, v in edges)
    M3_upsilon = sum(upsilon_degrees[u] + upsilon_degrees[v] for u, v in edges)
    
    return [m1, m2, m1_co, m2_co, generalized, modified, third, hyper, forgotten, 
            M1_upsilon, M2_upsilon, M3_upsilon]
```

### Enhanced BERT Training
The BERT model training was enhanced with the following implementation:

```python
def train_enhanced_bert():
    """Enhanced BERT training with better parameters and larger batch size"""
    bert_tokenizer = BertTokenizer.from_pretrained(BERT_LOCAL_PATH)
    bert_model = BertForSequenceClassification.from_pretrained(BERT_LOCAL_PATH, num_labels=2).to(DEVICE)
    
    # Balance the classes
    min_samples = min(len(humor_indices), len(non_humor_indices))
    balanced_indices = np.concatenate([
        np.random.choice(humor_indices, min_samples, replace=False),
        np.random.choice(non_humor_indices, min_samples, replace=False)
    ])
    
    # Larger batch size and gradient accumulation
    batch_size = 24  # Increased from 16
    accum_steps = 2  # Effective batch size of 48
    
    # Better optimizer settings with warmup
    optimizer = torch.optim.AdamW(bert_model.parameters(), lr=3e-5, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-5, total_steps=total_steps, pct_start=0.1
    )
    
    # Training loop with gradient accumulation
    for epoch in range(4):
        for batch_idx, batch in enumerate(train_loader):
            loss = outputs.loss / accum_steps  # Scale for gradient accumulation
            loss.backward()
            
            # Update only after accumulation steps
            if (batch_idx + 1) % accum_steps == 0 or batch_idx == len(train_loader) - 1:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
```