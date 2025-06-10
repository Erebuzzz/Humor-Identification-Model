# app.py

import streamlit as st
import pickle
import numpy as np
import torch
from transformers import BertTokenizer
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold

# Load the models and preprocessors
@st.cache(allow_output_mutation=True)
def load_models():
    models = {}
    with open("saved_models_new/svm_model.pkl", 'rb') as f:
        models['svm'] = pickle.load(f)
    with open("saved_models_new/naive_bayes_model.pkl", 'rb') as f:
        models['naive_bayes'] = pickle.load(f)
    with open("saved_models_new/mlp_adam_model.pkl", 'rb') as f:
        models['mlp_adam'] = pickle.load(f)
    with open("saved_models_new/mlp_rmsprop_model.pkl", 'rb') as f:
        models['mlp_rmsprop'] = pickle.load(f)
    with open("saved_models_new/stacking_ensemble_model.pkl", 'rb') as f:
        models['stacking_ensemble'] = pickle.load(f)
    with open("saved_models_new/bert_best_model.pth", 'rb') as f:
        models['bert'] = torch.load(f)
    return models

@st.cache
def load_preprocessors():
    with open("saved_models_new/preprocessors.pkl", 'rb') as f:
        scaler, variance_selector = pickle.load(f)
    return scaler, variance_selector

# Load GloVe and Word2Vec embeddings if needed
# (Assuming you have functions to load these embeddings)

# Function to preprocess input text
def preprocess_input(text, tokenizer, scaler, variance_selector):
    # Tokenization and feature extraction logic here
    # This should match the feature extraction used during training
    tokens = tokenizer(text, truncation=True, padding=True, max_length=256, return_tensors="pt")
    # Extract features and scale them
    features = extract_features_enhanced(tokens)  # Implement this function based on your model
    features = scaler.transform(features)
    features = variance_selector.transform(features)
    return features

# Function to get predictions
def get_predictions(model, features):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    else:
        return model.predict(features)

# Streamlit UI
st.title("Humor Identification Model")
st.markdown("### Test the Humor Identification Model")

# Input text area for user to enter text
input_text = st.text_area("Enter your text here:", height=200)

# Load models and preprocessors
models = load_models()
scaler, variance_selector = load_preprocessors()
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")  # Adjust path if needed

if st.button("Classify"):
    if input_text:
        # Preprocess input
        features = preprocess_input(input_text, tokenizer, scaler, variance_selector)
        
        # Get predictions from each model
        predictions = {}
        for name, model in models.items():
            preds = get_predictions(model, features)
            predictions[name] = preds
        
        # Display results
        st.markdown("### Prediction Results:")
        for name, pred in predictions.items():
            st.write(f"{name}: {'Humorous' if pred > 0.5 else 'Not Humorous'} (Probability: {pred:.2f})")
    else:
        st.warning("Please enter some text to classify.")

# Optional: Display additional information or visualizations
st.markdown("### Model Overview")
st.write("This application uses various models to classify text as humorous or not humorous. You can enter any text in the box above to see the predictions.")

# Optional: Add more features or visualizations as needed

if __name__ == "__main__":
    st.run()