# app.py

import streamlit as st
import pickle
import numpy as np
import torch
from transformers import BertTokenizer
import matplotlib.pyplot as plt
import seaborn as sns

# Load models and preprocessors
MODEL_SAVE_DIR = "saved_models_new"
with open(f"{MODEL_SAVE_DIR}/preprocessors.pkl", 'rb') as f:
    scaler, variance_selector = pickle.load(f)

# Load the models
models = {}
model_names = ['svm', 'naive_bayes', 'mlp_adam', 'mlp_rmsprop', 'stacking_ensemble']

for model_name in model_names:
    with open(f"{MODEL_SAVE_DIR}/{model_name}_model.pkl", 'rb') as f:
        models[model_name] = pickle.load(f)

# Load BERT model
BERT_LOCAL_PATH = r"C:\Users\kshit\OneDrive\Documents\GitHub\Humor-Identification-Model\bert-base-uncased"
bert_tokenizer = BertTokenizer.from_pretrained(BERT_LOCAL_PATH)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Streamlit app layout
st.title("Humor Identification Model")
st.markdown("### Test the Humor Identification Model")

# Model selection
selected_model = st.selectbox("Select Model", model_names)

# Text input for testing
user_input = st.text_area("Enter your text here:", height=150)

if st.button("Predict"):
    if user_input:
        # Preprocess input
        tokens = user_input.split()  # Simple tokenization
        features = extract_features_enhanced(tokens, user_input)  # Use your feature extraction function
        features = np.nan_to_num(features, nan=0.0)

        # Scale features
        features = scaler.transform([features])
        features = variance_selector.transform(features)

        # Get predictions
        if selected_model in models:
            model = models[selected_model]
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(features)
                prediction = probs[0][1]  # Probability of humorous
                label = "Humorous" if prediction > 0.5 else "Not Humorous"
                st.success(f"Prediction: {label} (Confidence: {prediction:.2f})")
            else:
                prediction = model.predict(features)
                label = "Humorous" if prediction[0] == 1 else "Not Humorous"
                st.success(f"Prediction: {label}")

        # BERT model prediction
        if selected_model == 'bert':
            inputs = bert_tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=192).to(DEVICE)
            with torch.no_grad():
                outputs = models['bert'](inputs['input_ids'], attention_mask=inputs['attention_mask'])
                probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
                prediction = probs[0][1]
                label = "Humorous" if prediction > 0.5 else "Not Humorous"
                st.success(f"Prediction: {label} (Confidence: {prediction:.2f})")

# Visualization section
st.markdown("### Zagreb Indices Visualization")
if st.button("Show Zagreb Indices"):
    # Load Zagreb indices data (you may need to adjust this part)
    trad_zagreb, upsilon_zagreb, labels_viz = export_zagreb_indices()  # Ensure this function is defined

    if len(trad_zagreb) > 0 and len(upsilon_zagreb) > 0:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot traditional Zagreb indices
        sns.boxplot(data=trad_zagreb, ax=ax[0])
        ax[0].set_title("Traditional Zagreb Indices")
        ax[0].set_xticklabels(["M1", "M2", "Co1", "Co2", "Gen", "Mod", "Third", "Hyper", "Forgotten"])
        
        # Plot Upsilon Zagreb indices
        sns.boxplot(data=upsilon_zagreb, ax=ax[1])
        ax[1].set_title("Upsilon Zagreb Indices")
        ax[1].set_xticklabels(["M1_υ", "M2_υ", "M3_υ"])
        
        st.pyplot(fig)
    else:
        st.warning("No Zagreb indices data available.")

# Footer
st.markdown("### About")
st.markdown("This application showcases a humor identification model using various machine learning techniques.")