# predict_pdf.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime
from sklearn.metrics.pairwise import cosine_similarity

from utils import dataset_processing
import config

# --- Load resources
with open(config.XAI_MODEL_rf, "rb") as f:
    rf_model = pickle.load(f)

with open(config.XAI_DATASET_vector, "rb") as f:
    tfidf_vectorizer = pickle.load(f)

finalframe = pd.read_pickle(config.XAI_DATASET_finalframe)

# --- Streamlit UI
st.title("📄 Predict PDF Category")
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    st.success(f"Uploaded file: {uploaded_file.name}")

    extracted_text = dataset_processing.extract_text_from_uploaded_pdf(uploaded_file)

    if not extracted_text:
        st.error("❌ Failed to extract text.")
        st.stop()

    st.subheader("📄 Extracted Text Preview")
    st.text(extracted_text[:1000])  # Show only first 1000 chars

    df_processed = dataset_processing.text_processing_for_prediction(extracted_text)
    st.divider()
    st.subheader("🧹 Preprocessed Data")
    st.dataframe(df_processed)

    # --- Vectorize
    tfidf_input = tfidf_vectorizer.transform(df_processed["Text_Data"])
    
    # --- Predict
    prediction = rf_model.predict(tfidf_input)[0]
    probabilities = rf_model.predict_proba(tfidf_input)[0]
    class_labels = rf_model.classes_

    st.divider()
    st.subheader("🎯 Prediction Result")
    st.markdown(f"### 🏷️ **Predicted Category: `{prediction}`**")

    # --- Probabilities
    st.subheader("📊 Prediction Confidence")
    for label, prob in zip(class_labels, probabilities):
        st.write(f"**{label}**: {prob:.2%}")

    # --- Top Feature Importances
    st.subheader("🔥 Top Features Learned by Model")

    try:
        feature_names = tfidf_vectorizer.get_feature_names_out()
        importances = rf_model.feature_importances_
        top_n = 20
        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [(feature_names[i], importances[i]) for i in indices]

        for feature, score in top_features:
            st.write(f"**{feature}**: {score:.4f}")
    except:
        st.warning("Feature importances are not available in this model.")

    # --- Similarity to training documents
    st.subheader("📚 Most Similar Documents from Training Set")

    X_train_tfidf = tfidf_vectorizer.transform(finalframe['Text_Data'])
    similarities = cosine_similarity(tfidf_input, X_train_tfidf)[0]
    top_k_idx = similarities.argsort()[-3:][::-1]
    similar_docs = finalframe.iloc[top_k_idx]

    for idx, row in similar_docs.iterrows():
        st.markdown(f"#### 📄 Example {idx}")
        st.markdown(f"**Category**: `{row['Category']}`")
        st.markdown("**Excerpt:**")
        st.text(row['Text_Data'][:500])
        st.markdown("---")

    # --- Save prediction log
    st.subheader("📝 Logging Result")
    log = {
        "filename": uploaded_file.name,
        "predicted_label": prediction,
        "probabilities": dict(zip(class_labels, probabilities)),
        "text_excerpt": extracted_text[:500],
        "timestamp": datetime.datetime.now().isoformat(),
    }

    with open("prediction_logs.jsonl", "a", encoding="utf-8") as f:
        import json
        f.write(json.dumps(log) + "\n")

    st.success("✅ Prediction logged.")
