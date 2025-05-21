import streamlit as st
import pickle, config
from utils import dataset_processing
import pandas as pd

vectorizer_pkl  = config.XAI_DATASET_vector

st.title("📄 Predict PDF Category")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    st.success(f"Uploaded file: {uploaded_file.name}")

    # Step 1: Extract text
    extracted_text = dataset_processing.extract_text_from_uploaded_pdf(uploaded_file)
    
    if extracted_text:
        st.subheader("📄 Extracted Text Preview")
        st.text(extracted_text[:1000])  # Preview first 1000 characters
    else:
        st.error("❌ Failed to extract text.")
        st.stop()

    # Step 2: Process text for prediction
    df_processed = dataset_processing.text_processing_for_prediction(extracted_text)
    st.divider()
    st.subheader("🧹 Cleaned & Processed Text")
    st.dataframe(df_processed)

    # Step 3: Load vectorizer and model
    with open(vectorizer_pkl, "rb") as f:
        tfidf_vectorizer = pickle.load(f)

    with open(config.XAI_MODEL_rf, "rb") as f:
        rf_model = pickle.load(f)

    # Step 4: Vectorize processed text
    tfidf_input = tfidf_vectorizer.transform(df_processed['Text_Data'])

    # Step 5: Predict
    prediction = rf_model.predict(tfidf_input)

    # Optional: map predicted class index to label
    label_map = ['Agreements', 'Taxes', 'Valuations']
    predicted_label = label_map[prediction[0]] if prediction[0] < len(label_map) else f"Class {prediction[0]}"

    # Step 6: Display result
    st.divider()
    st.subheader("🔮 Prediction Result")
    st.success(f"📌 Predicted Category: **{predicted_label}**")











    # -----------------------------
    # 📊 Prediction Confidence
    # -----------------------------
    if hasattr(rf_model, "predict_proba"):
        proba = rf_model.predict_proba(tfidf_input)[0]
        proba_df = pd.DataFrame({
            "Category": label_map,
            "Confidence": [f"{p:.2%}" for p in proba]
        })
        st.subheader("📊 Prediction Confidence")
        st.dataframe(proba_df)
    else:
        st.warning("Model does not support confidence scores (predict_proba missing).")

    # -----------------------------
    # 🔥 Top Features
    # -----------------------------
    st.subheader("🔥 Top Features (TF-IDF)")
    tfidf_values = tfidf_input.toarray()[0]
    feature_names = tfidf_vectorizer.get_feature_names_out()
    top_indices = tfidf_values.argsort()[::-1][:10]
    top_features = pd.DataFrame({
        "Term": [feature_names[i] for i in top_indices],
        "TF-IDF": [tfidf_values[i] for i in top_indices]
    })
    st.dataframe(top_features)

    # -----------------------------
    # 📚 Similar Docs
    # -----------------------------
    st.subheader("📚 Similar Documents from Training Set")
    try:
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        # Load X_train from HDF5
        X_train = pd.read_hdf(config.XAI_DATASET_finalframe_h5, key='X_train')
        raw_train_df = pd.read_pickle(config.XAI_DATASET_finalframe)

        similarities = cosine_similarity(tfidf_input, X_train)[0]
        top_doc_indices = np.argsort(similarities)[::-1][:3]

        similar_docs = pd.DataFrame({
            "Doc #": top_doc_indices,
            "Similarity": [f"{similarities[i]:.2%}" for i in top_doc_indices],
            "Category": raw_train_df.iloc[top_doc_indices]["Category"],
            "Snippet": raw_train_df.iloc[top_doc_indices]["Text_Data"].str[:150]
        })

        st.dataframe(similar_docs)
    except Exception as e:
        st.error(f"Failed to fetch similar documents: {e}")

    # -----------------------------
    # 📝 Logging
    # -----------------------------
    st.subheader("📝 Log Info")
    st.markdown(f"""
            - **TF-IDF Input Shape**: {tfidf_input.shape}
            - **Model**: {type(rf_model).__name__}
            - **Vectorizer**: {type(tfidf_vectorizer).__name__}
            - **PDF Text Length**: {len(extracted_text)} chars
            - **Top Term**: `{top_features.iloc[0]["Term"]}` with TF-IDF = {top_features.iloc[0]["TF-IDF"]:.4f}
            """)
