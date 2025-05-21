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



# # predict_pdf.py
# import streamlit as st
# from utils import dataset_processing

# st.title("📄 Predict PDF Category")

# uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# if uploaded_file is not None:
#     st.success(f"Uploaded file: {uploaded_file.name}")

#     extracted_text = dataset_processing.extract_text_from_uploaded_pdf(uploaded_file)
    
#     if extracted_text:
#         st.subheader("📄 Extracted Text Preview")
#         st.text(extracted_text[:1000])  # Show only the first 1000 chars
#     else:
#         st.error("❌ Failed to extract text.")

#     df_processed = dataset_processing.text_processing_for_prediction(extracted_text)
#     st.divider()
#     st.dataframe(df_processed)
