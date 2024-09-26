import streamlit as st, os
from streamlit_extras.stateful_button import button

import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


from utils import streamlit_components, face_pipline, image_processing, face_processing


dataset_pkl = os.getenv('XAI_DATASET_finalframe')
finalframe = pd.read_pickle(dataset_pkl)

streamlit_components.streamlit_ui('🦣 Term Frequency-Inverse Doc Frequency')

if button("TF-IDF?", key="button1"):
    # Converting the text data into vectors using TF-IDF
    # Generating 1000 features for the input for the model
    tfidfconverter = TfidfVectorizer(max_features=1000, stop_words=stopwords.words('english'))
    X = pd.DataFrame(tfidfconverter.fit_transform(finalframe['Text_Data']).toarray())
    st.dataframe(X)
    # X.columns = range(X.shape[1])
    labelencoder = LabelEncoder()  # Converting the labels to numeric labels
    y = labelencoder.fit_transform(finalframe['Category'])

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)