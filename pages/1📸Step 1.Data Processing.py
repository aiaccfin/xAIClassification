import streamlit as st
from streamlit_extras.stateful_button import button

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import os
import pandas as pd

from utils import streamlit_components, face_pipline, image_processing, face_processing
from utils import dataset_processing, streamlit_components, image_processing, face_processing
from utils import streamlit_components, face_pipline, image_processing, embedding_processing

streamlit_components.streamlit_ui('🦣 PDF Processing to pkl')

dataset_tax    = os.getenv('XAI_DATASET_FOLDER_tax')
dataset_valuation= os.getenv('XAI_DATASET_FOLDER_valuation')
dataset_agreement= os.getenv('XAI_DATASET_FOLDER_agreement')

dataset_pkl = os.getenv('XAI_DATASET_finalframe')


embeddings = os.getenv('PROD_EMBEDDINGS_ywsd')
model      = os.getenv('FACENET_MODEL')

st.text(f"dataset tax folder: {dataset_tax}")
st.text(f"dataset valuation folder: {dataset_valuation}")
st.text(f"dataset agreement folder: {dataset_agreement}")

if button("Convert?", key="but5"):
    textcontents_tax = dataset_processing.convert_pdf_to_txt(dataset_tax)
    df_tax = dataset_processing.text_processing(textcontents_tax, 'Taxes', 'tax,agreement,section,group,date')

    textcontents_valuation = dataset_processing.convert_pdf_to_txt(dataset_valuation)
    df_valuation = dataset_processing.text_processing(textcontents_valuation, 'Valuations', 'valuation,value,report,market,level')

    textcontents_agreement = dataset_processing.convert_pdf_to_txt(dataset_agreement)
    df_agreement = dataset_processing.text_processing(textcontents_agreement, 'Agreement', 'agreement,subcontractor,contractor,work,subcontract')

    frames = [df_tax, df_agreement, df_valuation]
    finalframe = pd.concat(frames, sort=False)
    finalframe = finalframe[['Identifiers', 'Text_Data', 'Category']]
    finalframe = finalframe.reset_index(drop=True)
    finalframe.to_pickle(dataset_pkl)
    st.dataframe(finalframe)