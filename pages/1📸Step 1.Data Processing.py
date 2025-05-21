import streamlit as st
from streamlit_extras.stateful_button import button

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import pandas as pd

import config
from utils import streamlit_components, dataset_processing


streamlit_components.streamlit_ui('🦣 PDF Processing to pkl')

dataset_tax    = config.XAI_DATASET_FOLDER_tax
dataset_valuation= config.XAI_DATASET_FOLDER_valuation
dataset_agreement= config.XAI_DATASET_FOLDER_agreement

dataset_pkl = config.XAI_DATASET_finalframe

st.text(f"dataset tax1 folder: {dataset_tax}")
st.text(f"dataset valuation folder: {dataset_valuation}")
st.text(f"dataset agreement folder: {dataset_agreement}")
st.text(f"pkl: {dataset_pkl}")

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