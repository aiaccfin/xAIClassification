import streamlit as st
from utils import streamlit_components

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from subprocess import Popen, PIPE
from pdfminer.layout import LAParams
import io
from io import StringIO
import os
import glob
import comtypes.client
import sys

from dotenv import load_dotenv
load_dotenv()

streamlit_components.streamlit_ui('🐬🦣 xAI PDF Classification 🍃🦭')


tab1, tab2 = st.tabs(["General",''])

with tab1:
    streamlit_components.general()
