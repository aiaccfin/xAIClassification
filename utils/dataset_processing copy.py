import os
import io
import string
import numpy
import pandas as pd
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from pdfminer.converter import TextConverter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.high_level import extract_text

from io import BytesIO

import streamlit as st

from sklearn import preprocessing
from sklearn.svm import SVC

from utils.face_processing import load_faces_from_train_val_prod, load_faces_prod, load_faces_with_path


def convert_pdf_to_txt(path):
    filelist = os.listdir(path)
    documentcollection = []
    for files in filelist:
        files = os.path.join(path, files)
        documentcollection.append(files)
    for ifiles in documentcollection:
        if ifiles.endswith('.pdf') or ifiles.endswith('.PDF'):  # different extensions on the raw data
            with open(ifiles, 'rb') as fh:
                for page in PDFPage.get_pages(fh,
                                              caching=True,
                                              check_extractable=True):
                    resource_manager = PDFResourceManager()
                    fake_file_handle = io.StringIO()
                    converter = TextConverter(resource_manager, fake_file_handle)
                    page_interpreter = PDFPageInterpreter(resource_manager, converter)
                    page_interpreter.process_page(page)

                    text = fake_file_handle.getvalue()  # extraction of the text data
                    yield text

                    # closing open handles
                    converter.close()
                    fake_file_handle.close()


def text_processing(textcontents, category, identifier):
    dftaxes = pd.DataFrame(textcontents, columns=['Text_Data'])
    dftaxes['Category'] = category  # Adding the taxes label 'Taxes'

    # Pre-processing the extracted textual data
    dftaxes['Text_Data'] = dftaxes['Text_Data'].apply(lambda x: " ".join(x.lower() for x in x.split())) # lower case conversion
    dftaxes['Text_Data'] = dftaxes['Text_Data'].str.replace('[^\w\s]','') # getting rid of special characters
    dftaxes['Text_Data'] = dftaxes['Text_Data'].str.replace('\d+', '') # removing numeric values from between the words
    dftaxes['Text_Data'] = dftaxes['Text_Data'].apply(lambda x: x.translate(string.digits)) # removing numerical numbers
    stop = stopwords.words('english')
    dftaxes['Text_Data'] = dftaxes['Text_Data'].apply(lambda x: " ".join(x for x in x.split() if x not in stop)) #removing stop words
    stemmer = WordNetLemmatizer()
    dftaxes['Text_Data'] = [stemmer.lemmatize(word) for word in dftaxes['Text_Data']] #converting words to their dictionary form
    dftaxes['Text_Data'] = dftaxes['Text_Data'].str.replace('shall', '')
    # st.dataframe(dftaxes)

    taxfreq = pd.Series(' '.join(dftaxes['Text_Data']).split()).value_counts()[:5]
    # plt.figure(figsize=(10, 5))
    # taxfreq.plot(kind='barh')
    # plt.title('Top 5 Word Frequencies in Documents')
    # plt.xlabel('Frequency')
    # plt.ylabel('Words')
    #
    # st.pyplot(plt)

    dftaxes['Identifiers'] = identifier
    st.dataframe(dftaxes)
    return dftaxes


def save_dataset(datasize_name, train, val):
    with st.spinner('saving ...'):
        trainX, trainy = load_faces_from_train_val_prod(train)
        testX,  testy  = load_faces_from_train_val_prod(val)
        numpy.savez_compressed(datasize_name, trainX, trainy, testX, testy)
    return datasize_name


def load_dataset_with_metadata(dataset_name):
    with numpy.load(dataset_name, allow_pickle=True) as data:
        trainX = data['trainX']
        trainy = data['trainy']
        testX = data['testX']
        testy = data['testy']
        # metadata = data['metadata'].item()  # .item() to convert array to dictionary
    return trainX, trainy, testX, testy


def save_dataset_prod(dataset, face_folder):
    # save one set of dataset
    
    with st.spinner('saving ...'):
        trainX, trainy, file_names = load_faces_with_path(face_folder)
        st.write(file_names)
        numpy.savez_compressed(dataset, trainX, trainy, file_names)
    return dataset



def extract_text_from_uploaded_pdf(uploaded_file):
    # Use pdfminer.high_level for simplicity and robustness
    text = extract_text(BytesIO(uploaded_file.read()))
    return text


def text_processing_for_prediction(text):
    import string
    import pandas as pd
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    df = pd.DataFrame([text], columns=['Text_Data'])

    # Pre-processing the extracted textual data
    df['Text_Data'] = df['Text_Data'].apply(lambda x: " ".join(x.lower() for x in x.split())) # lowercase
    df['Text_Data'] = df['Text_Data'].str.replace('[^\w\s]', '', regex=True)  # remove punctuation
    df['Text_Data'] = df['Text_Data'].str.replace('\d+', '', regex=True)      # remove numbers
    stop = stopwords.words('english')
    df['Text_Data'] = df['Text_Data'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))  # remove stopwords
    stemmer = WordNetLemmatizer()
    df['Text_Data'] = df['Text_Data'].apply(lambda x: " ".join(stemmer.lemmatize(word) for word in x.split()))  # lemmatize
    df['Identifiers'] = 'user_upload'
    return df
