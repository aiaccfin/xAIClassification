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

from pdf2image import convert_from_path

from PIL import Image

from io import BytesIO

import streamlit as st

from sklearn import preprocessing
from sklearn.svm import SVC

from utils.face_processing import load_faces_from_train_val_prod, load_faces_prod, load_faces_with_path


def convert_files_to_text(path):
    filelist = os.listdir(path)
    for filename in filelist:
        file_path = os.path.join(path, filename)
        ext = filename.lower()

        if ext.endswith('.pdf'):
            # Try extracting text-based first
            try:
                with open(file_path, 'rb') as fh:
                    for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
                        resource_manager = PDFResourceManager()
                        fake_file_handle = io.StringIO()
                        converter = TextConverter(resource_manager, fake_file_handle)
                        page_interpreter = PDFPageInterpreter(resource_manager, converter)
                        page_interpreter.process_page(page)
                        text = fake_file_handle.getvalue()

                        converter.close()
                        fake_file_handle.close()

                        if text.strip():
                            yield text
                        else:
                            raise ValueError("Empty page detected")

            except:
                # If above fails, fallback to OCR (image-based PDF)
                images = convert_from_path(file_path)
                for image in images:
                    text = pytesseract.image_to_string(image)
                    if text.strip():
                        yield text

        elif ext.endswith(('.jpg', '.jpeg', '.png')):
            try:
                image = Image.open(file_path)
                text = pytesseract.image_to_string(image)
                if text.strip():
                    yield text
            except Exception as e:
                print(f"OCR failed for image {filename}: {e}")

# def convert_pdf_to_txt(path):
#     filelist = os.listdir(path)
#     documentcollection = []
#     for files in filelist:
#         files = os.path.join(path, files)
#         documentcollection.append(files)
#     for ifiles in documentcollection:
#         if ifiles.endswith('.pdf') or ifiles.endswith('.PDF'):  # different extensions on the raw data
#             with open(ifiles, 'rb') as fh:
#                 for page in PDFPage.get_pages(fh,
#                                               caching=True,
#                                               check_extractable=True):
#                     resource_manager = PDFResourceManager()
#                     fake_file_handle = io.StringIO()
#                     converter = TextConverter(resource_manager, fake_file_handle)
#                     page_interpreter = PDFPageInterpreter(resource_manager, converter)
#                     page_interpreter.process_page(page)

#                     text = fake_file_handle.getvalue()  # extraction of the text data
#                     yield text

#                     # closing open handles
#                     converter.close()
#                     fake_file_handle.close()


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



# def extract_text_from_uploaded_pdf(uploaded_file):
#     # Use pdfminer.high_level for simplicity and robustness
#     text = extract_text(BytesIO(uploaded_file.read()))
#     return text


from pdf2image import convert_from_bytes
import pytesseract

def extract_text_from_uploaded_pdf(uploaded_file):
    # Try extracting text with pdfminer
    try:
        uploaded_file.seek(0)  # Ensure pointer is at start
        text = extract_text(BytesIO(uploaded_file.read()))
        if text.strip():  # If there's actual text content
            return text
    except Exception as e:
        st.warning(f"Text extraction with pdfminer failed: {e}")

    # Fallback to OCR if no text found
    st.info("Falling back to OCR for image-based PDF.")
    uploaded_file.seek(0)
    try:
        images = convert_from_bytes(uploaded_file.read())
        ocr_text = ""
        for img in images:
            ocr_text += pytesseract.image_to_string(img)
        return ocr_text
    except Exception as e:
        st.error(f"OCR extraction failed: {e}")
        return ""



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
