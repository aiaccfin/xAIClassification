import streamlit as st, os
from streamlit_extras.stateful_button import button
from utils import streamlit_components, face_pipline, image_processing, embedding_processing 

import os
import pandas as pd
import matplotlib.pyplot as plt

import pickle, requests, json

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB

streamlit_components.streamlit_ui('🦣 Naive Bayes Classifier')

model_nbc  = os.getenv('XAI_MODEL_nbc')
dataset_h5 = os.getenv('XAI_DATASET_finalframe_h5')
X_train = pd.read_hdf(dataset_h5 , key='X_train')
X_test = pd.read_hdf(dataset_h5, key='X_test')
y_train = pd.read_hdf(dataset_h5, key='y_train')
y_test = pd.read_hdf(dataset_h5, key='y_test')

my_tags = ['Agreements', 'Taxes','Valuations']

if button("Naive Bayes Classifier?", key="button3"):


    clf = MultinomialNB().fit(X_train, y_train)

    pickle.dump(clf, open(model_nbc, 'wb'))


    y_pred = clf.predict(X_test)
    st.text('Accuracy: %s' % accuracy_score(y_pred, y_test))
    st.text(classification_report(y_test, y_pred, target_names=my_tags))

    conf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    st.text('Confusion matrix:\n')
    st.text(conf_mat)

    labels = ['Agreement', 'Taxes', 'Valuations']

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)

    plt.xlabel('Predicted')
    plt.ylabel('Expected')
    st.pyplot(fig)
    