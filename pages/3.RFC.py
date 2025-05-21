import streamlit as st
from streamlit_extras.stateful_button import button
from utils import dataset_processing, streamlit_components, image_processing, face_processing

import os, config
import pandas as pd
import matplotlib.pyplot as plt

import pickle, requests, json

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

streamlit_components.streamlit_ui('🦣 Random Forest Classifier')

model_rf   = config.XAI_MODEL_rf

dataset_h5  = config.XAI_DATASET_finalframe_h5

X_train = pd.read_hdf(dataset_h5 , key='X_train')
X_test  = pd.read_hdf(dataset_h5, key='X_test')
y_train = pd.read_hdf(dataset_h5, key='y_train')
y_test  = pd.read_hdf(dataset_h5, key='y_test')

my_tags = ['Agreements', 'Taxes','Valuations']

if button("Run?", key="button2"):

    classifier = RandomForestClassifier(n_estimators=1200, random_state=1)  # defining 1000 nodes
    rf = classifier.fit(X_train, y_train)

    pickle.dump(rf, open(model_rf, 'wb'))

    y_pred = classifier.predict(X_test)


    st.text('Accuracy: %s' % accuracy_score(y_test, y_pred))
    st.text(classification_report(y_test, y_pred, target_names=my_tags))

    conf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    st.text('Confusion matrix:')
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