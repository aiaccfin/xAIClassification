import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path('./')
PDF_DIR = BASE_DIR / 'data/pdf'
MODEL_DIR = BASE_DIR / 'models'

# === xAI Paths ===
XAI_DATASET_FOLDER_tax       = PDF_DIR / "Taxes"
XAI_DATASET_FOLDER_valuation = PDF_DIR / "Valuations"
XAI_DATASET_FOLDER_agreement = PDF_DIR / "Agreements"

XAI_DATASET_finalframe   = PDF_DIR / "classification05.pkl"
XAI_DATASET_finalframe_h5= PDF_DIR / "classification05.h5"
XAI_DATASET_vector       = PDF_DIR / "classification05_tfidf.pkl"

XAI_MODEL_rf    = MODEL_DIR / "xai_rf.pkl"

# #Global
# TRAIN=./data/train/
# VAL  =./data/val/
# PROD = ./data/prod/
# PENDING = ./data/prod/pending/

# FACENET_MODEL = ./models/facenet_keras_2024.h5
# DATA_IMAGES=./data/images/
# YCC=./data/train/yangchenchen/

# # SVC
# DATASET_NAME=./data/embeddings/ycc-wxy-syz-faces-dataset.npz
# EMBEDDINGS_NAME=./data/embeddings/ycc-wxy-syz-faces-embeddings.npz

# # Training
# TRAINING_DATASET_ywsd   =./data/embeddings/training-dataset-ywsd.npz
# TRAINING_EMBEDDINGS_ywsd=./data/embeddings/training-embeddings-ywsd.npz

# # Production
# PROD_DATASET_ywsd   =./data/embeddings/prod-dataset-ywsd.npz
# PROD_EMBEDDINGS_ywsd=./data/embeddings/prod-embeddings-ywsd.npz

# #OneRest
# ONE_TRAIN_FOLDER = ./onerest_data/train/
# ONE_TEST_FOLDER  = ./onerest_data/test/

# #sunyunzhu
# # ONE_DATASET_TRAINING    =./onerest_data/embeddings/one-training-dataset-cheng.npz
# # ONE_EMBEDDINGS_TRAINING =./onerest_data/embeddings/one-training-embeddings-cheng.npz
# ONE_DATASET_TRAINING    =./onerest_data/embeddings/one-training-dataset-vicky.npz
# ONE_EMBEDDINGS_TRAINING =./onerest_data/embeddings/one-training-embeddings-vicky.npz
# ONE_DATASET_TESTING    =./onerest_data/embeddings/one-testing-dataset.npz
# ONE_EMBEDDINGS_TESTING =./onerest_data/embeddings/one-testing-embeddings.npz
# ONE_TESTING_FOLDER = ./onerest_data/test/testing/
# ONE_TESTING_SUBFOLDER = ./onerest_data/test/testing/sub