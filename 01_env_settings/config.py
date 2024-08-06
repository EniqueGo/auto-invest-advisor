# config.py

"""
@author: Esther Xu
Code tested on:
    - os: Debian 11
    - Python: 3.10.14; 3.11.2
    - Numpy: 1.25.2
    - Sklearn: 1.3.0
    - Tensorflow: 2.13.0; 2.14.0
    - Keras: 2.8; 2.14.0
    - Hadoop 3.3
    - Spark 3.3
    - Spark-nlp: 4.2.0
    - Prophet: 1.1.5

"""

#==================================System parameter setting==================================

# Local ENV Params Setting
ENV_LOCAL = "LOC"
DATA_FILE_DIR_LOCAL = "/Users/estherx/Documents/EstherxScripts/python/stonkgo/stonkgo_v2.2.0_GCP/0_data"
BITCOIN_DATA_FILE_PATH_LOCAL = f"{DATA_FILE_DIR_LOCAL}/bitcoin_data.csv"
SENTIMENT_DATA_FILE_PATH_LOCAL = f"{DATA_FILE_DIR_LOCAL}/train_test_lstm_reddit_data.csv"
BEST_MODEL_DIR_LOCAL=f"{DATA_FILE_DIR_LOCAL}"
PREDICTIONS_DIR_LOCAL =f"{DATA_FILE_DIR_LOCAL}"

# Local GCP Params Setting
ENV_GCP = "GCP"  # Google Cloud Platform
BUCKET_NAME = "adsp-capstone-enique-data"
DATA_FILE_DIR_GCP = "data"
BITCOIN_DATA_FILE_PATH_GCP = f"{DATA_FILE_DIR_GCP}/Bitcoin/bitcoin_data.csv"
SENTIMENT_DATA_FILE_PATH_GCP = f"{DATA_FILE_DIR_GCP}/Reddit/train_test_lstm_reddit_data.csv"
BEST_MODEL_DIR_GCP=f"{DATA_FILE_DIR_GCP}"
PREDICTIONS_DIR_GCP =f"{DATA_FILE_DIR_GCP}"

# Set Project Execution Environment
EXE_ENVIROMENT = ENV_LOCAL  # Set this to ENV_LOCAL or ENV_GCP based on your environment
if EXE_ENVIROMENT==ENV_LOCAL:
    MODEL_DIR=BEST_MODEL_DIR_LOCAL
    DATA_FILE_DIR = DATA_FILE_DIR_LOCAL
    PREDICTIONS_DIR=f"{DATA_FILE_DIR_LOCAL}"
if EXE_ENVIROMENT==ENV_GCP:
    MODEL_DIR=BEST_MODEL_DIR_GCP
    DATA_FILE_DIR = DATA_FILE_DIR_GCP
    PREDICTIONS_DIR=f"{DATA_FILE_DIR_GCP}"


# Define global constants
FORECAST_FREQUENCY = 'H'  # 'D' for daily
FORECAST_HOURS = 120

DEFAULT_REGRESSION_METRICS = ['mape', 'rmse', 'mae']
DEFAULT_CLASSIFICATION_METRICS = ['accuracy', 'precision', 'recall', 'f1_score']
DEFAULT_DECIMALS = 4

# ==============================DONT CHANGE=============================================
VAR_DATE = 'ds'
VAR_TARGET = 'y'
VAR_FORECAST = 'yhat'




#%%
