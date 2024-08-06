# utils/data_reader.py

"""
@author: Esther Xu
Code tested on:
    - os: Debian 11
    - Python: 3.10.14; 3.11.2
    - Numpy: 1.25.2
"""
import pandas as pd
from datetime import datetime, timedelta
from utils.GCPStorage_manager import GCPSManager
import google.auth.exceptions
from config import (EXE_ENVIROMENT,VAR_DATE, BITCOIN_DATA_FILE_PATH_LOCAL,
                    ENV_LOCAL,BITCOIN_DATA_FILE_PATH_GCP, SENTIMENT_DATA_FILE_PATH_LOCAL,
                    SENTIMENT_DATA_FILE_PATH_GCP,ENV_GCP,BUCKET_NAME,FORECAST_HOURS)

def get_btc_df_with_indicators(exe_env=EXE_ENVIROMENT):
    """
    Function to read the Bitcoin dataframe with indicators.
    """
    file_path = BITCOIN_DATA_FILE_PATH_LOCAL if exe_env == ENV_LOCAL else BITCOIN_DATA_FILE_PATH_GCP
    print(f"Bitcoin file path:{file_path}")
    df = __df_reader(file_path, exe_env)
    # print(df.info())
    # Set datetime
    df[VAR_DATE] = pd.to_datetime(df[VAR_DATE])

    return df

def get_sentiment_df_with_indicators(exe_env=EXE_ENVIROMENT):
    """
    Function to read the Sentiment dataframe with indicators.
    """
    file_path = SENTIMENT_DATA_FILE_PATH_LOCAL if exe_env == ENV_LOCAL else SENTIMENT_DATA_FILE_PATH_GCP
    print(f"Sentiment file path:{file_path}")
    df = __df_reader(file_path, exe_env)
    # Set datetime
    df[VAR_DATE] = pd.to_datetime(df[VAR_DATE])

    return df

def generate_model_data(n_sample=None):
    """
    Function to generate model data by merging Bitcoin and Sentiment hourly data.
    Ensures both datasets have the same date range and checks.
    n_sample: Get the newest datas
    """

    # Load and copy data
    btc_df = get_btc_df_with_indicators().copy()
    # print(btc_df.info())
    sentiment_df = get_sentiment_df_with_indicators().copy()

    # Check if btc_df's max date is yesterday
    max_btc_date = pd.to_datetime(btc_df[VAR_DATE]).max().date()

    max_sentiment_date = pd.to_datetime(sentiment_df[VAR_DATE]).max().date()
    # Check if btc_df's max date equals to  sentiment_df's max date
    if max_btc_date != max_sentiment_date:
        raise ValueError(f"The maximum date in btc_df is {max_btc_date}, which is not sentiment_df's date ({max_sentiment_date}).")

    # Ensure both DataFrames have the same length
    if len(btc_df) != len(sentiment_df):
        raise ValueError(f"The length of btc_df ({len(btc_df)}) does not match the length of sentiment_df ({len(sentiment_df)}).")

    # Merge Datasets: Bitcoin and Sentiment hourly data
    merged_df = btc_df.merge(sentiment_df, on=VAR_DATE, how='left')

    if n_sample is not None: # Always get newest data

        #calculate cuttof time
        sample_datetime_from = merged_df[VAR_DATE].max()-timedelta(hours=(n_sample-1))

        sample_datetime_from_nearest_hr = pd.to_datetime(sample_datetime_from).floor('H')

        # Get the data after the cutoff_datetime (including)
        filtered_data = merged_df[merged_df[VAR_DATE] >= sample_datetime_from_nearest_hr]
        return filtered_data

    return merged_df


def __df_reader(file_path, exe_env):
    """
    Private function to read CSV files from local or GCP storage.
    """
    if exe_env == ENV_GCP:  # GCP
        try:
            gcps = GCPSManager(bucket_name=BUCKET_NAME)
            return gcps.read_csv(file_path)
        except google.auth.exceptions.DefaultCredentialsError as e:
            print("Google Cloud credentials not found. Please set up Application Default Credentials.")
            print(e)
            return None
    elif exe_env == ENV_LOCAL:  # Local
        try:
            return pd.read_csv(file_path)
        except FileNotFoundError as e:
            print(f"File not found: {file_path}. Please ensure the file exists.")
            print(e)
            return None

def trim_dataset(df, cutoff_datetime, train_days=180, forecast_hours=FORECAST_HOURS):
    """
    Trims the dataset to include data within the specified number of days
    from the given current time, and add forecast hours data points.

    Parameters:
    - df: pd.DataFrame - The input DataFrame containing the data.
    - cutoff_datetime: str or pd.Timestamp - The cutoff datetime for trimming the data.The end time in Training Data
    - train_days: int or None - The number of days to include before the cutoff datetime.
    - forecast_hours: int or None - The number of hours to include after the cutoff datetime.

    Returns:
    - filtered_data: pd.DataFrame - The trimmed DataFrame.
    """
    current_datetime_nearest_hr = pd.to_datetime(cutoff_datetime).floor('H')

    df[VAR_DATE] = pd.to_datetime(df[VAR_DATE])


    cutoff_date = current_datetime_nearest_hr - pd.DateOffset(days=train_days)

    filtered_data = df[(df[VAR_DATE] > cutoff_date) &
                           (df[VAR_DATE] <= current_datetime_nearest_hr + timedelta(hours=forecast_hours))]

    train_dataset, test_data_set =split_data_by_cutoff_point(filtered_data)

    return train_dataset, test_data_set


def split_data_by_cutoff_point(df, test_rows =FORECAST_HOURS): # Forcast next 5 days * 24 hrs = 120 hrs (rows)
    data = df.copy()

    total_rows = len(data)
    split_point = total_rows -  test_rows

    train = data[:split_point]
    test = data[split_point:]
    return train, test

# Example:
# if __name__ == "__main__":
#     # aa= pd.read_csv("/Users/estherx/Documents/EstherxScripts/python/stonkgo/stonkgo_v2.2.0_GCP/0_data/bitcoin_data.csv")
#     # print(aa)
#     btc_df = get_btc_df_with_indicators()
#     # cutoff_datetime = '2023-12-31 23:00:00'
#     # trimmed_df = trim_dataset(btc_df, cutoff_datetime, train_days=30, forecast_hours=120)
#     # print(trimmed_df[VAR_DATE].min())
#     # print(trimmed_df[VAR_DATE].max())
#     # print(trimmed_df.shape[0])
#     merge_df = generate_model_data(n_sample=10)
#     # print(f"btc data:{btc_df.shape[0]}")
#     print(f"merge data:{merge_df.shape[0]}")
#     print(f"merge data start:{merge_df[VAR_DATE].min()}")
#     print(f"merge data end:{merge_df[VAR_DATE].max()}")
#     print("----------")
#     pass



#%%

#%%
