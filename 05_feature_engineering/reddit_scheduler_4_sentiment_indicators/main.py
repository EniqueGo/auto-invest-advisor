import base64
import functions_framework

import pandas as pd
import numpy as np
import os, sys
from datetime import datetime


from google.cloud import storage
import io
import joblib


class GCPSManager:
    def __init__(self, bucket_name):
        self.bucket_name = bucket_name
        self.gcs_client = storage.Client()
        self.bucket = self.gcs_client.bucket(bucket_name)

    def delete_files(self, file_dir, file_extension=None):
        """Delete all blobs in a given folder."""
        blobs = list(self.bucket.list_blobs(prefix=file_dir))
        for blob in blobs:
            if file_extension is None or blob.name.endswith(file_extension):
                try:
                    blob.delete()
                    print(f"Deleted file: {blob.name}")
                except Exception as e:
                    print(f"Error deleting file {blob.name}: {e}")

    def upload_files_to_gcs(self, src_file_dir, dest_file_dir):
        """Uploads files from a local folder to a GCS bucket."""
        for root, _, files in os.walk(src_file_dir):
            for file in files:
                local_path = os.path.join(root, file)
                blob_name = os.path.join(dest_file_dir, os.path.relpath(local_path, src_file_dir))
                blob = self.bucket.blob(blob_name)
                blob.upload_from_filename(local_path)
                print(f"Uploaded {local_path} to gs://{self.bucket_name}/{blob_name}")

    def read_and_concat_csvs(self, file_dir, col_names):
        """Read all CSV files from the specified GCS directory and concatenate them into one DataFrame."""
        dfs = []
        blobs = self.bucket.list_blobs(prefix=file_dir)
        for blob in blobs:
            if blob.name.endswith('.csv'):
                content = blob.download_as_string()
                df = pd.read_csv(io.BytesIO(content), names=col_names)
                dfs.append(df)
        concatenated_df = pd.concat(dfs, axis=0, ignore_index=True)
        return concatenated_df

    def read_csv(self, file_path):
        """Read a specific CSV file from GCS and return a DataFrame."""
        blob = self.bucket.blob(file_path)
        content = blob.download_as_string()
        df = pd.read_csv(io.BytesIO(content))
        return df

    def save_to_csv(self, df, file_path):
        """Save DataFrame to a CSV file in the specified GCS directory."""
        try:
            if not df.empty:
                buffer = io.StringIO()
                df.to_csv(buffer, index=False)
                buffer.seek(0)
                blob = self.bucket.blob(file_path)
                blob.upload_from_string(buffer.getvalue(), content_type='text/csv')
                print(f"Data saved to gs://{self.bucket.name}/{file_path}")
            else:
                print("No data. Check input data.")
        except Exception as e:
            print(f"Failed to save data: {e}")

    def save_model_to_gcs(self, model, model_path):
        """Save a model to GCP."""
        local_model_path = 'model.joblib'
        joblib.dump(model, local_model_path)
        blob = self.bucket.blob(model_path)
        blob.upload_from_filename(local_model_path)
        print(f'Model saved to gs://{self.bucket_name}/{model_path}')

    def load_model_from_gcs(self, model_path, local_model_path='model.joblib'):
        """Load a model from GCP."""
        blob = self.bucket.blob(model_path)
        blob.download_to_filename(local_model_path)
        model = joblib.load(local_model_path)
        return model

def unixtime_to_datetime(unix_timestamp, datetime_format='%Y-%m-%d %H:%M:%S'):
    """Converts a single Unix timestamp to a formatted datetime string."""

    # Convert Unix timestamp to datetime object
    dt_object = datetime.utcfromtimestamp(unix_timestamp)

    # Format datetime object as "m-d-y HH:mm:ss"
    formatted_datetime = dt_object.strftime(datetime_format)

    return formatted_datetime

def load_reddit_senti_data(historical_reddit_file_path,current_reddit_file_path):

    historical_reddit_raw= pd.read_parquet(historical_reddit_file_path)

    current_reddit_raw= pd.read_parquet(current_reddit_file_path)



    # Drop unnecessary columns
    historical_reddit = historical_reddit_raw.drop(columns=['id', 'author','permalink',
                                                            'selftext','title','name',
                                                            'author_created_utc','text', 'clean_text',
                                                            'is_self','is_original_content']).copy()

    processed_df = current_reddit_raw.copy()

    # Convert 'Created' datetime from Unixtime to datetime
    processed_df['created_utc'] = processed_df['created'].apply(lambda x: unixtime_to_datetime(x))

    return  historical_reddit, processed_df


def load_baseline_model_results(output_baseline_file_path):
    output_baseline= pd.read_csv(output_baseline_file_path)
    output_baseline['ds'] = pd.to_datetime(output_baseline['ds'])

    if "Unnamed: 0" in output_baseline.columns:
        output_baseline.drop(columns=["Unnamed: 0"], inplace=True)

    return  output_baseline


def zip_and_explode(df, col_labels, col_scores, prefix):

    # Zip the labels and scores together
    df['zipped'] = df.apply(lambda row: list(zip(row[col_labels], row[col_scores])), axis=1)

    # Explode the zipped column
    df_exploded = df.explode('zipped')

    # Create separate columns for labels and scores
    df_exploded[f'{prefix}label'], df_exploded[f'{prefix}score'] = zip(*df_exploded['zipped'])

    return df_exploded

def pivot_df(df, index_col, label_col, score_col, prefix):

    # Create a pivot table
    pivot_table = df.pivot_table(index=index_col, columns=label_col, values=score_col, aggfunc='first').reset_index()

    # Flatten the MultiIndex columns and add prefix
    pivot_table.columns = [f"{prefix}{col}" if col != index_col else col for col in pivot_table.columns]

    return pivot_table

def time_weighted_avg(series, window=120, decay_rate=0.5):
    weights = np.array([decay_rate**((window-i)/window) for i in range(window)])
    weighted_sum = np.convolve(series, weights, mode='valid')
    return np.concatenate((np.full(window-1, np.nan), weighted_sum / weights.sum()))


def calculate_sentiment_score(df):

    required_columns = ['tw_avg_positive', 'tw_avg_negative', 'tw_avg_neutral']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"DataFrame is missing required column: {col}")

    # Calculate the sentiment score using the given formula
    df['sentiment_score'] = 0 * df['tw_avg_negative'] + 50 * df['tw_avg_neutral'] + 100 * df['tw_avg_positive']

    return df

# Triggered from a message on a Cloud Pub/Sub topic.
@functions_framework.cloud_event
def hello_pubsub(cloud_event):
    # Print out the data from Pub/Sub, to prove that it worked
    #print(base64.b64decode(cloud_event.data["message"]["data"]))

    reddit_dir = "data/Reddit"
    historical_reddit_file_path = f"{reddit_dir}/cleaned_sentiment_submissions_2019-01-01_to_2023-12-31.parquet"
    current_reddit_file_path = 'data/relevant_process/6-senti-submissions/senti_data.parquet'
    output_baseline_file_path = f"data/Bitcoin/output_baseline.csv"

    historical_reddit_file_url = "gs://adsp-capstone-enique-data/data/Reddit/cleaned_sentiment_submissions_2019-01-01_to_2023-12-31.parquet"
    current_reddit_file_url = "gs://adsp-capstone-enique-data/data/relevant_process/6-senti-submissions/senti_data.parquet"
    output_baseline_file_url = "gs://adsp-capstone-enique-data/data/Bitcoin/output_baseline.csv"


    # Constants
    HORIZON_DAYS = 5 # Forcast next 5 days
    HORIZON_HOURS = HORIZON_DAYS * 24

    historical_reddit, current_reddit = load_reddit_senti_data(historical_reddit_file_url,
                                                               current_reddit_file_url)

    output_baseline = load_baseline_model_results(output_baseline_file_url)

    processed_df = current_reddit.copy()
    # Step 1: Zip and explode emotion and sentiment columns
    emotion_exploded_df = zip_and_explode(processed_df, 'emotion_labels', 'emotion_scores', 'emotion_')
    sentiment_exploded_df = zip_and_explode(processed_df, 'sentiment_labels', 'sentiment_scores', 'sentiment_')

    # Step 2: Pivot the exploded DataFrames
    emotion_pivot_df = pivot_df(emotion_exploded_df, 'id', 'emotion_label', 'emotion_score', 'emotion_')
    sentiment_pivot_df = pivot_df(sentiment_exploded_df, 'id', 'sentiment_label', 'sentiment_score', 'sentiment_')

    # Step 3: Join the pivoted DataFrames back to the original DataFrame
    cols_to_keep = [col for col in processed_df.columns if col not in ['emotion_labels', 'emotion_scores', 'sentiment_labels', 'sentiment_scores', 'zipped']]
    original_df = processed_df[cols_to_keep]

    exploded_df = original_df.merge(emotion_pivot_df, on='id', how='left').merge(sentiment_pivot_df, on='id', how='left')

    current_reddit = exploded_df[historical_reddit.columns].copy()

    submissions_df = pd.concat([historical_reddit, current_reddit], axis = 0)

    # Convert 'created_utc' to datetime and set as index
    submissions_df.rename(columns={'created_utc': 'ds'}, inplace=True)
    submissions_df['ds'] = pd.to_datetime(submissions_df['ds'])
    submissions_df = submissions_df.sort_values(by=['ds']).set_index('ds')

    # Filter submissions with an upvote ratio greater than 0.7
    submissions_df_filtered = submissions_df[submissions_df['upvote_ratio'] > 0.7].copy()

    # Resample to hourly frequency
    sentiment_df_hourly = submissions_df_filtered[['sentiment_positive',
                                                   'sentiment_negative',
                                                   'sentiment_neutral']].resample('H').mean()

    sentiment_df_hourly['post_count_by_hour'] = submissions_df_filtered.resample('H').size()


    # Generate the full date range
    full_date_range = pd.date_range(start=output_baseline['ds'].min(), end=output_baseline['ds'].max(), freq='H')

    # Reindex the DataFrame to include all dates in the full date range
    sentiment_df_hourly_reindexed = sentiment_df_hourly.reindex(full_date_range)

    sentiment_df_hourly_interpolated = sentiment_df_hourly_reindexed.interpolate(method='time')


    sentiment_df = sentiment_df_hourly_interpolated.copy()
    sentiment_df['sentiment_positive_momentum'] = sentiment_df['sentiment_positive'].diff()
    sentiment_df['sentiment_negative_momentum'] = sentiment_df['sentiment_negative'].diff()
    sentiment_df['sentiment_neutral_momentum'] = sentiment_df['sentiment_neutral'].diff()

    # Calculate sentiment volatility (using a 3-day rolling window)
    sentiment_df['sentiment_positive_volatility'] = sentiment_df['sentiment_positive'].rolling(window=24*3).std()
    sentiment_df['sentiment_negative_volatility'] = sentiment_df['sentiment_negative'].rolling(window=24*3).std()
    sentiment_df['sentiment_neutral_volatility'] = sentiment_df['sentiment_neutral'].rolling(window=24*3).std()


    # Calculate time-weighted average sentiment scores
    sentiment_df['tw_avg_positive'] = time_weighted_avg(sentiment_df['sentiment_positive'], window=HORIZON_HOURS, decay_rate=0.5)
    sentiment_df['tw_avg_negative'] = time_weighted_avg(sentiment_df['sentiment_negative'], window=HORIZON_HOURS, decay_rate=0.5)
    sentiment_df['tw_avg_neutral'] = time_weighted_avg(sentiment_df['sentiment_neutral'], window=HORIZON_HOURS, decay_rate=0.5)

    for lag in range(1, 8):
        sentiment_df[f'sentiment_positive_lag_{lag}hr'] = sentiment_df['sentiment_positive'].shift(lag)
        sentiment_df[f'sentiment_negative_lag_{lag}hr'] = sentiment_df['sentiment_negative'].shift(lag)
        sentiment_df[f'sentiment_neutral_lag_{lag}hr'] = sentiment_df['sentiment_neutral'].shift(lag)


    sentiment_df.fillna(method='ffill', inplace=True)
    sentiment_df.fillna(method='bfill', inplace=True)

    sentiment_df.reset_index(inplace = True)
    sentiment_df.rename(columns={'index':'ds'} , inplace=True)

    #====Save: train_test_lstm_reddit_data.csv=======
    sentiment_df.to_csv("gs://adsp-capstone-enique-data/data/Reddit/train_test_lstm_reddit_data.csv", index=False)
    print("train_test_lstm_reddit_data.csv saved")


    #=====Save: train_test_lstm_reddit_data.csv====
    sentiment_over_time = calculate_sentiment_score(sentiment_df)
    sentiment_over_time['sentiment_score'] = sentiment_over_time['sentiment_score'].astype(int)
    sentiment_over_time.to_csv("gs://adsp-capstone-enique-data/results/sentiment_over_time.csv", index=False)
    print("sentiment_over_time.csv saved")

