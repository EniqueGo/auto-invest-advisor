import base64
import functions_framework

from google.cloud import storage
import pandas as pd
import numpy as np
import io
import requests
import zipfile
import os
from datetime import timedelta, date, datetime,timezone
import time
import shutil


bucket_name="adsp-capstone-enique-data"
btc_dir = "data/Bitcoin"

btc_temp_dir = f"{btc_dir}/temp"

historical_btc_file_path = f"{btc_dir}/historical_bitcoin_data.csv"

loc_extract_to = "gs://adsp-capstone-enique-data/data/Bitcoin/extract_temp/"


class GCPSManager:
    def __init__(self, bucket_name):
        self.bucket_name = bucket_name
        self.gcs_client = storage.Client()
        self.bucket = self.gcs_client.bucket(bucket_name)

    #     def list_blobs(self, folder_name):
    #         """Return a pandas DataFrame with names and sizes of all blobs in a given folder."""
    #         blobs = list(self.bucket.list_blobs(prefix=folder_name))
    #         blob_name = [blob.name for blob in blobs]
    #         blob_size = [blob.size for blob in blobs]
    #         blob_df = pd.DataFrame(list(zip(blob_name, blob_size)), columns=['Name', 'Size'])
    #         return blob_df


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


    # def upload_to_bucket(self, source_file_name, destination_blob_name):
    #     blob = self.bucket.blob(destination_blob_name)
    #     blob.upload_from_filename(source_file_name)
    #     print(f"File {source_file_name} uploaded to {destination_blob_name}.")

    def upload_files_to_gcs(self, src_file_dir, dest_file_dir):

        """Uploads files from a local folder to a GCS bucket."""
        for root, _, files in os.walk(src_file_dir):
            for file in files:
                local_path = os.path.join(root, file)
                blob_name = os.path.join(dest_file_dir, os.path.relpath(local_path, src_file_dir))
                blob = self.bucket.blob(blob_name)
                blob.upload_from_filename(local_path)
                print(f"Uploaded {local_path} to gs://{self.bucket_name}/{blob_name}")


    # def read_csvs(self, file_dirs: list):
    #     """Read CSV files from the specified GCS directories and return a list of DataFrames."""
    #     dfs = []
    #     for prefix in files_dir:
    #         print(prefix)
    #         blobs = list(self.bucket.list_blobs(prefix=f"{prefix}/"))
    #         print(blobs)
    #         for blob in blobs:
    #             if blob.name.endswith('.csv'):
    #                 content = blob.download_as_string()
    #                 df = pd.read_csv(io.BytesIO(content))
    #                 dfs.append(df)
    #     return dfs


    def read_and_concat_csvs(self, file_dir, col_names):
        """Read all CSV files from the specified GCS directory and concatenate them into one DataFrame."""
        dfs = []

        # Iterate over blobs in the specified directory
        blobs = self.bucket.list_blobs(prefix=file_dir)
        for blob in blobs:
            if blob.name.endswith('.csv'):
                content = blob.download_as_string()
                # Specify column names since the CSV files don't have headers
                df = pd.read_csv(io.BytesIO(content), names=col_names)

                dfs.append(df)

        # Concatenate all dataframes
        concatenated_df = pd.concat(dfs, axis=0, ignore_index=True)

        return concatenated_df


    def read_csv(self, file_path):
        """Read a specific CSV file from GCS and return a DataFrame."""
        blob = self.bucket.blob(file_path)
        print(blob)
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


#=========Functions=======
def download_and_extract_zip(url, extract_to='.'):
    response = requests.get(url)
    if response.status_code == 200:
        with zipfile.ZipFile(io.BytesIO(response.content)) as thezip:
            thezip.extractall(path=extract_to)
        print(f"Files extracted to {extract_to}")
    else:
        print(f"Failed to download file: status code {response.status_code}")


def unixtimes_to_datetimes(unix_timestamp_series, datetime_format='%Y-%m-%d %H:%M:%S'):
    """Converts a Series of Unix timestamps to formatted datetime strings."""
    datetime_series = pd.to_datetime(np.floor(unix_timestamp_series / 1000), unit='s')
    return datetime_series

def delete_local_files(file_dir, file_extension=None, is_delete_dir=False):
    """
    Delete files from a local directory. Optionally, delete files with a specific extension.
    If is_delete_dir is True, also delete the directory itself.
    """
    for filename in os.listdir(file_dir):
        file_path = os.path.join(file_dir, filename)

        try:
            if os.path.isfile(file_path):
                print(file_path)
                if file_extension is None or filename.endswith(file_extension):
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")

        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

    if is_delete_dir and os.path.isdir(file_dir):
        try:
            shutil.rmtree(file_dir)
            print(f"Deleted folder: {file_dir}")
        except Exception as e:
            print(f"Error deleting directory {file_dir}: {e}")


# Triggered from a message on a Cloud Pub/Sub topic.
@functions_framework.cloud_event
def hello_pubsub(cloud_event):
    # Print out the data from Pub/Sub, to prove that it worked
    #print(base64.b64decode(cloud_event.data["message"]["data"]))

    gcps = GCPSManager(bucket_name="adsp-capstone-enique-data")
    historical_btc_df = gcps.read_csv(historical_btc_file_path)

    # Convert 'ds' to datetime and set as index
    historical_btc_df['ds'] = pd.to_datetime(historical_btc_df['ds'])

    # historical_biggest_datetime = historical_btc_df['ds'].max()
    historical_biggest_datetime = max(historical_btc_df['ds'])

    # Calculate start time of update
    start_update_datetime = historical_biggest_datetime + timedelta(hours=1)

    # Calculate yesterday date
    yesterday = (datetime.now()-timedelta(days=1)).astimezone(timezone.utc).date()

    print("calculated yesterday: ", yesterday)

    # Calculate data points(time points) required to download
    data_points_of_download = (yesterday-historical_biggest_datetime.date()).days*24

    dates = pd.date_range(start=start_update_datetime, end = yesterday, freq='D')
    print(dates)
    if dates.empty:
        print("No Data needs to be updated")
        return

    for date in dates:
        url = f"https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1h/BTCUSDT-1h-{date:%Y-%m-%d}.zip"
        # print(url)

        download_and_extract_zip(url, loc_extract_to)
        gcps.upload_files_to_gcs(loc_extract_to, btc_temp_dir)

    # ******** 4. Concatenate CSV files
    col_names = ['open_price','y_upper','y_lower','y',
                 'volume','ds','quote_asset_volume',
                 'num_of_trades','taker_buy_base_asset_volume',
                 'taker_buy_quote_asset_volume','unused_field_ignore']

    combined_df_binance = gcps.read_and_concat_csvs(btc_temp_dir, col_names)

    # Check if download data points correctly
    assert data_points_of_download == combined_df_binance.shape[0], f"Data point required to download is {data_points_of_download}"

    # ******** 5. Drop unused_field_ignore column
    combined_df_binance.drop(columns=["unused_field_ignore"], inplace=True)

    # ******** 6. Convert 'ds' from Unixtime to datetime
    combined_df_binance_clean = combined_df_binance.copy()

    combined_df_binance_clean['ds'] = unixtimes_to_datetimes(combined_df_binance_clean['ds'])

    # ******** 7. Sort the Data by 'ds'
    # It's crucial to sort the data by `ds` to ensure that any checks for gaps are done in chronological order.
    combined_df_binance_clean.sort_values(by=['ds'], inplace=True)

    # ******** 8. Standardize Timestamps
    combined_df_binance_clean.set_index("ds", inplace= True)
    combined_df_binance_clean.index = combined_df_binance_clean.index.floor('H')

    # ******** 9. Check if has Date Time Gap
    # Recalculate the time differences to see if there are any remaining gaps
    combined_df_binance_clean['time_diff'] = combined_df_binance_clean.index.to_series().diff()
    remaining_gaps = combined_df_binance_clean[combined_df_binance_clean['time_diff'] > pd.Timedelta(hours=1)]

    assert remaining_gaps.empty, f"New data has {remaining_gaps} gaps."
    combined_df_binance_clean.drop(columns=['time_diff'], inplace=True)
    combined_df_binance_clean.reset_index(inplace=True)

    # ******** 10. Combine the Historical and New DataFrames
    combined_df = pd.concat([historical_btc_df, combined_df_binance_clean], axis=0)

    # ******** 11. Save New Historical Dataset

    gcps.save_to_csv(combined_df, historical_btc_file_path)

    # ******** 12. Delete Temp CSVs
    gcps.delete_files(btc_temp_dir)

    delete_local_files(loc_extract_to,'.csv', is_delete_dir=True)
