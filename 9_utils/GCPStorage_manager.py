# utils/GCPStroage_manager.py
"""
@author: Esther Xu
Code tested on:
    - os: Debian 11
    - Python: 3.10.14; 3.11.2
    - Numpy: 1.25.2
    - Sklearn: 1.3.0
    - Tensorflow: 2.13.0; 2.14.0
    - Keras: 2.8; 2.14.0
"""
from google.cloud import storage
import pandas as pd
import io
import os
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
