from google.cloud import storage
import pandas as pd
from datetime import datetime
import functions_framework

BUCKET_NAME = 'stonkgo2-spark-bucket'
ORIGIN_PATH = "real-time-reddit-sub/"
DESTI_PATH = "combined-reddit-data/"
BACKUP_PATH = "real-time-reddit-sub-bk/"

SELECTED_FILE_NUM = 10

def combine_reddit_data():
    date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    processing_folder = f"{ORIGIN_PATH}combined_reddit_sub_{date_str}/"

    client = storage.Client()
    bucket = client.get_bucket(BUCKET_NAME)

    # List all CSV files in the ORIGIN_PATH
    blobs = list(bucket.list_blobs(prefix=ORIGIN_PATH))
    csv_files = [blob.name for blob in blobs if blob.name.endswith('.csv') and ORIGIN_PATH in blob.name]

    if len(csv_files) > 0:
        # Sort files by name
        csv_files.sort()
        oldest_files = csv_files[:SELECTED_FILE_NUM]

        # Move files to the processing folder within the same ORIGIN_PATH
        for file in oldest_files:
            source_blob = bucket.blob(file)
            new_blob = bucket.blob(f"{processing_folder}{file.split('/')[-1]}")
            bucket.copy_blob(source_blob, bucket, new_blob.name)
            source_blob.delete()

        # Read CSV files directly into DataFrame from the processing folder
        processing_files = [f"{processing_folder}{file.split('/')[-1]}" for file in oldest_files]
        dfs = [pd.read_csv(f"gs://{BUCKET_NAME}/{file}") for file in processing_files]
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df = combined_df.sort_values(by='created', ascending=False).drop_duplicates(subset='id', keep='first')

        # Save combined DataFrame back to GCS
        desti_blob_path = f"{DESTI_PATH}combined_reddit_sub_{date_str}.csv"
        combined_df.to_csv(f"gs://{BUCKET_NAME}/{desti_blob_path}", index=False)

        # Move the entire processing folder to the backup location
        for file in processing_files:
            source_blob = bucket.blob(file)
            new_blob = bucket.blob(file.replace(ORIGIN_PATH, BACKUP_PATH))
            bucket.copy_blob(source_blob, bucket, new_blob.name)
            source_blob.delete()

        print(f"Combined data saved to: gs://{BUCKET_NAME}/{desti_blob_path}")
    else:
        print(f"There are no files in gs://{BUCKET_NAME}/{ORIGIN_PATH}")


@functions_framework.cloud_event
def hello_pubsub(cloud_event):
    combine_reddit_data()