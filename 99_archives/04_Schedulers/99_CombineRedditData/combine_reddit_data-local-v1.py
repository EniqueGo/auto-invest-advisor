import os
import pandas as pd
from datetime import datetime

# Set the base bucket path
BUCKET_NAME = "/Users/estherx/UngoogleDrive/stonkgo_data/reddit/current/"

# Define specific paths based on the base bucket path
ORIGIN_PATH = os.path.join(BUCKET_NAME, "real-time-reddit-sub")
DESTI_PATH = os.path.join(BUCKET_NAME, "integrated-reddit-data")
BACK_PATH = os.path.join(BUCKET_NAME, "real-time-reddit-sub-bk")

# Number of files to select
SELECTED_FILE_NUM = 4

# Generate a date format file name
date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
integr_folder_name = f"integr_reddit_sub_{date_str}"
new_folder_path = os.path.join(ORIGIN_PATH, integr_folder_name)

# Ensure the new and destination directories exist
os.makedirs(new_folder_path, exist_ok=True)
os.makedirs(DESTI_PATH, exist_ok=True)  # Make sure destination path exists
os.makedirs(BACK_PATH, exist_ok=True)

# List all CSV files in the directory
csv_files = [f for f in os.listdir(ORIGIN_PATH) if f.endswith('.csv')]

# Check if there are enough files to process
if len(csv_files) > 0:
    # Sort files by creation time and select the oldest
    csv_files.sort(key=lambda x: os.path.getctime(os.path.join(ORIGIN_PATH, x)))
    oldest_files = csv_files[:SELECTED_FILE_NUM]

    # Move the oldest files to the new folder
    for file in oldest_files:
        os.rename(os.path.join(ORIGIN_PATH, file), os.path.join(new_folder_path, file))

    # Load the moved CSV files from the new directory
    dfs = [pd.read_csv(os.path.join(new_folder_path, file)) for file in oldest_files]

    # Combine the dataframes and remove duplicates based on 'id', keeping the latest 'created'
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.sort_values(by='created', ascending=False).drop_duplicates(subset='id', keep='first')

    # Save the combined dataframe to a new CSV file in the destination directory
    output_file_path = os.path.join(DESTI_PATH, f"integr_reddit_sub_{date_str}.csv")
    combined_df.to_csv(output_file_path, index=False)

    # Move the processed folder to the backup location
    new_backup_folder_path = os.path.join(BACK_PATH, integr_folder_name)
    os.rename(new_folder_path, new_backup_folder_path)
else:
    print(f"There are no files in {ORIGIN_PATH}")

#%%
