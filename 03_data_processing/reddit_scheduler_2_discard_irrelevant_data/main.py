from google.cloud import storage
import pandas as pd
import io
from datetime import datetime
import re
#from ktrain import text as ktext
import hashlib
import numpy as np
import functions_framework
# =========================== 0.Constants and Initialization

BUCKET_NAME = "adsp-capstone-enique-data"
BUCKET_PATH = f"gs://{BUCKET_NAME}"
SOURCE_PATH = "real-time-reddit-sub"
BACKUP_PATH = "real-time-reddit-sub-bk"

PROCESSING_PATH = "data/relevant_process"
COMBINED_PATH = f"{PROCESSING_PATH}/1-combined-files"
HISTORICAL_IR_RE_PATH = f"{BUCKET_PATH}/{PROCESSING_PATH}/2-historical-irre-re-files/historical_ir_relevant_data.parquet"

date_str = datetime.now().strftime('%Y%m%d%H%M%S')
new_combined_file_path = f'{COMBINED_PATH}/combined_{date_str}.csv'

client = storage.Client()
bucket = client.get_bucket(BUCKET_NAME)

btc_related_keywords = [
    "Bitcoin", "BTC", "Crypto", "Cryptocurrency", "Blockchain",
    "Satoshi", "Mining", "Wallet", "Ethereum", "Altcoin",
    "Investment", "Investing", "Trading", "Markets", "Portfolio",
    "Stocks", "ROI", "Bonds", "Shares", "Finance",
    "price", "news", "mining", "wallet",
    "trading", "invest", "put", "call", "buy", "sell", "exchange", "transaction",
    "value", "market", "cap", "volume", "supply", "demand", "price", "high", "low","ledger",
    "trend", "reversal", "momentum", "technical", "indicat", "indicator", "indicate",
    "flow", "coin", "decentralize", "DeFi", "wallet", "fiat", "cold",
    "block", "reward", "consensus", "binance", "chain", "gas",
    "rate", "borrow", "yield"
]


# ========= Helper Functions
def read_csv_from_blob(blob):
    content = blob.download_as_string()
    return pd.read_csv(io.BytesIO(content))

def save_data_to_csv(data, file_path):
    if not data.empty:
        buffer = data.to_csv(index=False)
        blob = bucket.blob(f"{file_path}")
        blob.upload_from_string(buffer, content_type='text/csv')
        print(f"Data saved to gs://{bucket.name}/{file_path}")

def delete_old_combined_files(current_file_path):

    blobs = bucket.list_blobs(prefix=COMBINED_PATH)
    for blob in blobs:
        # Check if the blob is not the current file and not a directory marker
        if blob.name != current_file_path and not blob.name.endswith('/'):
            print(f"Deleting old file: gs://{bucket.name}/{blob.name}")
            blob.delete()
        else:
            print(f"Skipping deletion of: gs://{bucket.name}/{blob.name}")


# ========= Main Function
def main_scheduler():

    # Step 1: Load Unprocessed Data (save combined file)
    unprocessed_data = load_unprocessed_data()

    print(f"unprocessed_data={unprocessed_data.shape[0]}")

    # Read historical irrelevent data
    historical_ir_re_df = pd.read_parquet(HISTORICAL_IR_RE_PATH)

    # Step 2: Filter Unprocessed Data Based on Known Irrelevant and Relevant Records
    filtered_data = filter_bitcoin_related(unprocessed_data, historical_ir_re_df)
    print(f"filtered_data={filtered_data.shape[0]}")

    # Steps 3: Data Processing: Concatenate(save the file if failed, delete combined file after saving)
    processed_data = data_processing(filtered_data)

    # Steps 4. Discard Irrelevent Data
    discard_data = discard_irrelevant(processed_data, historical_ir_re_df)

    # Step 5: Delete combined data
    if not discard_data.empty:
        delete_old_combined_files(f'{COMBINED_PATH}/0.csv') # Delete all combined files in COMBINED_PATH

    #return discard_data

# =========================== 1. Load Unprocessed Data

def load_unprocessed_data():
    dfs = []

    # Process files from both combined and unprocessed paths
    for prefix in [COMBINED_PATH, SOURCE_PATH]:
        blobs = list(bucket.list_blobs(prefix=f"{prefix}/"))
        for blob in blobs:
            if blob.name.endswith('.csv'):
                df = read_csv_from_blob(blob)
                if 'fetch_datetime' not in df.columns:
                    try:
                        file_datetime = blob.name.split('_')[-1].rstrip('.csv')
                        df['fetch_datetime'] = pd.to_datetime(file_datetime, format='%Y%m%d%H%M%S')
                    except ValueError as e:
                        print(f"Error parsing datetime from file name: {blob.name} -> {e}")
                        #df['fetch_datetime'] = pd.NaT
                        df['fetch_datetime'] = datetime.now()

                if 'self_text_md5' not in df.columns:
                    df['self_text_md5'] = df['body'].apply(
                        lambda x: hashlib.md5(x.encode('utf-8')).hexdigest() if isinstance(x, str) else None
                    )

                dfs.append(df)

                # Check if the blob should be moved based on its path
                if prefix == SOURCE_PATH:
                    new_blob_path = f"{BACKUP_PATH}/{blob.name.split('/')[-1]}"
                    bucket.rename_blob(blob, new_blob_path)
                    print(f"Moved {blob.name} to {new_blob_path}")

    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df['fetch_datetime'] = pd.to_datetime(combined_df['fetch_datetime'], errors='coerce')
        combined_df.sort_values(by='fetch_datetime', ascending=True, inplace=True)
        combined_df.drop_duplicates(subset='id', keep='last', inplace=True)

        # Save combined data and delete other combined files
        save_data_to_csv(combined_df, new_combined_file_path)

        # Delete old combined files except the newly saved one
        delete_old_combined_files(new_combined_file_path)

        return combined_df

    return pd.DataFrame()  # Return an empty DataFrame if no files were processed

# =========================== 2. Filter New Data, Removing irrelevant data (already been processed in f"{HISTORICAL_IRRE_RE_PATH}/historical_ir_relevant_data.parquet") if 'id' and 'self_text_md5' hasn't been changed) and also update relevant data if have


def filter_bitcoin_related(data, historical_ir_re_df):

    # Initialize 'is_bitcoin_related' with NaN
    data['is_bitcoin_related'] = np.nan


    # Merge the current data with the historical irrelevant data based on 'id' and 'self_text_md5'
    merged_data = data.merge(historical_ir_re_df[['id', 'self_text_md5', 'is_irrelevant']],
                             on=['id', 'self_text_md5'],
                             how='left')


    # Update 'is_bitcoin_related' only where 'is_irrelevant' is not NaN:
    data['is_bitcoin_related'] = np.where(pd.notna(merged_data['is_irrelevant']),
                                          merged_data['is_irrelevant'].astype('boolean'),
                                          data['is_bitcoin_related'])


    # Filter to keep only rows where 'is_bitcoin_related' is True and None
    filtered_data = data[data['is_bitcoin_related'] != False]


    return filtered_data


# =========================== 3. Data Processing: Concatenate `title` and `selftext` of submissions to Combine these two fileds into a single text field to gauge the overall sentiment of the post

def clean_text(text):
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\t+', ' ', text)
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9 @ # . , : - _ % $ ° | ?]', ' ', text)
    return text.strip()


def data_processing(data):

    data = data.copy()

    # Fill 'NaN' values in 'body' column with an empty string
    data['body'] = data['body'].fillna("")

    # Combine 'title' and 'body' into a new 'text' column
    data['text'] = data['title'] + " " + data['body']

    # Apply 'clean_text' function to 'text' column to clean up the text
    data['clean_text'] = data['text'].apply(clean_text)

    return data


# =========================== 4. Discard irrelevant Bitcoin-related Data

def classify_text(text):
    """
    try:
        zsl = ktext.ZeroShotClassifier()
        labels = ["bitcoin", "cryptocurrency", "investment", "crypto", "btc"]
        prediction = zsl.predict(text, labels=labels, batch_size=64)
        # print(prediction)
        return any(prob > 0.5 for prob in prediction)
    except Exception as e:
        print(f"Error in classify_text: {e}")
        return False
    """
    for keyword in btc_related_keywords:
        if keyword.lower() in text.lower():
            return True
    return False

def discard_irrelevant(data, historical_irre_re_df):
    if 'is_bitcoin_related' not in data.columns:
        raise ValueError("Column 'is_bitcoin_related' is missing from the data.")

    # Only apply 'classify_text' where 'is_bitcoin_related' is NaN
    nan_mask = data['is_bitcoin_related'].isna()
    # nan_mask = data[data['is_bitcoin_related']!=True]

    # print(f"None data =={data.loc[nan_mask, 'clean_text'].shape[0]}")

    if nan_mask.any():
        print(data.loc[nan_mask, 'clean_text'])
        # Only compute classify_text for NaN entries in 'is_bitcoin_related'
        data.loc[nan_mask, 'is_bitcoin_related'] = data.loc[nan_mask, 'clean_text'].apply(classify_text)

    # Ensure fetch_datetime is a Timestamp across all data
    data['fetch_datetime'] = pd.to_datetime(data['fetch_datetime'], errors='coerce')
    historical_irre_re_df['fetch_datetime'] = pd.to_datetime(historical_irre_re_df['fetch_datetime'], errors='coerce')

    # Combine current data with historical data
    dfs = [historical_irre_re_df, data[['id', 'self_text_md5', 'is_bitcoin_related', 'fetch_datetime']].rename(
        columns={"is_bitcoin_related": "is_irrelevant"})]
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.sort_values(by='fetch_datetime', ascending=True, inplace=True)
    combined_df.drop_duplicates(subset='id', keep='last', inplace=True)
    print(f"combine df ={combined_df.shape[0]}")


    # Saving the combined data
    buffer = combined_df.to_parquet(HISTORICAL_IR_RE_PATH)
    # print(f"Saved new data to {HISTORICAL_IR_RE_PATH}")

    # Filter to find entries that are related to bitcoin
    bitcoin_related = data[data['is_bitcoin_related']]

    save_data_to_csv(bitcoin_related, f"{PROCESSING_PATH}/3-relevant-submissions/relevant_submissions_{date_str}.csv")

    return bitcoin_related

@functions_framework.cloud_event
def hello_pubsub(cloud_event):
    main_scheduler()
