import base64
import functions_framework

# Triggered from a message on a Cloud Pub/Sub topic.
@functions_framework.cloud_event
def hello_pubsub(cloud_event):
    # Print out the data from Pub/Sub, to prove that it worked
    # print(base64.b64decode(cloud_event.data["message"]["data"]))
    main_scheduler()


from google.cloud import storage
import pandas as pd
import numpy as np
from datetime import datetime
import io
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pyspark.sql.functions import explode, arrays_zip, first, col
from torch.nn.functional import softmax

# =========================== 0.Constants and Initialization

BUCKET_NAME = "adsp-capstone-enique-data"
BUCKET_PATH = f"gs://{BUCKET_NAME}"

PROCESSING_PATH = "data/relevant_process"
SOURCE_PATH = f"{PROCESSING_PATH}/3-relevant-submissions"
BACKUP_PATH = f"{PROCESSING_PATH}/4-relevant-submissions-bk"

COMBINED_PATH = f"{PROCESSING_PATH}/5-combined-files"

SENTI_PATH = f"{PROCESSING_PATH}/6-senti-submissions"
MAIN_SENTI_PATH = f"{BUCKET_PATH}/{SENTI_PATH}/senti_data.parquet"

HISTORICAL_SENTI_PATH = f"{SENTI_PATH}/historical"

HISTORICAL_SCORES_PATH = f"{BUCKET_PATH}/{HISTORICAL_SENTI_PATH}/historical_scores_data.parquet"


date_str = datetime.now().strftime('%Y%m%d%H%M%S')
new_combined_file_path = f'{COMBINED_PATH}/combined_{date_str}.csv'
new_historical_senti_file_path = f'{HISTORICAL_SENTI_PATH}/his_{date_str}.csv'

client = storage.Client()
bucket = client.get_bucket(BUCKET_NAME)

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

def save_data_to_parquet(df, file_path):
    try:
        df.to_parquet(file_path)
        print(f"Data saved to {file_path}")
    except Exception as e:
        print(f"Failed to save data: {e}")

def delete_old_combined_files(current_file_path):

    blobs = bucket.list_blobs(prefix=COMBINED_PATH)
    for blob in blobs:
        # Check if the blob is not the current file and not a directory marker
        if blob.name != current_file_path and not blob.name.endswith('/'):
            print(f"Deleting old file: gs://{bucket.name}/{blob.name}")
            blob.delete()
        else:
            print(f"Skipping deletion of: gs://{bucket.name}/{blob.name}")




# =========================== 1. Load Unprocessed Data
def load_unprocessed_data():
    dfs = []

    # Process files from both combined and unprocessed paths
    for prefix in [COMBINED_PATH, SOURCE_PATH]:
        # print(f"prefix={prefix}")
        blobs = list(bucket.list_blobs(prefix=f"{prefix}/"))
        for blob in blobs:
            print(blob.name)

            if blob.name.endswith('.csv'):

                df = read_csv_from_blob(blob)
                dfs.append(df)

                # Check if the blob should be moved based on its path
                if prefix == SOURCE_PATH:
                    new_blob_path = f"{BACKUP_PATH}/{blob.name.split('/')[-1]}"
                    bucket.rename_blob(blob, new_blob_path)
                    print(f"Moved {blob.name} to {new_blob_path}")

    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)

        # Fill NaN in 'fetch_datetime' with current datetime
        now = datetime.now()
        combined_df['fetch_datetime'] = pd.to_datetime(combined_df['fetch_datetime'], errors='coerce')
        combined_df['fetch_datetime'].fillna(now, inplace=True)

        # Sort, drop duplicates and handle data saving
        combined_df.sort_values(by='fetch_datetime', ascending=True, inplace=True)
        combined_df.drop_duplicates(subset='id', keep='last', inplace=True)

        # Save combined data and delete other combined files
        save_data_to_csv(combined_df, new_combined_file_path)

        # Delete old combined files except the newly saved one
        delete_old_combined_files(new_combined_file_path)

        return combined_df

    return pd.DataFrame()  # Return an empty DataFrame if no files were processed

# =========================== 2. Filter New Data, Removing scored data if 'id' and 'self_text_md5' hasn't been changed)
# and also update emotion and sentiment scores based on historical reference file if possible.
# In order to sovle 2 models computational expensive.


def filter_scored_senti(data, historical_scores_df, necessary_columns):
    # Ensure the DataFrame has the necessary columns initialized to NaN
    for col in necessary_columns:
        if col not in data.columns:
            data[col] = np.nan

    # Define the columns to merge from the historical data
    historical_columns = ['id', 'self_text_md5','his_emotion_labels', 'his_emotion_scores',
                          'his_sentiment_labels', 'his_sentiment_scores']

    # if historical_scores_df has zero records
    if historical_scores_df.empty:
        merged_data = data
    else:
        # Merge the current data with the historical scored data based on 'id' and 'self_text_md5'
        merged_data = data.merge(historical_scores_df[historical_columns],
                                 on=['id', 'self_text_md5'],
                                 how='left')

    # Update the current data with historical data where available
    for col in necessary_columns:
        hist_col = 'his_' + col
        merged_data[col] = merged_data[hist_col].where(merged_data[hist_col].notna(), merged_data[col])

    # Drop historical columns after updating
    merged_data.drop(columns=[col for col in historical_columns if 'his_' in col], inplace=True, errors='ignore')

    return merged_data


# =========================== 3. Assign sentiment score to each text by using `Hugging Face Transformers`

### Model `SamLowe/roberta-base-go_emotions`

# Load emotion analysis model and tokenizer
model_name = 'SamLowe/roberta-base-go_emotions'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()  # Set the model to evaluation mode


def emotion_analysis(text: str):
    try:
        if not text:
            return None, None  # Handle empty text input

        # Tokenization and converting to appropriate tensor format
        tokens = tokenizer([text], padding=True, truncation=True, max_length=512, return_tensors="pt")
        input_ids = tokens['input_ids']
        attention_mask = tokens['attention_mask']

        # Model prediction
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            probabilities = softmax(outputs.logits, dim=-1)

        # Convert predictions to appropriate labels
        scores = probabilities.numpy().flatten()  # Convert probabilities to a flat numpy array if needed
        labels = [model.config.id2label[i] for i in range(len(scores))]  # Map each index to its corresponding label
        # print(f"scores=={scores}")
        # print(f"labels=={labels}")
        return labels, scores
    except Exception as e:
        print(f"Exception in emotion analysis: {e}")
        return ["error"], [0.0]  # Return 'error' and 0.0 as the score if an exception occurs



### Model `cardiffnlp/twitter-roberta-base-sentiment-latest`

# Load sentiment analysis model and tokenizer
sentiment_model_name = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)

def sentiment_analysis(text: str):
    try:
        if not text:
            return [], []  # Return empty lists for empty text

        # Tokenization and converting to appropriate tensor format
        tokens = sentiment_tokenizer([text], padding=True, truncation=True, max_length=512, return_tensors="pt")
        input_ids = tokens['input_ids']
        attention_mask = tokens['attention_mask']

        # Model prediction
        with torch.no_grad():
            outputs = sentiment_model(input_ids, attention_mask=attention_mask)
            probabilities = softmax(outputs.logits, dim=-1).squeeze()

        # Convert probabilities to a list of scores
        scores = probabilities.tolist()

        # Map predictions to labels
        labels = [sentiment_model.config.id2label[i] for i in range(len(scores))]

        return labels, scores
    except Exception as e:
        print(f"Exception in sentiment_analysis: {e}")
        return [], []


def expand_sentiment(row):
    labels, scores = sentiment_analysis(row['clean_text'])
    return pd.Series([labels, scores])


# =========================== 4. Update historical scores file


def update_historical_scores(processed_df, historical_scores_df):
    # Concatenate historical and processed dataframes
    dfs = [historical_scores_df, processed_df[['id', 'self_text_md5', 'fetch_datetime',
                                               'emotion_labels', 'emotion_scores',
                                               'sentiment_labels', 'sentiment_scores']].rename(
        columns={"emotion_labels": "his_emotion_labels",
                 "emotion_scores": "his_emotion_scores",
                 "sentiment_labels": "his_sentiment_labels",
                 "sentiment_scores": "his_sentiment_scores",})]

    combined_df = pd.concat(dfs, ignore_index=True)


    # Drop rows with any NaN values in specified columns and deduplicate
    combined_df.dropna(subset=['his_emotion_labels', 'his_emotion_scores', 'his_sentiment_labels', 'his_sentiment_scores'], inplace=True)
    combined_df.sort_values(by='fetch_datetime', ascending=True, inplace=True)
    combined_df.drop_duplicates(subset='id', keep='last', inplace=True)

    # Save to Parquet
    try:
        combined_df.to_parquet(HISTORICAL_SCORES_PATH)
        print(f"The historical scores data contains:{combined_df.shape[0]}; Updated historical scores data saved to {HISTORICAL_SCORES_PATH}")
    except Exception as e:
        print(f"Failed to save data: {e}")

# =========================== 5. Update the main scores file that has been used for the predictive model
def update_main_scores(processed_df):

    try:
        scores_df = pd.read_parquet(MAIN_SENTI_PATH)
    except FileNotFoundError:
        scores_df = pd.DataFrame()  # If no file exists, start with an empty DataFrame

    # Prepare the list of DataFrames to concatenate
    dfs = [scores_df, processed_df]

    # Combine all DataFrames
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.sort_values(by='fetch_datetime', ascending=True, inplace=True)
    combined_df.drop_duplicates(subset='id', keep='last', inplace=True)

    # Save the updated DataFrame to Parquet
    combined_df.to_parquet(MAIN_SENTI_PATH)
    print(f"The main scores data contains:{combined_df.shape[0]}; Updated main scores data saved to {MAIN_SENTI_PATH}")


# ==================================== Main Function

def main_scheduler():
    # Step 1: Load Unprocessed Data
    unprocessed_data = load_unprocessed_data()
    if unprocessed_data.empty:
        print("No data loaded.")
        return "No data loaded."
    print(unprocessed_data.shape[0])

    # Read historical scored data

    try:
        historical_scores_df = pd.read_parquet(HISTORICAL_SCORES_PATH)
    except FileNotFoundError:
        historical_scores_df = pd.DataFrame()  # If no file exists, start with an empty DataFrame

    # Step 2:  Process Data(Filter New Data, Removing scored data if 'id' and 'self_text_md5' hasn't been changed)
    # and also update emotion and sentiment scores data if possiable)

    necessary_columns = ['emotion_labels', 'emotion_scores', 'sentiment_labels', 'sentiment_scores']
    filtered_all_data = filter_scored_senti(unprocessed_data, historical_scores_df, necessary_columns)

    needing_scores_data = filtered_all_data[filtered_all_data[necessary_columns].isna().any(axis=1)]

    updated_data = filtered_all_data.dropna(subset=necessary_columns)

    print(f"update + needing_scores_data ={updated_data.shape[0]} + {needing_scores_data.shape[0]}")

    print("========")

    filtered_data = needing_scores_data.copy()

    # Step 3: Apply sentiment and emotion analysis only for new data which hasn't been assigned emotion and senti scores

    if not filtered_data.empty:
        results = filtered_data['clean_text'].apply(lambda text: pd.Series(emotion_analysis(text)))
        filtered_data.loc[:, 'emotion_labels'] = results[0]
        filtered_data.loc[:, 'emotion_scores'] = results[1]

        sentiment_results = filtered_data.apply(lambda row: expand_sentiment(row), axis=1, result_type='expand')
        filtered_data.loc[:, 'sentiment_labels'] = sentiment_results[0]
        filtered_data.loc[:, 'sentiment_scores'] = sentiment_results[1]

    print(f"processed_data ={filtered_data.head(1)}")

    # Concat old and new  data
    processed_df = pd.concat([updated_data, filtered_data], ignore_index=True)
    # print(filtered_data)
    print(f"processed_df=={processed_df.shape[0]}")


    # ************** Explode emotion and sentiment Data Start**************

    # def zip_and_explode(df, col_labels, col_scores, prefix):

    #     # Zip the labels and scores together
    #     df['zipped'] = df.apply(lambda row: list(zip(row[col_labels], row[col_scores])), axis=1)

    #     # Explode the zipped column
    #     df_exploded = df.explode('zipped')

    #     # Create separate columns for labels and scores
    #     df_exploded[f'{prefix}label'], df_exploded[f'{prefix}score'] = zip(*df_exploded['zipped'])

    #     return df_exploded

    # def pivot_df(df, index_col, label_col, score_col, prefix):

    #     # Create a pivot table
    #     pivot_table = df.pivot_table(index=index_col, columns=label_col, values=score_col, aggfunc='first').reset_index()

    #     # Flatten the MultiIndex columns and add prefix
    #     pivot_table.columns = [f"{prefix}{col}" if col != index_col else col for col in pivot_table.columns]

    #     return pivot_table

    # print("Exploding Starting...")
    # # Step 1: Zip and explode emotion and sentiment columns
    # emotion_exploded_df = zip_and_explode(processed_df, 'emotion_labels', 'emotion_scores', 'emotions_')
    # sentiment_exploded_df = zip_and_explode(processed_df, 'sentiment_labels', 'sentiment_scores', 'sentiment_')

    # print("Exploding step1 done...")
    # # Step 2: Pivot the exploded DataFrames
    # emotion_pivot_df = pivot_df(emotion_exploded_df, 'id', 'emotions_label', 'emotions_score', 'emotions_')
    # sentiment_pivot_df = pivot_df(sentiment_exploded_df, 'id', 'sentiment_label', 'sentiment_score', 'sentiment_')

    # print("Exploding step2 done...")
    # # Step 3: Join the pivoted DataFrames back to the original DataFrame
    # # cols_to_keep = [col for col in processed_df.columns if col not in ['emotion_labels', 'emotion_scores', 'sentiment_labels', 'sentiment_scores', 'zipped']]
    # # original_df = processed_df[cols_to_keep]
    # original_df = processed_df.copy()

    # print("Exploding step3 done...")
    # result_df = original_df.merge(emotion_pivot_df, on='id', how='left').merge(sentiment_pivot_df, on='id', how='left')

    # print("Exploding merged...")
    # ************** Explode emotion and sentiment Data End**************

    # Save results to historical path
    save_data_to_csv(processed_df, new_historical_senti_file_path)
    # save_data_to_csv(result_df, new_historical_senti_file_path)


    # Step 4: Update historical scores file
    update_historical_scores(processed_df, historical_scores_df)
    # update_historical_scores(result_df, historical_scores_df)


    # Step 5: Update the main scores file that has been used for the predictive model
    update_main_scores(processed_df)
    # update_main_scores(result_df)




    # Cleanup
    if not filtered_data.empty:
        delete_old_combined_files(f'{COMBINED_PATH}/0.csv')

