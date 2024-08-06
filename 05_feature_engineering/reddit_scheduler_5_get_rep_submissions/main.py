import base64
import functions_framework

import pandas as pd
import numpy as np
import os, sys
from datetime import datetime

from google.cloud import storage


BUCKET_NAME = "adsp-capstone-enique-data"

HORIZON_DAYS = 5 # Forcast next 5 days
HORIZON_HOURS = HORIZON_DAYS * 24

client = storage.Client()
bucket = client.get_bucket(BUCKET_NAME)



def unixtime_to_datetime(unix_timestamp, datetime_format='%Y-%m-%d %H:%M:%S'):
    """Converts a single Unix timestamp to a formatted datetime string."""

    # Convert Unix timestamp to datetime object
    dt_object = datetime.utcfromtimestamp(unix_timestamp)

    # Format datetime object as "m-d-y HH:mm:ss"
    formatted_datetime = dt_object.strftime(datetime_format)

    return formatted_datetime


def save_data_to_csv(data, file_path):
    if not data.empty:
        buffer = data.to_csv(index=False)
        blob = bucket.blob(f"{file_path}")
        blob.upload_from_string(buffer, content_type='text/csv')
        print(f"Data saved to gs://{bucket.name}/{file_path}")



def time_weighted_avg(series, window=120, decay_rate=0.5):
    weights = np.array([decay_rate**((window-i)/window) for i in range(window)])
    weighted_sum = np.convolve(series, weights, mode='valid')
    return np.concatenate((np.full(window-1, np.nan), weighted_sum / weights.sum()))


# Triggered from a message on a Cloud Pub/Sub topic.
@functions_framework.cloud_event
def hello_pubsub(cloud_event):
    # Print out the data from Pub/Sub, to prove that it worked
    print(base64.b64decode(cloud_event.data["message"]["data"]))

    df = pd.read_parquet("gs://adsp-capstone-enique-data/data/relevant_process/6-senti-submissions/senti_data.parquet")
    df = df[["id", "title","created", "upvote_ratio","permalink", "url", "num_comments", "clean_text", "sentiment_labels", "sentiment_scores" ]]


    # Convert created column to int
    df['created'] = df['created'].astype(int)

    # for loop each row in the dataframe
    for index, row in df.iterrows():

        for i in range(0, len(row['sentiment_labels'])):

            # Add the sentiment label and score to the dataframe
            df.at[index, "sentiment_"+row['sentiment_labels'][i]] = row['sentiment_scores'][i]


        # Convert the Unix timestamp to a formatted datetime string
        df.at[index, "created"] = unixtime_to_datetime(row['created'])


    df_rep_posts = df.copy()

    # Choose the latest max 500 posts
    df_rep_posts = df_rep_posts.sort_values(by=['created'], ascending=False).head(500)

    # Create "time_group" column to group the posts by every 6 hours
    df_rep_posts['time_group'] = pd.to_datetime(df_rep_posts['created']).dt.floor('6H')

    # Create "senti_most" column to compare "sentiment_negative", "sentiment_neutral", and "sentiment_positive" if any of them is the highest, then assign the highest label to "senti_most"
    df_rep_posts['senti_most'] = df_rep_posts[["id", 'sentiment_negative', 'sentiment_neutral', 'sentiment_positive']].idxmax(axis=1)

    # Change values from 'sentiment_negative', 'sentiment_neutral', 'sentiment_positive' to negative, neutral, positive
    df_rep_posts['senti_most'] = df_rep_posts['senti_most'].str.replace('sentiment_', '')

    # Save df_rep_posts
    df_rep_posts.to_csv("gs://adsp-capstone-enique-data/results/rep_posts.csv", index=False)

    print("rep_posts generated")
