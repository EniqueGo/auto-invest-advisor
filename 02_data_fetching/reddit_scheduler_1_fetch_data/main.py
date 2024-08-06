import base64
import functions_framework
import pandas as pd
import os
import praw
from prawcore.exceptions import NotFound
from datetime import datetime
from google.cloud import storage
import hashlib

# Setup Google Cloud Storage client
client = storage.Client()
bucket_name = 'adsp-capstone-enique-data'
bucket = client.get_bucket(bucket_name)

reddit = praw.Reddit(
    client_id=os.environ["reddit_client_id"],
    client_secret=os.environ["reddit_client_secret"],
    password=os.environ["reddit_password"],
    user_agent="my user agent",
    username="EniqueGo"
)

def fetch_reddit_data():
    data = []
    subreddit = reddit.subreddit('Bitcoin')
    try:
        for submission in subreddit.new(limit=1000):
            append_submission_data(submission, data)

        print("reddit data:", data)
        save_data_to_csv(data)
    except Exception as e:
        print(f"Error fetching data: {e}")
        save_data_to_csv(data)  # Save whatever has been fetched before crashing
    finally:
        print("Completed data fetching attempt.")

def save_data_to_csv(data):
    if data:
        filename = f'reddits_{datetime.now().strftime("%Y%m%d%H%M")}.csv'
        df = pd.DataFrame(data)
        buffer = df.to_csv(index=False)
        blob = bucket.blob(f"real-time-reddit-sub/{filename}")
        blob.upload_from_string(buffer, content_type='text/csv')
        print(f"Reddit Data saved to gs://{bucket_name}/real-time-reddit-sub/{filename}")

def userStatus(redditor):
    try:
        redditor.is_suspended
        return "suspended"
    except NotFound:
        return "deleted/suspended"
    except Exception:
        return "error"

def append_submission_data(submission, data):

    if_author_exists = True

    attrs_of_submission = vars(submission)

    if ("author" in attrs_of_submission) and (submission.author is not None):
        author = submission.author
        attrs_of_author= vars(author)
        if "id" not in attrs_of_author:
            if_author_exists = False

    else:
        if_author_exists = False


    try:

        data.append({
            'id': submission.id,
            'title': submission.title,
            'name': submission.name,
            'score': submission.score,
            'upvote_ratio': submission.upvote_ratio,
            'url': submission.url,
            'num_comments': submission.num_comments,
            'permalink': submission.permalink,
            'created': submission.created_utc,
            'body': submission.selftext,
            'author_flair_text': submission.author_flair_text,
            'author_id': author.id if if_author_exists else 0,
            'distinguished': submission.distinguished,
            'is_original_content': submission.is_original_content,
            'is_self': submission.is_self,
            # 'link_flair_template_id': submission.link_flair_template_id,
            'link_flair_text': submission.link_flair_text,
            'author_name': author.name if if_author_exists else "Deleted",
            'author_is_employee': author.is_employee if if_author_exists else False,
            'author_is_mod': author.is_mod if if_author_exists else False,
            'author_is_gold': author.is_gold if if_author_exists else False,
            # 'author_is_suspended': author.is_suspended if author else False,
            'author_link_karma': author.link_karma if if_author_exists else 0,
            'author_comment_karma': author.comment_karma if if_author_exists else 0,
            'author_has_verified_email': author.has_verified_email if if_author_exists else False,
            'author_cakeday': author.created_utc if if_author_exists else None,
            'author_status': userStatus(author) if if_author_exists else "Deleted",
            'self_text_md5': hashlib.md5(submission.selftext.encode('utf-8')).hexdigest() if type(submission.selftext) == str else '',
            'fetch_datetime': datetime.now()
        })
    except Exception as e:
        #logging.error(f"Error processing submission {submission.id} - {e}")
        print(f"Error processing submission - {e}")

@functions_framework.cloud_event
def hello_pubsub(cloud_event):
    fetch_reddit_data()

