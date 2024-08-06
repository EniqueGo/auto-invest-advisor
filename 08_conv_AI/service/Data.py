import os

import pandas as pd
import pytz
import streamlit as st
from datetime import datetime, timezone, timedelta
import yfinance as yf
from google.cloud import storage
from cachetools import cached, LRUCache, TTLCache

from stockstats import wrap, unwrap

def get_time_zone():
    return 'US/Eastern'

def get_time_now():
    """
    Get the current time
    :return: current time
    """
    #@TODO: to remove this line
    #Initialize time as 2023-12-27 09:00:00, using timezone of 'US/Eastern'
    # This is equivalent to the UTC time 2023-12-27 14:00:00
    #curr_time = datetime(2023, 12, 27, 9, 0, 0, tzinfo=pytz.timezone(get_time_zone()))

    #return curr_time

    #date_now = datetime.now() - timedelta(days=1)
    date_now = datetime.now()
    eastern = pytz.timezone(get_time_zone())

    return date_now.astimezone(eastern)

def convert_df_col_time_from_utc_to_tz(df, col='ds'):
    """
    Convert the time in the column from UTC to the timezone
    :param df: dataframe
    :param col: column name
    :return: dataframe
    """
    eastern = pytz.timezone(get_time_zone())
    df[col] = pd.to_datetime(df[col])
    df[col] = df[col].dt.tz_localize('UTC').dt.tz_convert(eastern)
    return df

def get_data_dir():
    """
    Get the data directory based on the environment
    :return: data directory
    """
    if st.secrets.env.env == 'dev':
        return st.secrets.env_dev.data_dir
    else:
        return st.secrets.env_prod.data_dir

@cached(cache=TTLCache(maxsize=32, ttl=60*10))
def get_btc_prediction():
    """
    Read the btc_predictions.csv
    :return: data
    """
    data = pd.read_csv(get_data_dir() + 'btc_predictions.csv')
    data['ds'] = pd.to_datetime(data['ds'])

    data = convert_df_col_time_from_utc_to_tz(data)

    # Change column name of yhat to close_price
    data.rename(columns={'yhat': 'close_price'}, inplace=True)

    return data[['ds', 'close_price']]

@cached(cache=TTLCache(maxsize=32, ttl=60*10))
def get_hist_btc_from_yh(from_date, to_date=None, interval='1h'):
    """
    Get the historical bitcoin data from Yahoo Finance
    :param from_date: from date e.g. '2023-03-01'
    :param to_date: to date e.g. '2023-04-02', None for today
    :param interval: interval e.g. '1h'
    :return: data
    """
    btc_yf = yf.Ticker('BTC-USD')
    btc_yf_hist = btc_yf.history(start=from_date, end=to_date, interval='1h')
    btc_yf_hist.reset_index(inplace=True)
    btc_yf_hist = btc_yf_hist[['Datetime', 'Close']]

    btc_yf_hist.columns = ['ds', 'close_price']

    # Convert ds to datetime
    eastern = pytz.timezone(get_time_zone())
    btc_yf_hist['ds'] = pd.to_datetime(btc_yf_hist['ds'])
    btc_yf_hist['ds'] = btc_yf_hist['ds'].dt.tz_convert(eastern)

    return btc_yf_hist

@cached(cache=TTLCache(maxsize=32, ttl=60*10))
def get_hist_btc_from_yh_for_tech_indicator(from_date, to_date=None, interval='1h'):
    """
    Get the historical bitcoin data from Yahoo Finance
    :param from_date: from date e.g. '2023-03-01'
    :param to_date: to date e.g. '2023-04-02', None for today
    :param interval: interval e.g. '1h'
    :return: data
    """
    btc_yf = yf.Ticker('BTC-USD')
    btc_yf_hist = btc_yf.history(start=from_date, end=to_date, interval='1h')
    btc_yf_hist.reset_index(inplace=True)

    # Convert ds to datetime
    eastern = pytz.timezone(get_time_zone())
    btc_yf_hist['Datetime'] = pd.to_datetime(btc_yf_hist['Datetime'])
    btc_yf_hist['Datetime'] = btc_yf_hist['Datetime'].dt.tz_convert(eastern)

    return btc_yf_hist



@cached(cache=TTLCache(maxsize=32, ttl=60*10))
def get_sentiment_data():
    """
    Read the sentiment_over_time.csv file
    :return: data
        ds: datetime
        sentiment_score: sentiment score
    """
    data = pd.read_csv(get_data_dir() + 'sentiment_over_time.csv')

    data = data[['ds', 'sentiment_score']]

    # number of rows
    n = data.shape[0]

    # Choose the last min(n, 24*30) rows
    data = data[-min(n, 24*30):]

    # Convert ds to datetime and convert to the timezone
    data['ds'] = pd.to_datetime(data['ds'])
    data = convert_df_col_time_from_utc_to_tz(data)

    data['sentiment_score'] = data['sentiment_score'].astype(int)

    #Remove the index
    data.reset_index(inplace=True, drop=True)

    return data

@cached(cache=TTLCache(maxsize=32, ttl=60*20))
def get_rep_posts():
    """
    Read the rep_posts.csv file
    :return:
    """
    data = pd.read_csv(get_data_dir() + 'rep_posts.csv',index_col=False)

    # Convert ds to datetime and convert to the timezone
    data['created'] = pd.to_datetime(data['created'])
    data = convert_df_col_time_from_utc_to_tz(data, col='created')

    #result is a dictionary
    result = {}

    # Put time range
    result['earliest_time'] = data['created'].min()

    # earliest day, only keep day string
    result['earliest_day_str'] = result['earliest_time'].strftime('%Y-%m-%d')

    # give a created_str
    data['created_str'] = data['created'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Chosen post id
    chosen_post_ids = []

    for senti in ['positive', 'neutral', 'negative']:
        result[senti] = {}

        # only keep the ones not in chosen_post_ids
        data = data[~data['id'].isin(chosen_post_ids)]

        # Sort the data by time_group and upvote_ratio
        df_rep_posts = data[data['senti_most'] == senti].sort_values(
            by=['time_group', 'upvote_ratio'], ascending=[False, True]).head(1)

        # if not empty
        if df_rep_posts.empty:
            result[senti]['title'] = ''
            result[senti]['upvote_ratio'] = 0
            result[senti]['url'] = '#'
            result[senti]['created'] = ''
            result[senti]['clean_text'] =  "Opps, no representative post found since "+result['earliest_day_str']+". Please check back later."
            continue

        result[senti]['title'] = df_rep_posts['title'].values[0]
        result[senti]['upvote_ratio'] = df_rep_posts['upvote_ratio'].values[0]
        result[senti]['url'] = "https://www.reddit.com" + df_rep_posts['permalink'].values[0]
        result[senti]['created'] = df_rep_posts['created_str'].values[0]


        # clean_text max length is 230
        result[senti]['clean_text'] = df_rep_posts['clean_text'].values[0][:220] + '...'

        # add id
        chosen_post_ids.append(df_rep_posts['id'].values[0])

    return result


@cached(cache=TTLCache(maxsize=32, ttl=60*30))
def get_merged_hist_pred_btc():
    """
    Merge the historical and prediction data
    1. Get the current time
    2. Get the historical data from Yahoo Finance from 10 days ago of the current time
    3. Get the prediction data
    4. Merge the historical and prediction data:
        - Use historical data until the last date of historical data
        - Use prediction data after the last date of historical data
        - Create a new column 'if_pred' to indicate if the data is prediction data:
            0 for historical data, 1 for prediction data
    :return: merged data
    """
    # Get the current time
    curr_time = get_time_now()

    # Get the historical data from Yahoo Finance from 10 days ago of the current time
    hist_data = get_hist_btc_from_yh(from_date=(curr_time - pd.Timedelta(days=10)).strftime('%Y-%m-%d'))

    hist_data = hist_data[hist_data['ds'] < curr_time]

    # Get the prediction data
    pred_data = get_btc_prediction()

    # Remove the rows where ds is in the past
    pred_data = pred_data[pred_data['ds'] >= curr_time]

    # pred_data can maximum have 120 rows
    pred_data = pred_data.head(120)

    # Merge the historical and prediction data
    # Use historical data until the last date of historical data
    # Use prediction data after the last date of historical data
    # Create a new column 'if_pred' to indicate if the data is prediction data:
    # 0 for historical data, 1 for prediction data
    hist_data['if_pred'] = 0
    pred_data['if_pred'] = 1


    merged_data = pd.concat([hist_data, pred_data],ignore_index=True)

    # Round the close_price to two decimal places
    merged_data['close_price'] = merged_data['close_price'].round(2)

    return merged_data


@cached(cache=TTLCache(maxsize=32, ttl=60*15))
def get_hist_btc_ob_from_stockstats():
    """
    Get the historical bitcoin data from Yahoo Finance and convert it to stockstats object
    :return: stockstats object
    """

    eastern = get_time_zone()
    curr_time = get_time_now()
    curr_time_str = curr_time.strftime('%Y-%m-%d')
    from_date_str = (curr_time - pd.Timedelta(days=10)).strftime('%Y-%m-%d')


    btc_yf = yf.Ticker('BTC-USD')
    btc_yf_hist = btc_yf.history(start=from_date_str, end=curr_time_str, interval='1h')
    btc_yf_hist.reset_index(inplace=True)

    btc_yf_hist['Datetime'] = pd.to_datetime(btc_yf_hist['Datetime'])
    btc_yf_hist['Datetime'] = btc_yf_hist['Datetime'].dt.tz_convert(eastern)

    btc_yf_hist = btc_yf_hist[['Datetime', 'Close', 'High', 'Low', 'Volume']]

    ob = wrap(btc_yf_hist)

    return ob