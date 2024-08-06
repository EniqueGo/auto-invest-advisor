import pandas as pd
import numpy as np
import ta
from google.cloud import storage
#import functions_framework


btc_df = pd.read_csv("gs://adsp-capstone-enique-data/data/Bitcoin/historical_bitcoin_data.csv")

btc_df['ds'] = pd.to_datetime(btc_df['ds'] )
btc_df['RSI'] = ta.momentum.RSIIndicator(btc_df['y']).rsi()

macd = ta.trend.MACD(btc_df['y'])
btc_df['MACD'] = macd.macd()
btc_df['MACD_signal'] = macd.macd_signal()
btc_df['MACD_diff'] = macd.macd_diff()


# Calculate typical price for MFI
btc_df['typical_price'] = (btc_df['y_upper'] + btc_df['y_lower'] + btc_df['y']) / 3

# Calculate raw money flow
btc_df['raw_money_flow'] = btc_df['typical_price'] * btc_df['volume']

# Calculate positive and negative money flow
btc_df['positive_money_flow'] = np.where(btc_df['typical_price'] > btc_df['typical_price'].shift(1), btc_df['raw_money_flow'], 0)
btc_df['negative_money_flow'] = np.where(btc_df['typical_price'] < btc_df['typical_price'].shift(1), btc_df['raw_money_flow'], 0)


# Calculate money flow ratio
positive_money_flow_rolling = btc_df['positive_money_flow'].rolling(window=14).sum()
negative_money_flow_rolling = btc_df['negative_money_flow'].rolling(window=14).sum()
money_flow_ratio = positive_money_flow_rolling / negative_money_flow_rolling

# Calculate MFI
btc_df['MFI'] = 100 - (100 / (1 + money_flow_ratio))


# Bollinger Bands for volatility indicator
rolling_mean = btc_df['y'].rolling(window=20).mean()
rolling_std = btc_df['y'].rolling(window=20).std()
btc_df['bollinger_upper'] = rolling_mean + (rolling_std * 2)
btc_df['bollinger_lower'] = rolling_mean - (rolling_std * 2)

btc_df.dropna(inplace=True)



btc_df.to_csv("gs://adsp-capstone-enique-data/data/Bitcoin/bitcoin_data.csv", index=False)