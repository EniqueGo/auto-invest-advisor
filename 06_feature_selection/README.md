This folder contains all resources related to the feature selection process, including:

- **feature_selection.py:** A Python script that implements various feature selection methods such as correlation selection, Random Forest feature importance, variance thresholding, and recursive feature elimination.
- **reddit_preprocess_feature_selection_v2.0.0a1.ipynb:** A Jupyter notebook documenting the preprocessing and feature selection steps applied to Reddit data, including exploratory data analysis and feature selection criteria.



| **Selected Feature**                      | **Description**                                                                                                  |
|----------------------------------|------------------------------------------------------------------------------------------------------------------|
| bollinger_lower                  | The lower band of the Bollinger Bands indicates the lower limit of the normal price range based on recent volatility. |
| bollinger_upper                  | The upper band of the Bollinger Bands represents the upper limit of the normal price range based on recent volatility. |
| MACD                             | The Moving Average Convergence Divergence, a trend-following momentum indicator showing the relationship between two moving averages of a security's price. |
| MACD_diff                        | The difference between the MACD line and the MACD signal line is used to identify changes in a trend's strength, direction, momentum, and duration. |
| MACD_signal                      | The signal line of the MACD, which is the 9-day EMA of the MACD, is used to generate buy and sell signals.        |
| MFI                              | The Money Flow Index, a momentum indicator that uses price and volume data to identify overbought or oversold conditions. |
| negative_money_flow              | The total value of the negative money flow represents the sum of the money flowing out of a security.         |
| num_of_trades                    | The total number of trades executed for the asset during a specific period.                                 |
| open_price                       | The price at which an asset first trades upon opening an exchange on a trading day.                       |
| positive_money_flow              | The total value of the positive money flow, representing the sum of the money flowing into security.           |
| prophet_trend                    | The trend component of the Prophet model's forecast indicates the general direction in which the price moves. |
| prophet_weekly                   | The weekly seasonality component of the Prophet model shows recurring weekly patterns in the price data.      |
| prophet_yhat                     | The predicted value of the price from the Prophet model.                                                        |
| prophet_yhat_lower               | The lower bound of the predicted value from the Prophet model, providing a confidence interval.                  |
| prophet_yhat_upper               | The upper bound of the predicted value from the Prophet model, providing a confidence interval.                  |
| quote_asset_volume               | The volume of the quote asset traded during a specific period.                                             |
| raw_money_flow                   | The product of the typical price and volume calculating the Money Flow Index.                    |
| RSI                              | The Relative Strength Index, a momentum oscillator that measures the speed and change of price movements to identify overbought or oversold conditions. |
| sentiment_negative               | The proportion of negative sentiment extracted from social media or other textual data sources.                  |
| sentiment_negative_volatility    | The volatility of negative sentiment over a given period.                                                  |
| sentiment_neutral                | The proportion of neutral sentiment extracted from social media or other textual data sources.                  |
| sentiment_neutral_volatility     | The volatility of neutral sentiment over a given period.                                                   |
| sentiment_positive               | The proportion of positive sentiment extracted from social media or other textual data sources.                  |
| sentiment_positive_volatility    | The volatility of positive sentiment over a given period.                                                  |
| taker_buy_base_asset_volume      | The volume of the base asset takers bought during a specific period.                                     |
| taker_buy_quote_asset_volume     | The volume of the quote asset takers bought during a specific period.                                    |
| tw_avg_negative                  | The time-weighted average of negative sentiment over a given period.                                             |
| tw_avg_neutral                   | The time-weighted average of neutral sentiment over a given period.                                              |
| tw_avg_positive                  | The time-weighted average of positive sentiment over a given period.                                             |
| typical_price                    | The average of the high, low, and close prices for a given period used in various technical indicators.        |
| volume                           | The total asset quantity traded during a specific period.                                            |
| y_lower                          | The lower bound of the actual observed value, providing a confidence interval for the real price.               |
| y_upper                          | The actual observed value's upper bound provides a confidence interval for the real price.               |
