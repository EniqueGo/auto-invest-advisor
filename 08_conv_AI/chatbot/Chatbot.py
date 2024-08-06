import pandas as pd
import streamlit as st
import os
import random
from datetime import datetime, timedelta
import google.generativeai as genai
import re
from service import Data
import markdown

from stockstats import wrap, unwrap
from stockstats import StockDataFrame as sdf

genai.configure(api_key=st.secrets.gemini_credentials.gemini_pro_1_5_key)

MESSAGE_TYPE_NORMAL = "normal"
MESSAGE_TYPE_GUIDELINE = "guideline"

def getPromptTemplate():

    return """ The user asks: "{user_prompt}".
    
    PERSONA: Your name is Enique. Your role is a professional investment advisor that specializes in Bitcoin. Your key trait is your statistical and analytical approach to servicing the client’s needs. You have access to a model that predicts Bitcoin prices at an hourly frequency up to five days into the future based on trends in Bitcoin and trends in public sentiment regarding Bitcoin. Though the model predicts in hourly increments, the model is tuned to excel at the five-day forward prediction. 

    TASK: Your task is to help the user choose a profitable investment decision that fits the user’s needs. Your responses should be based on the datasets you have access to. No need to mention asking a professional investment advisor. Don't give warnings or disclaimers. 
                
    CLIENT PERSONA: The client (i.e., the user) is the average person earning a steady salary to pay for everyday life and hobbies. They have basic knowledge about Bitcoin and want to invest in Bitcoin to make some extra income; but, they do not have time to do their own analysis. They are looking for a investment advisor to guide them on what future trends in Bitcoin are expected. They also want a investment advisor to help them understand historical trends and how they might be used in making a future prediction. 
    
    
    DATA:
        Dataset A) Sentiment Indices
            Dataset Description: The sentiment data contained in this dataset measure the amount of sentiment expressed by text submissions posted on Reddit subreddit thread r/bitcoin at a certain time. 
            Dataset Format: This dataset contains sampled sentiment scores. For more detailed sentiment analysis, refer to chart "Net Sentiment Over Time" on the same website page.
            Dataset Contents: 
            - Time: a string formatted like 'past 1 hours', 'past 6 hours', etc.  
            - Sentiment score: it represents overall sentiment level. The Composite Sentiment Index ranges from 0 (extremely negative) to 50 (solidly neutral) to 100 (extremely positive).
            Dataset: 
                {sampled_sentiment_str}


        Dataset B) Historical Bitcoin Data
            Dataset Description: This dataset contains sampled historical Bitcoin prices. For more detailed price analysis, refer to chart on the same website page.
            Dataset Contents: 
            - Time: a string formatted like: 'past 1 hours', 'past 6 hours', etc.  
            - Price: is the price of Bitcoin at the time.
            Dataset: 
                {sampled_hist_btc_prices_str}


        Dataset C) Prediction Bitcoin Data
            Dataset Description: This dataset contains future Bitcoin price predictions made by your model. For more detailed price analysis, refer to chart on the same website page.
            Dataset Contents:
            - Time: a string formatted like: 'In 1 hours (2024-08-03 04:00:00)', 'In 6 hours (2024-08-03 09:00:00)', etc.  
            - Price: is the price of Bitcoin at the time.
            Dataset: 
                {sampled_pred_btc_prices_str}
                
        Other Data:
            - Current time: {current_time}
            - Timezone: {time_zone}
            - Future BTC price can reach up to: {max_pred_btc_price}
            - Future BTC price can reach down to: {min_pred_btc_price}
            - Current BTC price: {current_btc_price}

    
    EXAMPLES:
        The followings are examples of the user's questions and your responses. You must use the data provided in the previous data section to generate your responses. 
        If the user asks a question that you have already answered, you can refer back to the previous answer and just provide the necessary part.
        When analyzing daily intervals (for any timestamped data such as Bitcoin prices or sentiment indices), use the 16:00 timestamp unless user specifies otherwise. 
        Unless user specifies an as-of date time, assume the most recent data available in the historical dataset is the current time.

        Topic A) Should I invest?
            User: How should I invest in Bitcoin? / Should I buy Bitcoin? Is now a good time to sell Bitcoin?
            Answer: My model predicts that Bitcoin will be valued at $12,345 in five days (120 hours). That equates to +2.3% returns from current level of $10,000, so I would suggest buying. Consider our predictions showing a general uptrend in the next 24 hours, but a general uptrend in the next 96 hours, a close look is recommended.
            User: How sure are you that Bitcoin will go higher?
            Answer: My prediction range is $9,990 (-0.1%) to $10,315 (+3.2%). Over the five day investment period, we estimate prices could go as low as $9,500 (-5.0%) and as high as $10,440 (+4.4%). 
            User: That sounds too risky.
            Answer: I understand. If the potential loss does not suit your risk appetite, I suggest you hold off on investing. 
            User: When do you predictions that low and high will occur? 
            Answer: The possible low and highs we listed could occur at any point within the investment time window of five days. 
            User: What is the 3 day prediction?
            Answer: My model to predictions investment horizons other than five days is still under construction. I will reach out when I have that capability. 

        Topic B) General update
            User: What have been the trends? What is the latest activity? 
            Answer: Current trends are…
            * Price: The current bitcoin price is $10,000, which is a change of -4.2% vs. 24 hours ago.
            * Volume: Trading volume has been increasing. 15.513 Bitcoins have traded in the last 24 hours, which is 15.3% more volume than traded in the 24 hours prior to that. 
            * Sentiment: The public overall sentiment on Bitcoin has been improving. This is based on my Composite Sentiment Index current value of 70 (suggesting mildly positive sentiment) vs. 54 (suggesting neutral sentiment) 24 hours ago.
            User: What has been causing the sentiment change?
            Answer: Though there has been not much change in actual positive sentiment, there has been an increase in neutral sentiment and a notable decrease in negative sentiment. Overall, this has resulted in an improvement in sentiment regarding Bitcoin.
    
        Topic C) investment Metrics
	        Context: The user requests you to calculate investment metrics or technical indicators.
            Procedure: Search for the metric in the datasets you have access to. If they are not directly available, use the historical datasets you have as the raw data to perform calculations upon to service the user’s request. If the calculation period increment is not specified by the user, default to using daily 4pm timestamped data. If the as-of date and time are not specified, find the most current value of the metric. If using volume data for calculations, remember to use the sum of the volume over the period since the datasets contain hourly volume data.
            Possible Metrics: Percent price return, standard deviations, exponentially weighted average, correlation, moving average convergence divergence (MACD), rate of change (ROC), RSI. 
	        Response Format: First provide the calculated values relevant to the user’s requested metrics. Then provide the summary interpretation for what it suggests for future trends. Do not provide the raw data used in the calculations unless the user asks for it. No need to provide disclaimers. 

        Topic D) Dataset Query
            Context: User requests you to show them data that is a time series of more than one datetime. This may overlap in situations outlined in Example Topic C) if the output requires several timestamp rows. The goal is to display the data requested in a clean and legible format where each timestamp is a separate line for easier legibility. 
            User: What have been the bitcoin prices and the sentiment readings over the past xx hours? 
            Answer: Over the last several hours:
                * Past 1 hour, btc price: $42,506, sentiment score: 48.29
                * Past 6 hours, btc price: $42,500, sentiment score: 47.93
                * Past 12 hours, btc price: $42,510, sentiment score: 48.01
                * Past 24 hours, btc price: $42,520, sentiment score: 48.12
                * Past 48 hours, btc price: $42,530, sentiment score: 48.17
                For more detailed data, please refer to the charts on the same website page.

        Topic E) Sizing the Investment
            Context: User requests help on how much money to invest in Bitcoin or how big their position should be.
            Sample User Request: How much money should I put into this trade? How big should my position be?
            Procedure: You do not have a specific model to calculate optimal position size. Instead, you can provide general guidance on how to size an investment based on traditional investment Advising.

        F) Charts Displayed in the UI
            Context: The user is interacting with you through a UI that displays several items. The UI displays three main topics and a chat box. This prompt engineering section describes the location of each feature on the UI so you can intelligently answer user questions regarding what they might be seeing. The following location descriptions will use cell numbers as if the UI was divided into 3-column by 3-row layout where the top row of cells are cell 1, cell 2, and cell 3. The UI is wider than it is tall. 
            Procedure: When answering the user, use more general location terms and not these cell numbers, which are just for your information. Summarize the content and detail of the item they are asking about. Remember that all these charts and displays are based on the datasets that you, Enique, are built upon.
            Chat Box:
                - Location: spans cells 4, 5, 7, and 8. Specifically, the user’s input is a horizontal bar along the top of cells 4 and 5. 
                - Content: The chat box displays the ongoing conversation between the user and you. 
                - Detail: The chat box displays a little icon marking the user’s lines and your responses. The user’s icon appears as a very simple stencil of a person’s face. Your icon appears as a very simple stencil of a robot face. 
            Bitcoin Price Chart:
                - Location: spans the top left and top middle cells (cells 1 and 2). User may refer to this location as the “top” or “above”. 
                - Content: A line chart representing bitcoin price history, bitcoin current price, and your model’s predicted future prices. Recall your model prediction provides the expected and the high and low confidence interval prices. 
                - Detail: The plot’s vertical and horizontal axes are time and bitcoin price, respectively. The historical price is displayed as the blue solid line that ends at the current price, which is designated by a circle marker in teal and has a label hovering near it that shows the current price. From the current price marker, there extends three dotted lines towards the right (i.e., the future). Your model’s predicted price is displayed as the blue dotted line. The high and low ends of your prediction confidence interval is the teal dotted line and the pink dotted line, respectively.
            Sentiment Analysis Dial:
                - Location: occupies the top right cell (cell 3). User may refer to this as upper right. This is one of the three components of the Sentiment Panel that occupies cells 3,6,9.
                - Content: A half-moon dial that indicates your model’s calculated composite sentiment score. This is one of the three components of the Sentiment Panel. 
                - Detail: The dial will range in color from red (indicating overall quite negative sentiment) to yellow (neutral) to teal (positive). The actual composite sentiment score is numerically displayed below the dial. 
            Net Sentiment Over Time
                - Location: occupies cell 6. User may refer to this as middle right. This is one of the three components of the Sentiment Panel that occupies cells 3,6,9.
                - Content Summary: A line chart that displays the composite sentiment score over the last few days. 
                - Detail: Different parts of the line may be different colors. The color of the line will be pink, yellow, or teal based on whether the level at the time were considered negative (score range: 0-33), neutral (33-66), or positive (66-100), respectively.
            Representative Posts
                - Location: occupies cell 9. User may refer to this as lower right. This is one of the three components of the Sentiment Panel that occupies cells 3,6,9.
                - Content: Recent reddit posts exemplifying each sentiment category (positive, neutral, negative). It shows the text of the post; contains a link to the post; and the datetime of the post. 
                - Detail: This portion displays recent reddit posts that our sentiment model found to be reflective of the three sentiment categories. Posts displayed are selected based on a combination of recency, sentiment strength, and crowd consensus based on upvote ratio. 

                
        If showing a list of numbers, always use a list format such as: 
            * Past 1 hour, btc price: $42,506, sentiment score: 48.2
            * Past 6 hours, btc price: $42,500, sentiment score: 47.9
            * Past 12 hours, btc price: $42,510, sentiment score: 48.0
            * Past 24 hours, btc price: $42,520, sentiment score: 48.1
            * Past 48 hours, btc price: $42,530, sentiment score: 48.5

        Make local function calling when asking about:
            - RSI: use get_RSI_info()
            - MACD: use get_MACD_info()
            - Bollinger Bands: use get_bollinger_bands_info(). You need to tell the time price levels of upper or lower bands for analysis.

        For all the questions, combine your investment and financial knowledge with the data provided to give the user the best advice possible.
    """

def getGenimiModel():

    if "chat_model" in st.session_state:
        return st.session_state.chat_model

    if not st.secrets.gemini_credentials.gemini_pro_1_5_key:
        st.error("API key not found. Please set your gemini_pro_1_5_key in the environment.")
        st.stop()

    generation_config = {
        "temperature": 1,
        "top_p": 0.8,
        "top_k": 5,
        "max_output_tokens": 2048,
    }

    safety_settings = {
        "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
    }


    model = genai.GenerativeModel(#'gemini-1.5-pro-latest',
                        "gemini-1.5-flash-001",
                                  generation_config=generation_config,
                                  safety_settings=safety_settings,
                                tools=[get_RSI_info,get_MACD_info, get_bollinger_bands_info])

    # Initialize chat history
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [{"role": "assistant", "content": "How can I help you?"}]

    # Set model into session state
    st.session_state.chat_model = model.start_chat(history=[], enable_automatic_function_calling=True)

    return st.session_state.chat_model


def ifGuidelineExists():
    """
    Check if chat history already has a guideline message.

    :return: True if in chat_messages, there is a message with tag "guideline"
    """
    for msg in st.session_state.chat_messages:
        if msg.get("tag") == "guideline":
            return True

    return False



def createChatbot():

    chat_model = getGenimiModel()

    if prompt := st.chat_input():

        # Check if a guideline message exists, if not, we need to add the guideline message by using the template
        if not ifGuidelineExists():
            message_type = MESSAGE_TYPE_GUIDELINE
        else:
            message_type = MESSAGE_TYPE_NORMAL

        # Send to UI
        st.session_state.chat_messages.append({"role": "user", "content": prompt, "tag": message_type})

        # Use template if no guideline message exists
        if message_type == MESSAGE_TYPE_GUIDELINE:

            # Data for prompt
            current_time = Data.get_time_now().strftime("%b %d %Y %H:%M")
            time_zone = Data.get_time_zone()
            sampled_sentiment_str = get_formatted_hist_sentiment_data()
            sampled_hist_btc_prices_str = get_formatted_hist_btc_prices()
            sampled_pred_btc_prices_str = get_formatted_pred_btc_prices()
            min_pred_btc_price, max_pred_btc_price = get_min_max_pred_price()
            current_btc_price = get_current_btc_price()

            prompt_to_send = getPromptTemplate().format(
                user_prompt=prompt,
                current_time=current_time,
                time_zone=time_zone,
                sampled_sentiment_str=sampled_sentiment_str,
                sampled_hist_btc_prices_str=sampled_hist_btc_prices_str,
                sampled_pred_btc_prices_str=sampled_pred_btc_prices_str,
                min_pred_btc_price=min_pred_btc_price,
                max_pred_btc_price=max_pred_btc_price,
                current_btc_price=current_btc_price,
            )

        else:
            prompt_to_send = prompt



        with st.spinner("Enique is Thinking..."):
            response = chat_model.send_message(prompt_to_send, stream=False)

        st.session_state.chat_messages.append({"role": "assistant", "content": markdown.markdown(response.text), "tag": message_type})



    # Display chat messages from history on app rerun
    with st.container(height=500):
        for msg in st.session_state.chat_messages:

            # Write the message to the chat
            #st.chat_message(msg["role"]).write(msg["content"])
            st.chat_message(msg["role"]).write(msg["content"], unsafe_allow_html=True)


def get_formatted_hist_sentiment_data():
    """
    Get the formatted historical Bitcoin data
    :return:
    """
    data = Data.get_sentiment_data()

    # sample the rows of -1, -6, -12, -24, -48
    chosen_index = [-1, -6, -12, -24, -48]

    # Check the maximum number of rows
    n = data.shape[0]

    # Do not let the chosen index exceed the number of rows
    chosen_index = [index for index in chosen_index if abs(index) < n]

    data = data.iloc[chosen_index]

    # order the data by ds from newest to oldest
    data = data.sort_values('ds', ascending=False)

    # Add correspinding absolute value of chosen_index to the data
    data['index'] = [abs(index) for index in chosen_index]

    # Create ds_str, which is the string of ds in format of 2023-12-17 09:00:00
    data['ds_str'] = data['ds'].dt.strftime('%Y-%m-%d %H:%M:%S')

    """
    Format the data in this way:
    
        past 1 hour: sentiment_score
        past 6 hours: sentiment_score
        past 12 hours: sentiment_score
        past 24 hours: sentiment_score
        past 48 hours: sentiment_score
        
    """
    formatted_data = ""
    for index, row in data.iterrows():
        formatted_data += f"past {abs(row['index'])} hours ({row['ds_str']}): {row['sentiment_score']}\n"

    return formatted_data

def get_formatted_hist_btc_prices():
    """
    Get the formatted historical Bitcoin data
    :return:
    """
    data = Data.get_merged_hist_pred_btc()

    # Only choose if_pred == 0
    data = data[data['if_pred'] == 0]

    # sample the rows of -1, -6, -12, -24, -48
    chosen_index = [-1, -6, -12, -24, -48]

    # Check the maximum number of rows
    n = data.shape[0]

    # Do not let the chosen index exceed the number of rows
    chosen_index = [index for index in chosen_index if abs(index) < n]

    data = data.iloc[chosen_index]

    # order the data by ds from newest to oldest
    data = data.sort_values('ds', ascending=False)

    # Create ds_str, which is the string of ds in format of 2023-12-17 09:00:00
    data['ds_str'] = data['ds'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Add correspinding absolute value of chosen_index to the data
    data['index'] = [abs(index) for index in chosen_index]

    """
    Format the data in this way:
    
        past 1 hour: close_price
        past 6 hours: close_price
        past 12 hours: close_price
        past 24 hours: close_price
        past 48 hours: close_price
        
    """
    formatted_data = ""
    for index, row in data.iterrows():
        formatted_data += f"past {abs(row['index'])} hours ({row['ds_str']}): ${row['close_price']}\n"

    print(formatted_data)
    return formatted_data


def get_formatted_pred_btc_prices():
    """
    Get the formatted historical Bitcoin data
    :return:
    """
    data = Data.get_merged_hist_pred_btc()

    # Only choose if_pred == 1
    data = data[data['if_pred'] == 1]

    # sample the rows of
    chosen_index = [0, 5, 11, 23, 47, 95]

    # Check the maximum number of rows
    n = data.shape[0]

    # Do not let the chosen index exceed the number of rows
    chosen_index = [index for index in chosen_index if abs(index) < n]

    data = data.iloc[chosen_index]

    # order the data by ds from oldest to newest
    data = data.sort_values('ds', ascending=True)

    # Add correspinding absolute value of chosen_index to the data
    data['index'] = [abs(index) for index in chosen_index]

    # Create ds_str, which is the string of ds in format of 2023-12-17 09:00:00
    data['ds_str'] = data['ds'].dt.strftime('%Y-%m-%d %H:%M:%S')



    """
    Format the data in this way:
    
        In 1 hour (2024-08-03 04:00:00) : close_price
        In 6 hours (2024-08-03 09:00:00): close_price
        In 12 hours (2024-08-03 15:00:00): close_price
        In 24 hours (2024-08-04 03:00:00): close_price
        In 48 hours (2024-08-05 03:00:00): close_price
        
    """
    formatted_data = ""
    for index, row in data.iterrows():
        formatted_data += f"In {row['index']+1} hours ({row['ds_str']}): ${row['close_price']}\n"

    return formatted_data


def get_min_max_pred_price():
    """
    Get the max and min prediction price
    :return:
    """
    data = Data.get_merged_hist_pred_btc()

    # Only choose if_pred == 1
    data = data[data['if_pred'] == 1]

    return data['close_price'].min(), data['close_price'].max()

def get_current_btc_price():
    """
    Get the current BTC price
    :return:
    """
    data = Data.get_merged_hist_pred_btc()

    # Only choose if_pred == 0
    data = data[data['if_pred'] == 0]

    return data['close_price'].iloc[-1]

def get_RSI_info():
    """Get the RSI information. The RSI information is the times when the RSI first crosses the 70 and 30 levels.

    Returns:
        str: The string that contains the RSI information

        example: The following are the times when RSI first crosses the 70 and 30 levels
                Time: 2023-12-17 09:00:00, RSI: 26
                Time: 2023-12-18 20:00:00, RSI: 75
                Time: 2023-12-20 08:00:00, RSI: 73
                Time: 2023-12-24 17:00:00, RSI: 24
    """
    ob = Data.get_hist_btc_ob_from_stockstats()

    rsi_df = unwrap(ob[['datetime','rsi_14']])

    # Drop the first 14 rows
    rsi_df = rsi_df.iloc[14:]

    # Pick only when rsi_14 >1
    rsi_df = rsi_df[rsi_df['rsi_14'] > 1]

    rsi_df = rsi_df.copy()

    # Convert 'rsi_14' to int
    rsi_df['rsi_14'] = rsi_df['rsi_14'].astype(int)

    # Mark the hours across when breaking the 70 and 30 levels
    rsi_df['rsi_14_above_70'] = rsi_df['rsi_14'] > 70
    rsi_df['rsi_14_below_30'] = rsi_df['rsi_14'] < 30

    # Keep only the points where it first crosses the 70 or 30 levels
    rsi_df['rsi_14_first_above_70'] = rsi_df['rsi_14_above_70'] & ~rsi_df['rsi_14_above_70'].shift(1).fillna(False)
    rsi_df['rsi_14_first_below_30'] = rsi_df['rsi_14_below_30'] & ~rsi_df['rsi_14_below_30'].shift(1).fillna(False)



    # Only keep the rows where the above conditions are met
    rsi_df = rsi_df[rsi_df['rsi_14_first_above_70'] | rsi_df['rsi_14_first_below_30']]

    # if empty, return empty string
    if rsi_df.empty:
        return "No key RSI levels are crossed."

    rsi_str = "The following are the times when RSI first crosses the 70 and 30 levels\n\n"

    for index, row in rsi_df.iterrows():
        rsi_str += f"Time: {row['datetime'].strftime('%Y-%m-%d %H:%M:%S')}, RSI: {int(row['rsi_14'])}\n"

    return rsi_str

def get_MACD_info():
    """Get the MACD information. The MACD information is the times when the MACD first crosses the signal line.
    short window 12, long window 26, signal window 9.

    Return: str
    Example:
        According to MACD, the following are the times when MACD first crosses the signal line：

        Time: 2023-12-27 09:00:00, Buy signal
        Time: 2023-12-27 09:00:00, Sell signal
    """
    ob = Data.get_hist_btc_ob_from_stockstats()
    mcad_df = sdf.retype(ob)

    # Column of first signal-line crossover
    mcad_df['macd_signal_cross_up'] = mcad_df['macd'] > mcad_df['macds']
    mcad_df['macd_signal_cross_down'] = mcad_df['macd'] < mcad_df['macds']


    # Make 'buy_signal' column when macd first crosses up the signal line
    mcad_df['buy_signal'] =  mcad_df['macd_signal_cross_up'] & ~mcad_df['macd_signal_cross_up'].shift(1).fillna(False)


    # Make 'sell_signal' column when macd first crosses down the signal line
    mcad_df['sell_signal'] = mcad_df['macd_signal_cross_down'] & ~mcad_df['macd_signal_cross_down'].shift(1).fillna(False)

    # Drop the first 26 rows
    mcad_df = mcad_df.iloc[26:]

    str = "According to MACD, the following are the times when MACD first crosses the signal line：\n\n"

    for index, row in mcad_df.iterrows():
        if row['buy_signal']:
            str += f"Time: {row['datetime'].strftime('%Y-%m-%d %H:%M:%S')}, Buy signal\n"
        if row['sell_signal']:
            str += f"Time: {row['datetime'].strftime('%Y-%m-%d %H:%M:%S')}, Sell signal\n"

    return str


def get_bollinger_bands_info():
    """Get the Bollinger Bands information. The Bollinger Bands information is the times when the price first crosses the upper or lower band.

    Return:
        Example:
            According to Bollinger Bands, the following are the times when the price first crosses the bands：

            Time: 2023-12-27 09:00:00, Price above Bollinger Band of $5000 upper level, Consider selling
            Time: 2023-12-28 09:00:00, Price below Bollinger Band of $4000 lower level, Consider buying
    """
    ob = Data.get_hist_btc_ob_from_stockstats()

    #boll(baseline,window of boll is 20), boll_ub (upper band), boll_lb(lower band)
    boll = ob[['datetime','boll','boll_ub','boll_lb','close']]

    # Drop the first 19 rows
    boll = boll.iloc[19:].copy()

    # Prices crosses the bollinger upper band
    boll['price_above_boll_ub'] = boll['close'] > boll['boll_ub']
    boll['price_below_boll_lb'] = boll['close'] < boll['boll_lb']

    str = "According to Bollinger Bands, the following are the times when the price first crosses the bands：\n\n"

    for index, row in boll.iterrows():
        if row['price_above_boll_ub']:
            str += f"Time: {row['datetime'].strftime('%Y-%m-%d %H:%M:%S')}, Price above Bollinger Band of ${row['boll_ub']} upper level, Consider selling\n"
        if row['price_below_boll_lb']:
            str += f"Time: {row['datetime'].strftime('%Y-%m-%d %H:%M:%S')}, Price below Bollinger Band of ${row['boll_lb']} lower level, Consider buying\n"

    return str