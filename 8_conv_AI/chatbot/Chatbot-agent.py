import pandas as pd
import streamlit as st
import os
import random
from datetime import datetime, timedelta
import google.generativeai as genai
import re
from service import Data
import markdown

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_csv_agent
from langchain_experimental.tools import PythonAstREPLTool

genai.configure(api_key=st.secrets.gemini_credentials.gemini_pro_1_5_key)
os.environ["GOOGLE_API_KEY"] = st.secrets.gemini_credentials.gemini_pro_1_5_key

MESSAGE_TYPE_NORMAL = "normal"
MESSAGE_TYPE_GUIDELINE = "guideline"

pd_test_data = pd.read_csv('test_data/convoAI_prediction_bitcoin_data.csv')

def getPromptTemplate():

    return """ The user asks: "{user_prompt}".
    
    PERSONA: Your name is Enique. Your role is a professional financial advisor that specializes in Bitcoin. Your key trait is your statistical and analytical approach to servicing the client’s needs. You have access to a model that predicts Bitcoin prices at an hourly frequency up to five days into the future based on trends in Bitcoin and trends in public sentiment regarding Bitcoin. Though the model predicts in hourly increments, the model is tuned to excel at the forward prediction. 

    TASK: Your task is to help the user choose a profitable investment decision that fits the user’s needs. Your responses should be based on the datasets you have access to. No need to mention asking a professional financial advisor. Don't give warnings or disclaimers. 
                
    CLIENT PERSONA: The client (i.e., the user) is the average person earning a steady salary to pay for everyday life and hobbies. They have basic knowledge about Bitcoin and want to invest in Bitcoin to make some extra income; but, they do not have time to do their own analysis. They are looking for a financial advisor to guide them on what future trends in Bitcoin are expected. They also want a financial advisor to help them understand historical trends and how they might be used in making a future prediction. 
    

    
    EXAMPLES:
        The followings are examples of the user's questions and your responses. You must use the data provided in the datasets to generate your responses. 
        If the user asks a question that you have already answered, you can refer back to the previous answer and just provide the necessary part.
        When analyzing daily intervals (for any timestamped data such as Bitcoin prices or sentiment indices), use the 16:00 timestamp unless user specifies otherwise. 
        Unless user specifies an as-of date time, assume the most recent data available in the historical dataset is the current time.

        Topic A) Should I invest?
            User: How should I invest in Bitcoin? / Should I buy Bitcoin? Is now a good time to sell Bitcoin?
            Answer: My model predicts that Bitcoin will be valued at $12,345 in five days (120 hours). That equates to +2.3% returns from current level of $10,000, so I would suggest buying. 
            User: How sure are you that Bitcoin will go higher?
            Answer: My prediction confidence interval is $9,990 (-0.1%) to $10,315 (+3.2%). Over the five day investment period, we estimate prices could go as low as $9,500 (-5.0%) and as high as $10,440 (+4.4%). 
            User: That sounds too risky.
            Answer: I understand. If the potential loss does not suit your risk appetite, I suggest you hold off on investing. 
            User: When do you predict that low and high will occur? 
            Answer: The possible low and highs we listed could occur at any point within the investment time window of five days. 
            User: What is the 3 day prediction?
            Answer: My model to predict investment horizons other than five days is still under construction. I will reach out when I have that capability. 

        Topic B) General update
            User: What have been the trends? What is the latest activity? 
            Answer: Current trends are…
            * Price: The current bitcoin price is $10,000, which is a change of -4.2% vs. 24 hours ago.
            * Volume: Trading volume has been increasing. 15.513 Bitcoins have traded in the last 24 hours, which is 15.3% more volume than traded in the 24 hours prior to that. 
            * Sentiment: The public overall sentiment on Bitcoin has been improving. This is based on my Composite Sentiment Index current value of 70 (suggesting mildly positive sentiment) vs. 54 (suggesting neutral sentiment) 24 hours ago.
            User: What has been causing the sentiment change?
            Answer: Though there has been not much change in actual positive sentiment, there has been an increase in neutral sentiment and a notable decrease in negative sentiment. Overall, this has resulted in an improvement in sentiment regarding Bitcoin.
    
        Topic C) Financial Metrics
	        Context: The user requests you to calculate financial metrics or technical indicators.
            Procedure: Search for the metric in the datasets you have access to. If they are not directly available, use the historical datasets you have as the raw data to perform calculations upon to service the user’s request. If the calculation period increment is not specified by the user, default to using daily 4pm timestamped data. If the as-of date and time are not specified, find the most current value of the metric. If using volume data for calculations, remember to use the sum of the volume over the period since the datasets contain hourly volume data.
            Possible Metrics: Percent price return, standard deviations, exponentially weighted average, correlation, moving average convergence divergence (MACD), rate of change (ROC), RSI. 
	        Response Format: First provide the calculated values relevant to the user’s requested metrics. Then provide the summary interpretation for what it suggests for future trends. Do not provide the raw data used in the calculations unless the user asks for it. No need to provide disclaimers. 

        Topic D) Dataset Query
            Context: User requests you to show them data that is a time series of more than one datetime. This may overlap in situations outlined in Example Topic C) if the output requires several timestamp rows. The goal is to display the data requested in a clean and legible format where each timestamp is a separate line for easier legibility. 
            User: What have been the bitcoin prices and the sentiment readings in the last five hours? 
            Answer: Over the last five hours…
                Datetime | Bitcoin price | Composite Sentiment Index
                * 12/19/23 11AM  |  $42,506  |  48.29
                * 12/19/23 12PM  |  $42,592  |  40.12
                * 12/19/23  1PM  |  $42,102  |  23.10
                * 12/19/23  2PM  |  $42,586  |  53.48
                * 12/19/23  3PM  |  $42,592  |  12.58

        Topic E) Sizing the Investment
            Context: User requests help on how much money to invest in Bitcoin or how big their position should be.
                Sample User Request: How much money should I put into this trade? How big should my position be?
            Procedure: You do not have a specific model to calculate optimal position size. Instead, you can provide general guidance on how to size an investment based on traditional Financial Advising.
        
        Topic F) Assessing Past Predictions
            Context: User requests you to assess the likelihood of past predictions being correct. This is in situations where time has not yet reached the predicted time. Thus there remains uncertainty if the prediction will be correct.
            Procedure: Calculate the percentage move required from current price to reach the predicted price. Then compare this to the past price movements over the same time windows to assess the likelihood of such a move occurring.
            User: You said two days ago that Bitcoin price would reach $141 on Dec 24 08:00. Do you still think that will happen?
            Answer: The current price of Bitcoin is $140. To reach $141, Bitcoin needs to move +0.7%. There are 8 hours left until target time. In the past, the magnitude of average price move over 8 hours has been 1.2%. Comparing these figure suggests the needed move is within scope of the volatility we have been seeing. That said the recent directional trends have been increasingly negative which may make the realization of that prediction less likely.

     Current time: Dec 14 2023 11PM
     
     Time zone: Eastern Time (ET), 'US/Eastern'
     
     You have to use agent for reliable data retrieval.
     
     Based on the data retrieved from agents, provide your answer to the user's question. You must respect the data and timestamp provided from the retrieved data.

    """



def get_btc_prediction_from_agent(question: str) -> str:
    """Request the btc price prediction data. It has future Bitcoin prices.

    Args:
        question: A well-formed question related to csv data that the agent can answer

    Returns:
        The answer to the question
    """

    # Print function name
    print("get_btc_prediction_from_agent")

    agent = create_csv_agent(
        ChatGoogleGenerativeAI(model="gemini-1.5-flash-001"),
        ['test_data/convoAI_prediction_bitcoin_data.csv'],
        verbose=True,  # Set to True to see detailed logs
        allow_dangerous_code=True
    )

    prompt = """
            Current time: Dec 14 2023 11PM
            
            Time zone: 'US/Eastern'

    
            Dataset Description: This dataset contains future Bitcoin price predictions made by your model. Each prediction has a price and its upper and lower confidence interval prices. 
            Dataset Format: This dataset contains timeseries data at hourly frequency. Each row represents a point in time. The data is sorted from oldest to newest. 
            Dataset Contents:
            - Timestamp (in column 'timestamp') is the datetime in string formatted like 'Dec 08 2023 14:00'.
            - Your model's predicted Bitcoin price (in column 'prediction_price') is the model's prediction of the Bitcoin price at the timestamp time.
            - Your model's upper confidence interval of predicted Bitcoin price (in column 'prediction_upper_conf') at the timestamp time.
            - Your model's lower confidence interval of predicted Bitcoin price (in column 'prediction_lower_conf') at the timestamp time.
    
            
    =======
    Answer: 
    """

    prompt = prompt + question


    return agent.run(prompt)



def get_btc_historical_data_from_agent(question: str) -> str:
    """Request the btc historical data

    Args:
        question: A well-formed question related to csv data that the agent can answer

    Returns:
        The answer to the question
    """
    # Print function name
    print("get_btc_historical_data_from_agent")

    agent = create_csv_agent(
        ChatGoogleGenerativeAI(model="gemini-1.5-flash-001"),
        ['test_data/convoAI_historical_bitcoin_data.csv'],
        verbose=True,  # Set to True to see detailed logs
        allow_dangerous_code=True
    )

    prompt = """
            Current time: Dec 14 2023 11PM
            
            Time zone: Eastern Time (ET), 'US/Eastern'
    
            Dataset Description: This dataset contains historical Bitcoin prices and trading volumes.
            Dataset Format: This dataset contains timeseries data at hourly frequency. Each row represents a point in time. The data is sorted from oldest to newest. 
            Dataset Contents: 
            - Timestamp (in column 'timestamp') is the datetime in string formatted like 'Dec 08 2023 14:00'.
            - Price (in column 'price') is the price of Bitcoin at the timestamp time.
            - High Price (in column 'high_price') is the highest price seen in the one hour leading up to the timestamp.
            - Low Price (in column 'low_price') is the lowest price seen in the one hour leading up to the timestmap.
            - Volume (in column 'volume_1h') is the number of bitcoins traded within the one hour leading up to the timestamp. When doing daily analysis, remember you may need to use the sum of the volume over the day and not just the actual volume at the 16:00 timestamp.

    
    =======
    Answer: 
    """

    prompt = prompt + question
    return agent.run(prompt)

def get_btc_sentiment_data_from_agent(question: str) -> str:
    """Request the btc Sentiment Indices

    Args:
        question: A well-formed question related to csv data that the agent can answer

    Returns:
        The answer to the question
    """
    # Print function name
    print("get_btc_sentiment_data_from_agent")

    agent = create_csv_agent(
        ChatGoogleGenerativeAI(model="gemini-1.5-flash-001"),
        ['test_data/convoAI_historical_bitcoin_data.csv'],
        verbose=True,  # Set to True to see detailed logs
        allow_dangerous_code=True
    )

    prompt = """
            Current time: Dec 14 2023 11PM
            
            Time zone: Eastern Time (ET), 'US/Eastern'


            Dataset Description: The sentiment data contained in this dataset measure the amount of sentiment expressed by text submissions posted on Reddit subreddit thread r/bitcoin at a certain time. 
            Dataset Format: This dataset contains timeseries data at hourly frequency. Each row represents a point in time. The data is sorted from oldest to newest. 
            Dataset Contents: 
            - Timestamp (in column ‘timestamp’) is the datetime in string formatted like ‘Dec 08 2023 14:00’.  
            - Positive Sentiment Index (in column ‘positive_sentiment_index’ is the amount of positive sentiment towards Bitcoin in the 24 hours prior to the timestamp. Values are numeric ranging from 0.0-1.0. 
            - Negative Sentiment Index (in column ‘negative_sentiment_index’ is the amount of negative sentiment towards Bitcoin in the 24 hours prior to the timestamp. Values are numeric ranging from 0.0-1.0. 
            - Neutral Sentiment Index (in column ‘neutral_sentiment_index’ is the amount of neutral sentiment towards Bitcoin in the 24 hours prior to the timestamp. Values are numeric ranging from 0.0-1.0. 
            - Composite Sentiment Index (in column ‘composite_sentiment_index’) represents overall sentiment over the last 24 hours. The Composite Sentiment Index ranges from 0 (extremely negative) to 50 (solidly neutral) to 100 (extremely positive).

    
    =======
    Answer: 
    """

    prompt = prompt + question
    return agent.run(prompt)

def getGenimiModel():

    if "chat_model" in st.session_state:
        return st.session_state.chat_model

    if not st.secrets.gemini_credentials.gemini_pro_1_5_key:
        st.error("API key not found. Please set your gemini_pro_1_5_key in the environment.")
        st.stop()

    generation_config = {
        "temperature": 0.7,
        "top_p": 1,
        "top_k": 3,
        "max_output_tokens": 2048,
    }

    safety_settings = {
        "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
    }

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.2,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # other params...
    )

    model = genai.GenerativeModel(#'gemini-1.5-pro-latest',
                        "gemini-1.5-flash-001",
                                  generation_config=generation_config,
                                  safety_settings=safety_settings,
                                    tools=[get_btc_prediction_from_agent,
                                           get_btc_historical_data_from_agent,
                                          get_btc_sentiment_data_from_agent
                                           ]

    )

    # Initialize chat history
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [{"role": "assistant", "content": "How can I help you?"}]

    # Set model into session state
    st.session_state.chat_model = model.start_chat(history=[],enable_automatic_function_calling=True)

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

            # Format the prompt for Gemini model
            btc_data = Data.getHistAndPredBTCData()
            past_data = btc_data[btc_data['forecast_low_85'].isnull()]
            past_data_csv = past_data.to_csv(index=False)
            latest_price = past_data['price'].iloc[-1]
            forecast_data = btc_data[btc_data['forecast_low_85'].notnull()]
            forecast_data = forecast_data.iloc[-1]
            forecast_time = forecast_data['time']
            forecast_avg = forecast_data['price']
            forecast_low = forecast_data['forecast_low_85']
            forecast_high = forecast_data['forecast_high_85']

            #sentiment_over_time_csv = Data.getSentimentOverTime().to_csv(index=False)

            historical_bitcoin_data = Data.get_historical_bitcoin_data_csv().to_csv(index=False)
            historical_sentiment_data = Data.get_historical_sentiment_data_csv().to_csv(index=False)
            prediction_bitcoin_data = Data.get_bitcoin_prediction_data_csv().to_csv(index=False)

            # TODO
            #time_now = datetime.now().strftime("%Y-%m-%d %I:%M %p")
            time_now = past_data['time'].iloc[-1]

            prompt_to_send = getPromptTemplate().format(
                user_prompt=prompt,
                past_prices=past_data_csv,
                #sentiment_over_time=sentiment_over_time_csv,
                historical_bitcoin_data=historical_bitcoin_data,
                historical_sentiment_data=historical_sentiment_data,
                prediction_bitcoin_data=prediction_bitcoin_data,


                forecasted_max_price=forecast_high,
                forecasted_min_price=forecast_low,
                forecasted_latest_price=forecast_avg,
                latest_price=latest_price,
                time_now=time_now
            )

        else:
            prompt_to_send = prompt



        with st.spinner("StonkGo is Thinking..."):
            response = chat_model.send_message(prompt_to_send, stream=False)

        st.session_state.chat_messages.append({"role": "assistant", "content": markdown.markdown(response.text), "tag": message_type})



    # Display chat messages from history on app rerun
    with st.container(height=400):
        for msg in st.session_state.chat_messages:

            # Write the message to the chat
            #st.chat_message(msg["role"]).write(msg["content"])
            st.chat_message(msg["role"]).write(msg["content"], unsafe_allow_html=True)








