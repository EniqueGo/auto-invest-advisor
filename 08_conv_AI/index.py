import os
import streamlit as st

from predictions import PriceChart
from sentiment import GaugeChart
from sentiment import NetSentiOverTime
from sentiment import RepresentativePosts
from chatbot import Chatbot
from service import Data

# Set the new timezone to New York time
os.environ['TZ'] = Data.get_time_zone()


st.set_page_config(
    page_title="Bitcoin Prediction v6.01 | Enique",
    page_icon="img/favicon.png",
    layout="wide"
)

# CSS
with open('style.css') as f:
    css = f.read()
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

left_col, right_col = st.columns([9,3])


url_param_dic = st.query_params.to_dict()

#check if url_param_dic has "slogan" key
if "slogan" in url_param_dic:
    if url_param_dic["slogan"] == "GimmeBTC":
        disable_chatbot = False
else:
    disable_chatbot = True




with right_col:
    # Add Sentiment Analysis
    st.header("Sentiment Analysis")
    GaugeChart.renderChart()

    st.header("Net Sentiment Over Time")
    NetSentiOverTime.renderChart()

    st.header("Representative Posts")
    RepresentativePosts.renderContent()



with left_col:

    st.header("Bitcoin Prediction")
    st.write(PriceChart.renderChart())

    # Add Chatbot
    if not disable_chatbot:
        Chatbot.createChatbot()
    else:
        notice_str = """
        Dear Guest, \n

        We have intentionally turned off the chatbox. Please join us again during our live session. \n
        
        Enique offers an exceptional feature for Bitcoin investment.\n
        
        Our team, Anna, Esther, and Mu, wish you a wonderful day. \n
        
        Thank you for visiting. See you soon!
        """
        st.write(notice_str)

#%%

#%%

#%%

#%%
