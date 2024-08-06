import streamlit as st
import pandas as pd
from string import Template
from service import Data


def renderContent():

    df = Data.get_rep_posts()


    positive_text = df['positive']['clean_text']
    neutral_text = df['neutral']['clean_text']
    negative_text = df['negative']['clean_text']

    positive_url = df['positive']['url']
    neutral_url = df['neutral']['url']
    negative_url = df['negative']['url']

    positive_time = df['positive']['created']
    neutral_time = df['neutral']['created']
    negative_time = df['negative']['created']


    template_html = """
    <div class="sentiment-container">
        <div class="sentiment positive">
            <div class="senti-label">Positive</div>
            <div class="dashed-line"></div>
            <div class="senti-text">$positive_text
            <a class="senti-link" href="$positive_url" target="_blank">[More]</a>
            <div class="senti-time">- $positive_time</div>
            </div>
        </div>
        <div class="sentiment neutral">
            <div class="senti-label">Neutral</div>
            <div class="dashed-line"></div>
            <div class="senti-text">$neutral_text
            <a class="senti-link"  href="$neutral_url" target="_blank">[More]</a>
            <div class="senti-time">- $neutral_time</div>
            </div>
        </div>
        <div class="sentiment negative">
            <div class="senti-label">Negative</div>
            <div class="dashed-line"></div>
            <div class="senti-text">$negative_text
            <a class="senti-link" href="$negative_url" target="_blank">[More]</a>
            <div class="senti-time">- $negative_time</div
            </div>
        </div>
    </div>
    """
    template = Template(template_html)
    filled_html = template.substitute(

        positive_text=positive_text,
        neutral_text=neutral_text,
        negative_text=negative_text,

        positive_url=positive_url,
        neutral_url=neutral_url,
        negative_url=negative_url,
        positive_time=positive_time,
        neutral_time=neutral_time,
        negative_time=negative_time
    )

    st.markdown(filled_html, unsafe_allow_html=True)