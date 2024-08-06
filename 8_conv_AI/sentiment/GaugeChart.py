import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from service import Data

def renderChart():

    sentiment_over_time = Data.get_sentiment_data()
    sentiment_score = sentiment_over_time['sentiment_score'].iloc[-1]


    # Use different color for scores
    if sentiment_score < 33:
        bar_color = "#ff5395"
        image_path = "img/senti-negative.png"
    elif sentiment_score < 67:
        bar_color = "#ffb200"
        image_path = "img/senti-neutral.png"
    else:
        bar_color = "#2ec4ab"
        image_path = "img/senti-positive.png"


    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=sentiment_score,
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': bar_color},
            'steps': [
                {'range': [0, 33], 'color': "white"},
                {'range': [34, 66], 'color': "white"},
                {'range': [67, 100], 'color': "white"}],
            'threshold': {
           #     'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': sentiment_score}}))


    # Add a img/senti-positive.png
    fig.add_layout_image(
        dict(
            source=Image.open(image_path, mode='r'),  # Replace with your image URL
            xref="paper",
            yref="paper",
            x=0.95,  # Position of the image
            y=0.95,  # Position of the image
            sizex=0.3,
            sizey=0.3,
            xanchor="right",
            yanchor="top"
        )
    )
   
    fig.update_layout(
        margin=dict(l=20, r=20, t=10, b=5),
        height=140,
        width=None,
        paper_bgcolor="white",
    )

    st.plotly_chart(fig, use_container_width=True)



