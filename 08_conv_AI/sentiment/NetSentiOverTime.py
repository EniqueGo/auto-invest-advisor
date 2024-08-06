import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from service import Data



# Function to create the gauge chart
def renderChart():

    # Sample data
    df = Data.get_sentiment_data()

    # Separate negative(0-33), neutral(34-66) and positive(67-100) scores
    df['negative'] = df['sentiment_score'].apply(lambda x: x if x < 34 else None)
    df['neutral'] = df['sentiment_score'].apply(lambda x: x if x >= 34 and x < 67 else None)
    df['positive'] = df['sentiment_score'].apply(lambda x: x if x >= 67 else None)

    # Get the highest sentiment score
    max_score = df['sentiment_score'].max()
    min_score = df['sentiment_score'].min()


    # Create the figure
    fig = go.Figure()

    # Add positive sentiment
    fig.add_trace(go.Scatter(
        x=df['ds'], y=df['negative'],
        mode='lines', line=dict(color="#ff5395"),
        #fill='tozeroy',
        name='Negative Sentiment'
    ))

    fig.add_trace(go.Scatter(
        x=df['ds'], y=df['neutral'],
        mode='lines', line=dict(color="#ffb200"),
        #fill='tozeroy',
        name='Neutral Sentiment'
    ))

    fig.add_trace(go.Scatter(
        x=df['ds'], y=df['positive'],
        mode='lines', line=dict(color="#2ec4ab"),
        #fill='tozeroy',
        name='Neutral Sentiment'
    ))

    # Update layout
    fig.update_layout(
        title='',
        xaxis_title=None,
        yaxis_title=None,
        yaxis=dict(range=[0, 100],autorange=True, fixedrange=False),
        showlegend=False,
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=110,
        margin=dict(l=50, r=20, t=0, b=2)
    )


    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig, use_container_width=True)
