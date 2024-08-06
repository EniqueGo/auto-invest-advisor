import streamlit as st
import plotly.graph_objects as go
from service import Data

def renderChart():
    # data columns: ds, close_price, if_pred
    data = Data.get_merged_hist_pred_btc()

    # past data
    past_data = data[data['if_pred']==0]

    past_time = past_data['ds']
    past_price = past_data['close_price']

    # Forcast data
    forecast_data = data[data['if_pred']==1]

    # forecast time is the last record of forecast_data
    forecast_time_max = forecast_data['ds'].iloc[-1]


    forecast_avg = forecast_data['close_price']
    forecast_low = forecast_data['close_price'].min()
    forecast_high = forecast_data['close_price'].max()

    """
    # Sample data
    past_time = ['2021', '2022', '2023']
    past_price = [100, 150, 145.74]  # Adjust this with actual values
    forecast_time = ['2023', '2024']
    forecast_high = [145.74, 200]
    forecast_avg = [145.74, 175.69]
    forecast_low = [145.74, 122]
    """
    # Create the figure
    fig = go.Figure()

    # Draw the lines

    # Past Data
    fig.add_trace(go.Scatter(x=past_time, y=past_price,
                             mode='lines', name='Past Data', line=dict(color='#3b5fc6')))

    # Forecasted close_price
    fig.add_trace(go.Scatter(x=forecast_data['ds'], y=forecast_data['close_price'],
                             mode='lines', name='Forecasted Data', line=dict(dash='dot', color='#3b5fc6')))


    fig.add_trace(go.Scatter(x=[past_data.iloc[-1]['ds'],forecast_time_max], y=[past_data.iloc[-1]['close_price'], forecast_high],
                             mode='lines', name='Forecasted Avg', line=dict(dash='dot', color='#2daa98')))
    """
    fig.add_trace(go.Scatter(x=[past_data.iloc[-1]['ds'],forecast_time], y=[past_data.iloc[-1]['close_price'], forecast_avg],
                             mode='lines', name='Forecasted Avg', line=dict(dash='dot', color='#3b5fc6')))
    """
    fig.add_trace(go.Scatter(x=[past_data.iloc[-1]['ds'],forecast_time_max], y=[past_data.iloc[-1]['close_price'], forecast_low],
                             mode='lines', name='Forecasted Low', line=dict(dash='dot', color='#ec1a66')))


    # Add Point Markers

    fig.add_trace(go.Scatter(x=[past_data.iloc[-1]['ds']], y=[past_data.iloc[-1]['close_price']], mode='markers+text',
                             name='Current', text=[past_data.iloc[-1]['close_price']], textposition='top right'))

    """
    fig.add_trace(go.Scatter(x=[forecast_time], y=[forecast_avg], mode='markers+text',
                             name='Forecasted Avg', text=[forecast_avg], textposition='top right',
                             textfont_color='#3b5fc6'))
    """

    fig.add_trace(go.Scatter(x=[forecast_time_max], y=[forecast_high], mode='markers+text',
                             name='Forecasted High', text=[forecast_high], textposition='top right',
                             textfont_color='#2daa98'))

    fig.add_trace(go.Scatter(x=[forecast_time_max], y=[forecast_low], mode='markers+text',
                            name='Forecasted Low', text=[forecast_low], textposition='top right',
                            textfont_color='#ec1a66'))


    # Update layout
    fig.update_layout(
        title='',
        xaxis_title=None,
        yaxis_title=None,
        yaxis={'side': 'right'},
        margin=dict(l=20, r=20, t=2, b=2),
        autosize=True,
        width=None,
        height=200,
        showlegend=False
    )

    # Render the plotly figure in Streamlit
    st.plotly_chart(fig, use_container_width=True)