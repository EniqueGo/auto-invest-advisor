# utils/viz.py

"""
@author: Esther Xu
Code tested on:
    - os: Debian 11
    - Python: 3.10.14; 3.11.2
    - Numpy: 1.25.2
    - Sklearn: 1.3.0
"""

import matplotlib.pyplot as plt
import seaborn as sns

def plot_forecast_vs_actual(actual_df,
                            forecast_df,
                            title='',
                            xlabel='Date',
                            ylabel='Close Price',show_ci = False):
    """
    Plot predicted versus actual Bitcoin prices, including confidence intervals.

    Parameters:
    - forecast (DataFrame): The forecast DataFrame containing the dates ('ds'), predicted values ('yhat'), and confidence intervals ('yhat_lower' and 'yhat_upper').
    - actual (DataFrame): The actual Bitcoin data containing the dates, closing prices ('y'), and optionally high and low values ('y_upper', 'y_lower').
    """

    fig, ax = plt.subplots(figsize=(20, 10))

    # Plotting the predicted values
    sns.lineplot(x='ds', y='yhat', data=forecast_df, label='Predicted', color='deepskyblue', alpha=0.6, linewidth=2)

    if show_ci == True:
        # Plotting the confidence intervals for predictions
        ax.fill_between(forecast_df['ds'], forecast_df['yhat_lower'], forecast_df['yhat_upper'], color='deepskyblue', alpha=0.3, label='Predicted CI')

    # Plotting the actual values
    sns.lineplot(x='ds', y='y', data=actual_df, label='Actual', color='coral', alpha=0.7, linewidth=2)

    # Customize the x-axis
    ax.xaxis.set_major_locator(plt.MaxNLocator(24))  # Limit the number of x-ticks to improve plot readability
    plt.xticks(rotation=45)  # Rotate x-tick labels
    ax.grid(True)  # Enable grid

    # Labels and title
    plt.xlabel(xlabel if xlabel else 'Date', fontsize=14)
    plt.ylabel(ylabel if ylabel else 'Bitcoin Price (USD)', fontsize=14)
    plt.title(title if title else f"Bitcoin Forecast vs Actual Prices from {actual_df['ds'].min()} to {actual_df['ds'].max()}", fontsize=16)


def plot_hourly_residual_error(timestamps, y_true, y_pred, title='Residual Error Over Time'):
    """
    Plots the hourly residual error with timestamps on the x-axis and residual error percentage on the y-axis.

    Parameters:
    - timestamps (pd.Index or np.array): Timestamps for the x-axis.
    - y_true (np.array): True values of the target variable.
    - y_pred (np.array): Predicted values of the target variable.
    - title (str): Title of the plot.
    """
    # Calculate residual error in percentage terms
    residual_error_pct = ((y_true - y_pred) / y_true) * 100

    # Plot the residual error
    plt.figure(figsize=(14, 7))
    plt.plot(timestamps, residual_error_pct, label='Residual Error (%)', color='red')
    plt.xlabel('Timestamp')
    plt.ylabel('Residual Error (%)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_train_validation_loss_values(history):
    # Plot training & validation loss values
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.show()

