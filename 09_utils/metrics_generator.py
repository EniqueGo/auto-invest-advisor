# utils/metrics_generator.py

"""
@author: Esther Xu
Code tested on:
    - os: Debian 11
    - Python: 3.10.14; 3.11.2
    - Numpy: 1.25.2
    - Sklearn: 1.3.0
"""
import pandas as pd
import numpy as np
from sklearn.metrics import (mean_squared_error,
                             mean_absolute_error,
                             mean_absolute_percentage_error,
                             r2_score,
                             accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score)


# Attempt to import configuration from config.py
try:
    from config import DEFAULT_REGRESSION_METRICS, DEFAULT_CLASSIFICATION_METRICS, DEFAULT_DECIMALS
except ImportError:
    # Set default values if config.py is not available
    DEFAULT_REGRESSION_METRICS = ['mape', 'mse', 'rmse', 'mae', 'r2']
    DEFAULT_CLASSIFICATION_METRICS = ['accuracy', 'precision', 'recall', 'f1_score']
    DEFAULT_DECIMALS = 4


class MetricsGenerator:
    def __init__(self, y_true, y_pred, is_regression_task, target_name="", metrics=None, decimals=DEFAULT_DECIMALS):
        """
        Initialize the MetricsGenerator with true and predicted values.

        Parameters:
        y_true (pd.Series or np.ndarray): The ground truth values.
        y_pred (pd.Series or np.ndarray): The predicted values.
        decimals (int): Number of decimal places for rounding metrics.
        target_name (str): Target Name.
        """
        self.y_true = np.array(y_true)  # Ensure y_true is a NumPy array
        self.y_pred = np.array(y_pred)  # Ensure y_pred is a NumPy array
        self.target_name = target_name
        self.decimals = decimals
        self.is_regression_task = is_regression_task
        if self.is_regression_task and metrics is None:
            self.metrics = DEFAULT_REGRESSION_METRICS
        elif not self.is_regression_task and metrics is None:
            self.metrics = DEFAULT_CLASSIFICATION_METRICS
        else:
            self.metrics = metrics

    def performance(self, data_point=None, customized_brief=""):
        if self.is_regression_task:
            return self.__regression_metrics(data_point, customized_brief)
        else:
            return self.__classification_metrics(customized_brief)

    def __regression_metrics(self, data_point=None, customized_brief=""):
        """
        Calculate and return regression metrics along with a summary.

        Parameters:
        data_point (int): Specific data point to calculate metrics for.
        customized_brief (str): Custom brief for the summary.

        Returns:
        df (pd.DataFrame): DataFrame containing the calculated metrics.
        summary (str): Summary of the calculated metrics.
        """
        df = pd.DataFrame(columns=self.metrics)
        summary = customized_brief + "\n" if customized_brief else f"The performance of target {self.target_name}:\n"

        y_true = self.y_true
        y_pred = self.y_pred

        if data_point is not None:
            if len(self.y_pred) < data_point:
                raise ValueError(f"The prediction DataFrame does not have at least {data_point} rows.")
            else:
                y_true = np.array([self.y_true[data_point-1]])  # Single-element array
                y_pred = np.array([self.y_pred[data_point-1]])  # Single-element array

                summary = customized_brief + "\n" if customized_brief else f"The performance at {data_point} point for {self.target_name}\n"

        for metric in self.metrics:
            if metric == 'mape':
                mape = mean_absolute_percentage_error(y_true, y_pred)
                df.loc[0, 'mape'] = mape
                summary += f'Mean Absolute Percentage Error (MAPE): {np.round(mape, self.decimals)}\n'

            if metric == 'mse':
                mse = mean_squared_error(y_true, y_pred)
                df.loc[0, 'mse'] = mse
                summary += f'Mean Squared Error (MSE): {np.round(mse, self.decimals)}\n'

            if metric == 'rmse':
                rmse = mean_squared_error(y_true, y_pred, squared=True)
                df.loc[0, 'rmse'] = rmse
                summary += f'Root Mean Squared Error (RMSE): {np.round(rmse, self.decimals)}\n'

            if metric == 'mae':
                mae = mean_absolute_error(y_true, y_pred)
                df.loc[0, 'mae'] = mae
                summary += f'Mean Absolute Error (MAE): {np.round(mae, self.decimals)}\n'

            if metric == 'r2':
                r2 = r2_score(y_true, y_pred)
                df.loc[0, 'r2'] = r2
                summary += f'R-squared: {np.round(r2, self.decimals)}\n'

        return df, summary

    def __calc_mape_per_datapoint(self):
        """
        Calculate MAPE for each data point and return as a list.
        """
        mapes = []
        for i in range(len(self.y_true)):
            y_true = np.array([self.y_true[i]])  # Single-element array
            y_pred = np.array([self.y_pred[i]])
            mape = mean_absolute_percentage_error(y_true, y_pred)
            mapes.append(mape)
        return mapes

    def calc_datapoint_mape_stats(self, datapoint, datapoint_name='hour'):
        """
        Calculate the average and standard deviation of MAPE over datapoint(e.g., 120 hours).
        Group the MAPE values by datapoint (hourly) buckets and compute statistics for each bucket.

        Returns:
        overall_avg (float): Overall average MAPE.
        overall_std (float): Overall standard deviation of MAPE.
        stats (pd.DataFrame): DataFrame with average and standard deviation of MAPE for each datapoint.
        """
        mapes = self.__calc_mape_per_datapoint()

        # Calculate overall statistics
        overall_avg = np.mean(mapes)
        overall_std = np.std(mapes)

        # Group MAPE values by datapoint (hourly) buckets and compute statistics
        stats_list = []
        for i in range(datapoint):
            mapes_for_each = mapes[i::datapoint]  # Select MAPE values for each data point
            avg_mape = np.mean(mapes_for_each)
            std_mape = np.std(mapes_for_each)
            stats_list.append({datapoint_name: i, 'avg_mape': avg_mape, 'std_mape': std_mape})

        stats = pd.DataFrame(stats_list)
        return overall_avg, overall_std, stats

    def __classification_metrics(self, customized_brief="", average='binary'):
        """
        Calculate and return classification metrics.

        Parameters:
        customized_brief (str): Custom brief for the summary.
        average (str): Type of averaging performed on the data ('binary', 'micro', 'macro', 'weighted').

        Returns:
        df (pd.DataFrame): DataFrame containing the calculated classification metrics.
        summary (str): Summary of the calculated metrics.
        """
        summary = customized_brief if customized_brief else f"The performance of target {self.target_name}:\n"
        df = pd.DataFrame(columns=self.metrics)

        for metric in self.metrics:
            if metric == 'accuracy':
                accuracy = accuracy_score(self.y_true, self.y_pred)
                df.loc[0, 'accuracy'] = accuracy
                summary += f"Accuracy: {np.round(accuracy, self.decimals)}\n"

            if metric == 'precision':
                precision = precision_score(self.y_true, self.y_pred, average=average)
                df.loc[0, 'precision'] = precision
                summary += f"Precision: {np.round(precision, self.decimals)}\n"

            if metric == 'recall':
                recall = recall_score(self.y_true, self.y_pred, average=average)
                df.loc[0, 'recall'] = recall
                summary += f"Recall: {np.round(recall, self.decimals)}\n"

            if metric == 'f1_score':
                f1 = f1_score(self.y_true, self.y_pred, average=average)
                df.loc[0, 'f1_score'] = f1
                summary += f"F1 Score: {np.round(f1, self.decimals)}\n"

        return df, summary





