# models/baseline.py
"""
@author: Esther Xu
Code tested on:
    - os: Debian 11
    - Python: 3.10.14; 3.11.2
    - Numpy: 1.25.2
"""

import numpy as np
import itertools
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import add_changepoints_to_plot, plot_cross_validation_metric
from utils.metrics_generator import MetricsGenerator

try:
    from config import FORECAST_FREQUENCY, FORECAST_HOURS,VAR_DATE,VAR_TARGET,VAR_FORECAST
except ImportError:
    # Set default values if config.py is not available
    FORECAST_FREQUENCY = 'H'  # 'D'
    FORECAST_HOURS = 120

class ProphetModelBuilder:
    """
    Class to build and evaluate Prophet models for time series forecasting.
    """

    def __init__(self, df, params=None,
                 regressors=None,
                 n_future=FORECAST_HOURS,
                 freq=FORECAST_FREQUENCY):
        """
        Initialize the ProphetModelBuilder with the provided data.

        Parameters:
        df (pd.DataFrame): DataFrame containing the historical data.
        params (dict): Dictionary of Prophet model parameters.
        regressors (list or str): List of regressor column names or a single regressor.
        n_future (int): Number of future periods to forecast.
        freq (str): Frequency of the forecast periods (e.g., 'H' for hourly).
        rename_columns_prefix (str): Prefix to rename the forecasted columns.
        """
        if VAR_TARGET not in df.columns or VAR_DATE not in df.columns:
            raise ValueError(f"Prophet requires target column name must be {VAR_TARGET} and date column must be {VAR_DATE}")

        self.df = df
        self.params = params if params is not None else {}
        self.regressors = [regressors] if isinstance(regressors, str) else regressors
        self.n_future = n_future
        self.freq = freq

        # self.forecast_df = None

        # Build and fit a Prophet model with the provided parameters.

        self.model = Prophet(**self.params)
        if self.regressors:
            for regressor in self.regressors:
                self.model.add_regressor(regressor)
        self.model.fit(self.df)



    def forecast(self):
        """
        Forecast future values using the fitted Prophet model.

        Returns:
        Prophet, pd.DataFrame, pd.DataFrame, pd.DataFrame: Fitted model, complete forecast, train forecast, future forecast.
        """

        future = self.model.make_future_dataframe(periods=self.n_future, freq=self.freq)

        if self.regressors:
            for regressor in self.regressors:
                future[regressor] = self.df[regressor].mean()

        self.forecast_df = self.model.predict(future)

        train_forecast = self.forecast_df[:len(self.df)]
        future_forecast = self.forecast_df[len(self.df):]


        return self.model, self.forecast_df, train_forecast, future_forecast

    def evaluate_model(self):
        """
        Evaluate the model using cross-validation.

        Returns:
        pd.DataFrame: DataFrame containing the cross-validation performance metrics.
        """
        df_cv = cross_validation(
            self.model,
            initial=f'{round(len(self.df) * 0.7)} hours',  # 70% of data
            period='1440 hours',  # 60 days (24*60) will be added to the training dataset for each additional model
            horizon=f'{self.n_future} hours',
            parallel='processes'
        )
        df_p = performance_metrics(df_cv, rolling_window=1)
        return df_cv, df_p

    def plot_components(self):
        """
        Visualize the components of the forecast.
        """
        self.model.plot_components(self.forecast_df)

    def plot_changepoints(self):
        """
        Visualize the changepoints in the forecast.
        """
        fig = self.model.plot(self.forecast_df)
        add_changepoints_to_plot(fig.gca(), self.model, self.forecast_df)

    def plot_cross_validation_metric(self, df_cv, metric='mape'):
        """
        Visualize the cross-validation performance metrics.

        Parameters:
        df_cv (pd.DataFrame): DataFrame containing the cross-validation results.
        metric (str): Metric to plot (default is 'mape').
        """
        plot_cross_validation_metric(df_cv, metric=metric)

class ProphetModelTuner:
    def __init__(self, df, actual_future_df):
        """
        Initialize the ProphetModelTuner with the provided data.

        Parameters:
        df (pd.DataFrame): DataFrame containing the historical data.
        actual_future_df (pd.DataFrame): DataFrame containing the actual future data for evaluation.
        """
        self.df = df
        self.actual_future_df = actual_future_df

    def __get_all_regressor_combos(self, regressors):
        """
        Generate all possible subsets of regressors.

        Returns:
        list: List of tuples containing all possible combinations of regressors.
        """
        combinations = []
        for i in range(len(regressors) + 1):
            combinations.extend(itertools.combinations(regressors, i))
        return combinations

    def evaluate_regressors(self, regressors, params=None):
        """
        Evaluate the model using different combinations of regressors.

        Returns:
        list: List of DataFrames containing the performance metrics for each regressor combination.
        """
        try:
            results = []
            all_regressor_combos = self.__get_all_regressor_combos(regressors)
            for regressor in all_regressor_combos:
                # Convert tuple to list
                regressor = list(regressor) if regressor else None

                model_builder = ProphetModelBuilder(self.df, params=params, regressors=regressor)
                _, _, _, future_forecast = model_builder.forecast()

                mg = MetricsGenerator(
                    y_true=self.actual_future_df[VAR_TARGET].values,
                    y_pred=future_forecast[VAR_FORECAST].values,
                    is_regression_task=True,
                    target_name="Close Price"
                )
                df_p, _ = mg.performance()
                results.append(df_p)
            return results
        except Exception as e:
            print(f"Error in evaluate_regressors: {e}")

    def tune_hyperparams(self, param_grid=None, regressors=None):
        """
        Tune hyperparameters for the Prophet model.

        Returns:
        dict: Best hyperparameters found.
        list: List of DataFrames containing the performance metrics for each parameter combination.
        """
        if param_grid is None:
            param_grid = {
                "weekly_seasonality": [True],
                "interval_width": [0.95],  # Confidence interval of 95%
                'seasonality_mode': ['multiplicative'],  # Given the data's volatility and growth
                'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5, 0.6],  # To give higher value to prior trend
                'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],  # To control the flexibility of seasonality components
                'changepoint_range': np.linspace(0.7, 0.8, 11)  # Values from 0.7 to 0.8
            }

        # Generate all combinations of parameters
        all_params = [
            dict(zip(param_grid.keys(), v))
            for v in itertools.product(*param_grid.values())
        ]

        results = []
        rmses = []
        for params in all_params:
            model_builder = ProphetModelBuilder(self.df, params=params, regressors=regressors)
            df_cv, df_p = model_builder.evaluate_model()

            rmses.append(df_p['rmse'].values[0])
            results.append(df_p)

        # Find the best parameters
        best_params = all_params[np.argmin(rmses)]
        return best_params, results


def BaselineModelOracle(df, actual_future_df=None,
                        retune_model= False,
                        rename_columns_prefix=""):
    if retune_model:
        model_tuner = ProphetModelTuner(df, actual_future_df)
        best_params, results = model_tuner.tune_hyperparams()
    else:
        best_params = {
            'weekly_seasonality': True,
            'interval_width': 0.95,
            'seasonality_mode': 'multiplicative',
            'changepoint_prior_scale': 0.6,
            'seasonality_prior_scale': 0.01,
            'changepoint_range': 0.76
        }
    baseline_builder= ProphetModelBuilder(df,params=best_params)

    best_model, best_forecast, best_train_forecast, best_future_forecast = baseline_builder.forecast()

    if rename_columns_prefix !="":
        best_forecast = best_forecast.rename(columns=lambda x: f"{rename_columns_prefix}_{x}" if x not in df.columns else x)
        best_train_forecast = best_train_forecast.rename(columns=lambda x: f"{rename_columns_prefix}_{x}" if x not in df.columns else x)
        best_future_forecast = best_future_forecast.rename(columns=lambda x: f"{rename_columns_prefix}_{x}" if x not in df.columns else x)

    return best_model, best_forecast, best_train_forecast, best_future_forecast
