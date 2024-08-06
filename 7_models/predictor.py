# models/predictor.py
"""
@author: Esther Xu
Code tested on:
    - os: Debian 11
    - Python: 3.10.14; 3.11.2
    - Numpy: 1.25.2
    - Sklearn: 1.3.0
    - Tensorflow: 2.13.0; 2.14.0
    - Keras: 2.8; 2.14.0
"""

import pandas as pd
import numpy as np
import keras_tuner as kt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from config import FORECAST_HOURS, VAR_DATE, VAR_TARGET, FORECAST_FREQUENCY, VAR_FORECAST,PREDICTIONS_DIR,DEFAULT_REGRESSION_METRICS
from models.baseline import BaselineModelOracle
from data_feature_selection.feature_selection import FeatureSelector
from utils.viz import plot_train_validation_loss_values
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from utils.metrics_generator import MetricsGenerator
from datetime import datetime
import os
import joblib

class EniqueOracle:
    def __init__(self, df, model_dir,n_future=FORECAST_HOURS, retrain_feature_selection=False, check_multicollinearity=False):
        """
        Initialize the EniqueOracle class.

        Parameters:
        - df: DataFrame containing the dataset.
        - n_future: Number of hours to forecast into the future.
        - retrain_feature_selection: Boolean to decide whether to retrain feature selection.
        - check_multicollinearity: Boolean to check multicollinearity during feature selection.
        - model_path: Path to save or load the trained model.
        """
        self.df = df
        self.n_future = n_future
        self.retrain_feature_selection = retrain_feature_selection
        self.check_multicollinearity = check_multicollinearity
        self.model_path = f"{model_dir}/best_enique_model.h5"
        self.hyperparams_path = "best_hyperparams.pkl"
        self.__build_baseline_oracle()

    def __build_baseline_oracle(self):
        """
        Build the baseline oracle model using the dataset.
        """
        baseline_model, _, baseline_train_forecast, baseline_future_forecast = BaselineModelOracle(self.df, rename_columns_prefix="prophet")
        baseline_train_forecast = self.df.merge(baseline_train_forecast, on=VAR_DATE, how='left')
        self.df = baseline_train_forecast
        self.baseline_future_forecast = baseline_future_forecast
        self.baseline_model = baseline_model

    def __feature_selection(self):
        """
        Perform feature selection on the dataset.
        """
        if self.retrain_feature_selection:
            selected_features = FeatureSelector(self.df).feature_selection(check_multicollinearity=self.check_multicollinearity)
            print(f"selected_features{selected_features}")
        else:
            selected_features = ['sentiment_neutral_lag_2hr', 'bollinger_lower', 'prophet_yhat_lower',
                                 'sentiment_positive_lag_6hr', 'sentiment_neutral', 'sentiment_negative_lag_7hr',
                                 'open_price', 'raw_money_flow', 'sentiment_neutral_lag_7hr', 'RSI',
                                 'sentiment_neutral_volatility', 'sentiment_neutral_lag_3hr', 'MACD_diff',
                                 'num_of_trades', 'prophet_yhat_upper', 'y_lower', 'bollinger_upper',
                                 'tw_avg_negative', 'sentiment_positive_volatility', 'quote_asset_volume',
                                 'sentiment_negative_lag_6hr', 'sentiment_neutral_lag_1hr', 'MACD', 'MFI',
                                 'sentiment_negative_lag_3hr', 'sentiment_neutral_lag_5hr', 'volume',
                                 'tw_avg_positive', 'sentiment_negative', 'prophet_weekly', 'taker_buy_base_asset_volume',
                                 'sentiment_negative_volatility', 'tw_avg_neutral', 'typical_price', 'MACD_signal',
                                 'sentiment_neutral_lag_4hr', 'positive_money_flow', 'sentiment_neutral_lag_6hr',
                                 'sentiment_negative_lag_5hr', 'sentiment_negative_lag_2hr', 'taker_buy_quote_asset_volume',
                                 'y_upper', 'negative_money_flow', 'sentiment_negative_lag_1hr', 'sentiment_negative_lag_4hr',
                                 'prophet_yhat', 'prophet_trend']

            if 'prophet_yearly' in self.df.columns:
                selected_features += ['prophet_yearly']
            if 'prophet_daily' in self.df.columns:
                selected_features += ['prophet_daily']

        selected_features += [VAR_DATE, VAR_TARGET]
        return selected_features

    def __prepare_data_for_lstm(self, n_past, validation_split=0.2):
        """
        Prepare the dataset for LSTM training.

        Parameters:
        - n_past: Number of past observations to use for training.
        - validation_split: Fraction of the data to use for validation.

        Returns:
        - trainX, trainY: Training features and targets.
        - valX, valY: Validation features and targets.
        - scaler: Scaler object used for data normalization.
        """
        cols = self.df.columns.to_list()
        if VAR_DATE in cols:
            cols.remove(VAR_DATE)

        df_for_training = self.df[cols].astype(float)

        scaler = StandardScaler()
        df_for_training_scaled = scaler.fit_transform(df_for_training)

        split_point = int(len(df_for_training_scaled) * (1 - validation_split))
        train_data = df_for_training_scaled[:split_point]
        val_data = df_for_training_scaled[split_point:]

        trainX, trainY = [], []
        for i in range(n_past, len(train_data) - self.n_future + 1):
            trainX.append(train_data[i - n_past:i, :])
            trainY.append(train_data[i + self.n_future - 1, -1])

        trainX, trainY = np.array(trainX), np.array(trainY)

        valX, valY = [], []
        for i in range(n_past, len(val_data) - self.n_future + 1):
            valX.append(val_data[i - n_past:i, :])
            valY.append(val_data[i + self.n_future - 1, -1])

        valX, valY = np.array(valX), np.array(valY)

        return trainX, trainY, valX, valY, scaler

    def create_model(self, hp):
        """
        Create an LSTM model with hyperparameter tuning.

        Parameters:
        - hp: Hyperparameters object for tuning.

        Returns:
        - model: Compiled Keras model.
        """
        units = hp.Int('units', min_value=32, max_value=512, step=32)
        dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)
        learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG')

        model = Sequential()
        model.add(LSTM(units, return_sequences=True, input_shape=(None, self.input_shape), kernel_regularizer=l2(0.01)))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(units, kernel_regularizer=l2(0.01)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1, kernel_regularizer=l2(0.01)))  # Output layer for 'y'

        optimizer = hp.Choice('optimizer', values=['adam', 'rmsprop'])
        model.compile(optimizer=optimizer, loss='mean_absolute_percentage_error')
        return model

    def hyperparameter_tuning(self, X_train, y_train, max_trials):
        """
        Perform hyperparameter tuning for the LSTM model.

        Parameters:
        - X_train, y_train: Training features and targets.
        - max_trials: Maximum number of trials for hyperparameter tuning.

        Returns:
        - best_model: Best model obtained from the tuning process.
        - history: Training history of the best model.
        """
        self.input_shape = X_train.shape[2]

        tuner = kt.RandomSearch(
            self.create_model,
            objective='val_loss',
            max_trials=max_trials,
            executions_per_trial=3,
            directory='hyperparam_tuning',
            project_name='enique_bitcoin_price_prediction'
        )

        stop_early = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        tuner.search(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[stop_early])

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        best_params = {
            'units': best_hps.get('units'),
            'dropout_rate': best_hps.get('dropout_rate'),
            'learning_rate': best_hps.get('learning_rate'),
            'epochs': 100,
            'batch_size': 32,
            'n_past': 120 * 3,
            'patience': 10
        }

        with open(self.hyperparams_path, 'wb') as f:
            joblib.dump(best_params, f)

        print(f"""
        The hyperparameter search is complete. The optimal number of units is {best_params['units']}, 
        the optimal dropout rate is {best_params['dropout_rate']}, and the optimal learning rate is {best_params['learning_rate']}.
        """)

        best_model = tuner.hypermodel.build(best_hps)
        history = best_model.fit(X_train, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'], validation_split=0.2, callbacks=[stop_early])

        return best_model, history

    def daily_retraining(self, max_trials=5):
        """
        Daily retraining of the LSTM model using the latest data.

        Returns:
        - history: Training history of the retrained model.
        """
        with open(self.hyperparams_path, 'rb') as f:
            best_params = joblib.load(f)

        selected_features = self.__feature_selection()
        self.df = self.df[selected_features]
        X_train, y_train, X_val, y_val, scaler = self.__prepare_data_for_lstm(best_params['n_past'], validation_split=0.2)

        if os.path.exists(self.model_path):
            best_model = load_model(self.model_path)
        else:
            best_model, _ = self.hyperparameter_tuning(X_train, y_train, max_trials=max_trials)

        stop_early = EarlyStopping(monitor='val_loss', patience=best_params['patience'], restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

        history = best_model.fit(X_train, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'], validation_data=(X_val, y_val), callbacks=[stop_early, reduce_lr])

        best_model.save(self.model_path)

        return history

    def get_enique_oracle(self, retune=False,max_trials=5):
        """
        Train and forecast using the EniqueOracle model.

        Parameters:
        - retune: Boolean to decide whether to retune the hyperparameters.

        Returns:
        - history: Training history of the best model.
        - df_forecast: Forecasted values.
        - baseline_future_forecast: Baseline forecast values.
        """
        if not os.path.exists(self.hyperparams_path):
            print(f"Hyperparameters file not found. Retuning...")
            retune = True

        if retune:
            selected_features = self.__feature_selection()
            self.df = self.df[selected_features]
            X_train, y_train, X_val, y_val, scaler = self.__prepare_data_for_lstm(120 * 3, validation_split=0.2)
            best_model, history = self.hyperparameter_tuning(X_train, y_train, max_trials=max_trials)
            best_model.save(self.model_path)
        else:
            with open(self.hyperparams_path, 'rb') as f:
                best_params = joblib.load(f)
            selected_features = self.__feature_selection()
            self.df = self.df[selected_features]
            X_train, y_train, X_val, y_val, scaler = self.__prepare_data_for_lstm(best_params['n_past'], validation_split=0.2)

            if os.path.exists(self.model_path):
                best_model = load_model(self.model_path)
                history = None
            else:
                best_model, history = self.hyperparameter_tuning(X_train, y_train, max_trials=max_trials)
                best_model.save(self.model_path)

        current_sequence = X_val[-1].reshape(1, X_val.shape[1], X_val.shape[2])

        predict_period_dates = pd.date_range(start=self.df[VAR_DATE].max(), periods=self.n_future, freq=FORECAST_FREQUENCY).tolist()

        prediction = []
        for _ in range(self.n_future):
            pred = best_model.predict(current_sequence)[0, 0]
            prediction.append(pred)
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, -1] = pred

        prediction = np.array(prediction).reshape(-1, 1)
        yhat = scaler.inverse_transform(np.repeat(prediction, X_train.shape[2], axis=-1))[:, -1]

        if len(yhat) != len(predict_period_dates):
            print("Length mismatch: y_pred_future and predict_period_dates")
            print(f"Length of y_pred_future: {len(yhat)}")
            print(f"Length of predict_period_dates: {len(predict_period_dates)}")

        df_forecast = pd.DataFrame({VAR_DATE: np.array(predict_period_dates[:len(yhat)]), VAR_FORECAST: yhat})
        df_forecast[VAR_DATE] = pd.to_datetime(df_forecast[VAR_DATE])
        df_forecast.to_csv(f"{PREDICTIONS_DIR}/btc_predictions_{datetime.now()}.csv")
        return history, df_forecast, self.baseline_future_forecast

    def evaluate_model(self, test_df, enique_forecast_df, baseline_future_forecast, current_datetime,save_to_dir):
        """
        Evaluate the performance of the model and compare it with the baseline.

        Parameters:
        - test_df: DataFrame containing the test dataset.
        - enique_forecast_df: DataFrame containing the forecasted values from the EniqueOracle model.
        - baseline_future_forecast: DataFrame containing the forecasted values from the baseline model.
        - current_datetime: Current date and time for saving results.

        Returns:
        - results_df: DataFrame containing the evaluation results.
        """
        y_true = test_df[VAR_TARGET]
        y_pred_enique = enique_forecast_df[VAR_FORECAST]
        y_pred_baseline = baseline_future_forecast[f'prophet_{VAR_FORECAST}']

        performance_enique = MetricsGenerator(y_true, y_pred_enique, is_regression_task=True, metrics=DEFAULT_REGRESSION_METRICS)
        df_p_enique, summary_enique = performance_enique.performance()
        print(f"Enique: {summary_enique}\n")
        print(f"{mean_absolute_percentage_error(y_true, y_pred_enique)}")
        print(f"{mean_absolute_error(y_true, y_pred_enique)}")
        print(f"{mean_squared_error(y_true, y_pred_enique, squared=True)}")

        performance_baseline = MetricsGenerator(y_true, y_pred_baseline, is_regression_task=True, metrics=DEFAULT_REGRESSION_METRICS)
        df_p_baseline, summary_baseline = performance_baseline.performance()
        print(f"Baseline: {summary_baseline}")

        results = []
        results.append({
            'current_datetime': current_datetime,
            'training_start': self.df[VAR_DATE].min(),
            'training_end': self.df[VAR_DATE].max(),
            'validation_size': len(self.df) * 0.2,
            'mystery_start': test_df[VAR_DATE].min(),
            'mystery_end': test_df[VAR_DATE].max(),
            'y': y_true,
            'yhat_enique': y_pred_enique,
            'yhat_baseline': y_pred_baseline,
            'mae_enique': df_p_enique['mae'],
            'rmse_enique': df_p_enique['rmse'],
            'mape_enique': df_p_enique['mape'],
            'metrics_enique': summary_enique,
            'mae_baseline': df_p_baseline['mae'],
            'rmse_baseline': df_p_baseline['rmse'],
            'mape_baseline': df_p_baseline['mape'],
            'metrics_baseline': summary_baseline
        })
        results_df = pd.DataFrame(results)
        results_df.to_pickle(f"{save_to_dir}/backtesting_{datetime.now()}.pkl")

        return results_df


#%%
