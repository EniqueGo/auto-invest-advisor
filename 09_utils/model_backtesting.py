
# utils/model_backtesting.py

"""
@author: Esther Xu
Code tested on:
    - os: Debian 11
    - Python: 3.10.14; 3.11.2
    - Numpy: 1.25.2
    - Sklearn: 1.3.0
"""

from utils.metrics_generator import MetricsGenerator
from utils.data_utils import trim_dataset, split_data_by_cutoff_point, prepare_data,generate_model_data
from models import EniqueOracle

import pandas as pd
from config import VAR_TARGET,VAR_FORECAST,VAR_DATE,DEFAULT_REGRESSION_METRICS

initial_datetime = pd.to_datetime('2024-07-01 01:30:00')
end_datetime = pd.to_datetime('2024-07-20 23:15')

def stepwise_training_and_prediction(initial_datetime,
                                     end_datetime,
                                     n_past=120*3,
                                     n_future=120,
                                     train_days=180,
                                     step_hours=1,
                                     validation_split=0.2,
                                     epochs=100,
                                     batch_size=32,
                                     patience=10,
                                     verbose=1,
                                     save_to_path="backtesting_1"):
    current_datetime = pd.to_datetime(initial_datetime).floor('H')
    end_datetime = pd.to_datetime(end_datetime).floor('H')
    merge_df = generate_model_data()

    results = []
    while current_datetime <= end_datetime:
        print(f'Current datetime: {current_datetime}')

        # 1. Trim dataset to get train (including validation data) and true_future_df
        train_df, test_df = trim_dataset(merge_df, current_datetime, train_days=train_days)
        oracle = EniqueOracle(df=train_df, n_future=n_future,
                              retrain_feature_selection=False,
                              validation_split=validation_split)
        enique_forecast_df, baseline_future_forecast = (
            oracle.EniqueOracle(n_past=n_past, epochs=epochs, batch_size=batch_size,
                                patience=patience, verbose=verbose))

        y_true = test_df[VAR_TARGET]
        y_pred_enique = enique_forecast_df[VAR_FORECAST]
        y_pred_baseline = baseline_future_forecast[f'prophet_{VAR_FORECAST}']

        performance_enique = MetricsGenerator(y_true, y_pred_enique,
                                              is_regression_task=True,
                                              metrics=DEFAULT_REGRESSION_METRICS)
        df_p_enique, summary_enique = performance_enique.performance()
        print(f"Enique: {summary_enique}")

        performance_baseline = MetricsGenerator(y_true, y_pred_baseline,
                                                is_regression_task=True,
                                                metrics=DEFAULT_REGRESSION_METRICS)
        df_p_baseline, summary_baseline = performance_baseline.performance()
        print(f"Baseline: {summary_baseline}")

        results.append({
            'current_datetime': current_datetime,
            'training_start': train_df[VAR_DATE].min(),
            'training_end': train_df[VAR_DATE].max(),
            'validation_size': len(train_df)*validation_split,
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

        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_pickle(f"{save_to_path}.pkl")

        # Step forward in time
        current_datetime += pd.Timedelta(hours=step_hours)

    return results_df

