# setup.py
"""
@author: Esther Xu
Code tested on:
    - os: Debian 11
    - Python: 3.10.14; 3.11.2
    - Numpy: 1.25.2
    - Sklearn: 1.3.0
    - Tensorflow: 2.13.0; 2.14.0
    - Keras: 2.8; 2.14.0
    - Hadoop 3.3
    - Spark 3.3
    - Spark-nlp: 4.2.0
    - Prophet: 1.1.5

"""
from models.predictor import EniqueOracle
from utils.data_reader import generate_model_data, trim_dataset
from config import (FORECAST_HOURS,VAR_DATE,MODEL_DIR)

if __name__ == "__main__":


    # print("====================Are you ready? Go=================>")
    # print("====================Loading Newset Data=================")
    # # For Newest forecast
    merge_df = generate_model_data()
    print(f"merge data start: {merge_df[VAR_DATE].min()}")
    print(f"merge data end: {merge_df[VAR_DATE].max()}")
    print(f"merge data has: {merge_df.shape[0]}")
    # print("====================End with Data Loading=================")


    # FORCASTING:************************** I'm a happy a cutting line****************************
    print("====================Starting Forcasting=================")

    oracle = EniqueOracle(df=merge_df,model_dir=f"{MODEL_DIR}", n_future=FORECAST_HOURS, retrain_feature_selection=False,
                          check_multicollinearity=False)


    history, enique_forecast_df, baseline_future_forecast = oracle.get_enique_oracle(retune=False,max_trials=1)


    print("====================End with Forcasting And Data Saved=================")

    print("====================End with Journey=================")



# TRANING AND TESING:************************** I'm a happy a cutting line****************************

# print("------------------------Loading Testing Data----------------------")
# current_datetime='2024-07-28'
# train_df, test_df = trim_dataset(merge_df, current_datetime, train_days=360*4)
# print(f"train start: {train_df[VAR_DATE].min()}")
# print(f"train end: {train_df[VAR_DATE].max()}")
# print(f"train has: {train_df.shape[0]}")
# print(f"test start: {test_df[VAR_DATE].min()}")
# print(f"test end: {test_df[VAR_DATE].max()}")
# print(f"test has: {test_df.shape[0]}")
#
# print("------------------------End with Data Loading----------------------")
#
# print("------------------------Starting Tuning and Evaluation------------------------")
#
# oracle = EniqueOracle(df=train_df,model_dir=f"{MODEL_DIR}", n_future=FORECAST_HOURS, retrain_feature_selection=False,
#                       check_multicollinearity=False)
#
#
# history, enique_forecast_df, baseline_future_forecast = oracle.get_enique_oracle(retune=False,max_trials=5)
#
# results_df = oracle.evaluate_model(test_df, enique_forecast_df, baseline_future_forecast, current_datetime,save_to_dir=DATA_FILE_DIR)
# print(results_df)
#
# print("------------------------Data Saved------------------------")
# print("------------------------End with Journey------------------------")

#%%

#%%
