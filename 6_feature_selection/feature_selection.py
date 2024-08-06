# data_feature_selection/feature_selection.py

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
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from config import VAR_TARGET, VAR_DATE
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in scalar divide")

class FeatureSelector:
    def __init__(self, df):
        fs_df = df.copy()
        if VAR_DATE in fs_df.columns:
            fs_df.drop(columns=[VAR_DATE], inplace=True)
        self.df = fs_df
        self.X_feature_selection = self.df.drop(columns=[VAR_TARGET])
        self.y_feature_selection = self.df[VAR_TARGET]

    def correlation_selection(self, corr_threshold=0.1):
        """Select features based on correlation with the target variable."""
        corrs = self.df.corr()
        target_corrs = corrs[VAR_TARGET].abs().sort_values(ascending=False)
        corr_features = target_corrs[target_corrs > corr_threshold].index.tolist()
        corr_features = [feature for feature in corr_features if feature != VAR_TARGET]
        return corr_features

    def random_forest_selection(self, top_n=20):
        """Select top features based on Random Forest feature importance."""
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(self.X_feature_selection, self.y_feature_selection)
        feature_importances = pd.Series(rf.feature_importances_, index=self.X_feature_selection.columns)
        feature_importances = feature_importances.sort_values(ascending=False)
        rf_features = feature_importances.index[:top_n].tolist()
        return rf_features

    def variance_threshold(self, threshold=0.1):
        """Remove features with low variance."""
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(self.X_feature_selection)
        vt_features = self.X_feature_selection.columns[selector.get_support()].tolist()
        return vt_features

    def recursive_feature_elimination(self, n_features_to_select=20, check_multicollinearity=True):
        """Recursive Feature Elimination (RFE) to select important features."""
        X_no_zero_variance = self.X_feature_selection.loc[:, (self.X_feature_selection != self.X_feature_selection.iloc[0]).any()]
        lr = LinearRegression()

        # Ensure n_features_to_select is set correctly
        if isinstance(n_features_to_select, int):
            n_features_to_select = min(n_features_to_select, X_no_zero_variance.shape[1])

        rfe = RFE(estimator=lr, n_features_to_select=n_features_to_select)
        rfe.fit(X_no_zero_variance, self.y_feature_selection)
        rfe_features = X_no_zero_variance.columns[rfe.support_].tolist()

        if check_multicollinearity:
            X_selected_rfe = X_no_zero_variance[rfe_features]
            X_selected_rfe = sm.add_constant(X_selected_rfe)
            model_rfe = sm.OLS(self.y_feature_selection, X_selected_rfe).fit()
            p_values = model_rfe.pvalues

            selected_features = p_values[p_values < 0.05].index.tolist()
            if 'const' in selected_features:
                selected_features.remove('const')

            vif_data = pd.DataFrame()
            vif_data['feature'] = X_selected_rfe.columns
            vif_data['VIF'] = [variance_inflation_factor(X_selected_rfe.values, i) for i in range(X_selected_rfe.shape[1])]

            vif_data = vif_data.replace([np.inf, -np.inf], np.nan).dropna()
            vif_filtered_features = vif_data[vif_data['VIF'] < 10]['feature'].tolist()
            if 'const' in vif_filtered_features:
                vif_filtered_features.remove('const')

            return vif_filtered_features

        return rfe_features

    def feature_selection(self, check_multicollinearity=True):
        """Perform feature selection using multiple methods and return the selected features."""
        corr_features = self.correlation_selection()
        rf_features = self.random_forest_selection()
        vt_features = self.variance_threshold()
        rfe_features = self.recursive_feature_elimination(n_features_to_select=20, check_multicollinearity=check_multicollinearity)
        selected_features = list(set(corr_features) | set(rf_features) | set(vt_features) | set(rfe_features))
        return selected_features

# Example
# if __name__ == "__main__":
    # from utils.data_reader import generate_model_data
    #
    # # Load the dataset
    # btc_df = generate_model_data().copy()
    #
    # # Initialize the FeatureSelector
    # feature_selector = FeatureSelector(btc_df)
    #
    # # Perform feature selection
    # selected_features = feature_selector.feature_selection(check_multicollinearity=False)
    #
    # print(f"Selected Features:{selected_features}")



#%%
