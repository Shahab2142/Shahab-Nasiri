#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 15:36:05 2023

@author: shahabyousef-nasiri
"""

import pandas as pd
from cleaning_outliers_filters import *
from feature_engineering import *
from XGBOOST import *

"""""""""""""""""""""""""""""""""""""""
DATA PROCESSING, FEATURE ENGINEERING
"""""""""""""""""""""""""""""""""""""""

file_path = 'features.csv'
percentage_column_threshold = 0.185
rows_to_drop = 310
target_to_analyze = 'Target 1'
target_to_drop = 'Target 2'
spearman_correlation_alpha = 0.7
MAD_std_threshold = 3
cross_correlation_alpha = 0.025
auto_correlation_alpha = 0.025
max_lag = 52
auto_decompositions_spearman_alpha = 0.1
rolling_stats_periods = [4, 12, 52]
rolling_stats_types = ['mean', 'var', 'std', 'min', 'max']
fourier_feature_periods = [12, 26, 52]
fourier_feature_harmonics = [1, 2, 3]
waves_to_add = ['sin', 'cos']


def processing_and_feature_engineering(file_path, percentage_column_threshold, rows_to_drop, 
                                          target_to_analyze, target_to_drop,
                                          spearman_correlation_alpha, MAD_std_threshold, cross_correlation_alpha, auto_correlation_alpha, max_lag,
                                          auto_decompositions_spearman_alpha, rolling_stats_periods, rolling_stats_types, fourier_feature_periods, fourier_feature_harmonics, waves_to_add):
    """
    Perform processing and feature engineering on the dataset.
    """
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Perform the series of preprocessing and feature engineering steps
    df = dropping_columns_rows(df, percentage_column_threshold, rows_to_drop, target_to_analyze, target_to_drop)
    df = replace_inf_with_nan(df, target_to_analyze)
    df = spearman_correlation_filter(df, target_to_analyze, spearman_correlation_alpha)
    df = constant_duplicate_filter(df)
    df = outlier_detection_MAD(df, MAD_std_threshold)
    df = auto_lag_feature_creation_wiz(df, target_to_analyze, max_lag, cross_correlation_alpha, auto_correlation_alpha)
    df = auto_decompositions_feature_creation_wiz(df, target_to_analyze, auto_decompositions_spearman_alpha)
    df = add_rolling_statistics_features(df, target_to_analyze, rolling_stats_periods, rolling_stats_types)
    df = add_fourier_features(df, fourier_feature_periods, fourier_feature_harmonics,waves_to_add )
    df = add_time_features(df)
    df = add_cyclical_features(df)

    # Save the processed DataFrame to a CSV file
    df.to_csv('PROCESSED_DATA.csv', index=False)

    return df

processing_and_feature_engineering(file_path, percentage_column_threshold, rows_to_drop, target_to_analyze, target_to_drop,
                                      spearman_correlation_alpha, MAD_std_threshold, cross_correlation_alpha, auto_correlation_alpha, max_lag,
                                      auto_decompositions_spearman_alpha, rolling_stats_periods,rolling_stats_types, fourier_feature_periods,fourier_feature_harmonics, waves_to_add)


"""""""""""""""""""""""
PREDICTING USING XGBOOST
"""""""""""""""""""""""
space = {
    'objective': 'reg:squarederror',
    'max_depth': hp.choice('max_depth', range(3, 10)),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
    'subsample': hp.uniform('subsample', 0.7, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.7, 1),
    'n_estimators': hp.choice('n_estimators', range(50, 300)),
    'reg_alpha': hp.uniform('reg_alpha', 0, 1),
    'reg_lambda': hp.uniform('reg_lambda', 0, 1)
}

max_evals = 10
importance_multiplier = 2
n_splits = 10

df = pd.read_csv('PROCESSED_DATA.csv')
X = df.drop([target_to_analyze], axis=1)
y = df[target_to_analyze]

final_model, X_train, X_test, y_train, y_test, important_features = selection_hyperparameterization(X, y, space, max_evals, importance_multiplier, n_splits)
train_mse, train_r2, test_mse, test_r2 = evaluate(final_model, X_train, X_test, y_train, y_test, important_features)


"""""""""""""""""""""""""""""""""""
FORECASTING USING XGBOOST + PROPHET
"""""""""""""""""""""""""""""""""""

# space = {
#     'objective': 'reg:squarederror',
#     'max_depth': hp.choice('max_depth', range(3, 10)),
#     'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
#     'subsample': hp.uniform('subsample', 0.7, 1),
#     'colsample_bytree': hp.uniform('colsample_bytree', 0.7, 1),
#     'n_estimators': hp.choice('n_estimators', range(50, 300)),
#     'reg_alpha': hp.uniform('reg_alpha', 0, 1),
#     'reg_lambda': hp.uniform('reg_lambda', 0, 1)
# }

# max_evals = 25
# importance_multiplier = 2
# n_splits = 10

# df = pd.read_csv('PROCESSED_DATA.csv')
# X = df.drop([target_to_analyze], axis=1)
# y = df[target_to_analyze]

# final_model, X_train, X_test, y_train, y_test, important_features = selection_hyperparameterization(X, y, space, max_evals, importance_multiplier, n_splits)
# X_test = data_leakage(X_test)
# X_test = feature_imputing_fb(X_test, X_train)
# train_mse, train_r2, test_mse, test_r2 = evaluate(final_model, X_train, X_test, y_train, y_test, important_features)
