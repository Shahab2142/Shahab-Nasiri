#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 15:59:02 2023

@author: shahab-nasiri
"""

from hyperopt import hp, fmin, tpe, Trials, space_eval, STATUS_OK
from sklearn.model_selection import cross_val_score, TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import re
from prophet import Prophet
import pandas as pd

"""""""""""""""""""""""""""""""""""""""""""""""""""""
XGBOOST FEATURE SELECTION, OPTIMIZATION, EVALUATION
"""""""""""""""""""""""""""""""""""""""""""""""""""""

def feature_selection(X, y, importance_multiplier=1):
    
    """
    Identifies important features by fitting an XGBoost model and selecting features with importances above a defined threshold.
    """
    
    model = xgb.XGBRegressor()
    model.fit(X, y)
    feature_importances = model.feature_importances_
    threshold = importance_multiplier * np.mean(feature_importances)
    important_features = X.columns[feature_importances > threshold]
    return important_features

def objective(params, X, y, important_features, n_splits):
    
    """
    The objective function for Bayesian optimization that evaluates the model's performance with cross-validation on the important features.
    """
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    mse = -cross_val_score(xgb.XGBRegressor(**params), X[important_features], y, cv=tscv, scoring='neg_mean_squared_error').mean()
    return {'loss': mse, 'status': STATUS_OK}

def perform_bayesian_optimization(space, max_evals, X, y, important_features, n_splits):
    
    """
    Conducts Bayesian optimization over a defined hyperparameter space to find the best hyperparameters for the XGBoost model.
    """
   
    trials = Trials()
    best = fmin(fn=lambda params: objective(params, X, y, important_features, n_splits), space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    return space_eval(space, best)

def selection_hyperparameterization(X, y, space, max_evals=50, importance_multiplier=1, n_splits=3):
    
    """
    Conducts feature selection and hyperparameter optimization for the XGBoost model.
    """
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Feature Selection on the training set
    important_features = feature_selection(X_train, y_train, importance_multiplier)
    
    # Hyperparameter Optimization on the important features
    best_params = perform_bayesian_optimization(space, max_evals, X_train, y_train, important_features, n_splits)
    
    # Train the final model on the selected features
    final_model = xgb.XGBRegressor(**best_params)
    final_model.fit(X_train[important_features], y_train)
    
    return final_model, X_train, X_test, y_train, y_test, important_features

def evaluate(final_model, X_train, X_test, y_train, y_test, important_features):
    
    """
    Evaluates the trained model's performance on both training and test data.
    """
    
    # Predict on training data
    y_train_pred = final_model.predict(X_train[important_features])
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    # Predict on test data
    y_test_pred = final_model.predict(X_test[important_features])
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Plot feature importances of the final model
    plt.figure(figsize=(12, 6))
    sorted_idx = np.argsort(final_model.feature_importances_)[::-1]
    plt.bar(X_train[important_features].columns[sorted_idx], final_model.feature_importances_[sorted_idx], color='red')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.xticks(rotation='vertical')
    plt.tight_layout()
    plt.show()
    
    # Results
    print(f"Train MSE: {train_mse}")
    print(f"Train R^2: {train_r2}")
    print(f"Test MSE: {test_mse}")
    print(f"Test R^2: {test_r2}")
    print(f"Test Target Variance: {np.var(y_test)}")
    
    return train_mse, train_r2, test_mse, test_r2

"""""""""""""""""""""""""""""""""""""""""
MITIGATING DATA LEAKAGE + FBPROPHET IMPUTING
"""""""""""""""""""""""""""""""""""""""""

def data_leakage(X_test):
    
    """
    Mitigates data leakage in the test set by filling specific feature columns with NaNs according to predefined rules.
    """
    
    for column in X_test.columns:
        if re.match(r'^Feature \d+$', column):
            X_test[column] = np.nan

        elif re.match(r'^(Feature|Target) \d+_lag\d+$', column):
            lag = int(re.findall(r'_lag(\d+)$', column)[0])
            X_test[column][lag:] = np.nan

        elif any(word in column for word in ['residual', 'seasonal', 'trend']):
            X_test[column] = np.nan

        elif 'rolling' in column:
            X_test[column] = np.nan

    return X_test


def feature_imputing_fb(X_test, X_train):
    
    """
   This function iterates through each 'Feature X' column in X_train, creates and fits a Prophet model,
   and imputes missing values in the corresponding column of X_test using the model's forecast.
   The function adds both weekly and yearly seasonality to each model.
   """
    
    for column in X_train.columns:
        if re.match(r'^Feature \d+$', column):
            # Create a synthetic date-time index
            dates = pd.date_range(start='2021-01-01', periods=len(X_train), freq='W')
            
            # Prepare the training data for Prophet
            train_data = pd.DataFrame({
                'ds': dates,
                'y': X_train[column]
            })

            # # Initialize and fit the Prophet model
            model = Prophet()
            model.add_seasonality(name='monthly', period=4, fourier_order=10) # Monthly seasonality
            model.add_seasonality(name='yearly', period=52, fourier_order=10) # Yearly seasonality
            model.fit(train_data)

            # Prepare the testing data
            future_dates = model.make_future_dataframe(periods=len(X_test))
            forecast = model.predict(future_dates)

            # Replace the test column with predictions
            X_test[column] = forecast['yhat'][-len(X_test):].values

    return X_test