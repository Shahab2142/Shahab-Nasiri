#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 20:30:45 2023

@author: shahabyousef-nasiri
"""
from statsmodels.tsa.stattools import adfuller, pacf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import ccf
from prophet import Prophet
from scipy.stats import spearmanr
import scipy.stats as stats
from scipy.stats import median_abs_deviation
from statsmodels.tsa.seasonal import seasonal_decompose

"""""""""""""""""
CLEANING & OUTLIERS 
"""""""""""""""""

def dropping_columns_rows(df, percentage_column_threshold, rows_to_drop,target_to_analyze,target_to_drop):
    
    """
    This function cleans the dataset by dropping the Unnamed column, rows with NaNs in the target column, and columns based on a threshold of non-NaN data. 
    The user can also choose to trim the the data columns down a certain number of rows.
    """
    
    df = df.drop('Unnamed: 0', axis=1)
    df = df.drop(target_to_drop,axis=1)
    df = df.dropna(subset=[target_to_analyze])
    
    col_threshold = percentage_column_threshold * len(df)
    df = df.dropna(axis=1, thresh=col_threshold)
    df = df.iloc[rows_to_drop:]
    df = df.reset_index(drop=True)    
    
    return df


def replace_inf_with_nan(df,target_to_analyze):
    
    """
    This function replaces infinite values in the DataFrame with NaN.
    """
    
    for column in df.columns:
        if column not in [target_to_analyze]:
            df[column] = df[column].replace([np.inf, -np.inf], np.nan)
            
    return df


def outlier_detection_MAD(df, threshold):
    
    """
    This function replaces outliers in the Features columns with NaN, based on the MAD method and a threshold modified z-score.
    """

    pattern = re.compile(r'^Feature \d+$') 
    
    for column in df.columns:
        if pattern.match(column):
            
            series = df[column]
            median = np.median(series)
            mad = median_abs_deviation(series, scale='normal')
            modified_z_scores = 0.6745 * (series - median) / mad

            outliers = np.abs(modified_z_scores) > threshold
            # Plotting the data and highlighting outliers
            # plt.figure(figsize=(10, 5))
            # plt.plot(series, label='Data', marker='o', linestyle='', color='blue')
            # plt.plot(series[outliers], marker='o', linestyle='', color='red', label='Outliers')
            # plt.xlabel('Index')
            # plt.ylabel('Value')
            # plt.title(f'Outliers in {column}')
            # plt.legend()
            # plt.show()
            
            # Replacing outliers with NaN
            df.loc[outliers, column] = np.nan
   
    return df

"""""""""""""""""
STATISTICAL FILTERS
"""""""""""""""""

def constant_duplicate_filter(df):
 
    """
    This function removes constant features and duplicate rows from the DataFrame.
    """
    
    constant_features = df.columns[df.std() == 0]
    df = df.drop(columns=constant_features)
    df = df.T.drop_duplicates(keep='first').T

    return df


def is_stationary(series, significance=0.025):
    
    """
    This function checks the stationarity of a time series using the Augmented Dickey-Fuller test, returning True if the p-value is less than or equal to the specified significance level.
    """
    result = adfuller(series.dropna(), autolag='AIC')
    
    return result[1] <= significance


def spearman_correlation_filter(df, target_to_analyze, threshold):
    
    """
    This function removes features from the DataFrame that have a Spearman correlation coefficient with the target variable exceeding the specified threshold.
    """
    
    pattern = re.compile(r'^Feature \d+$')
    
    columns_to_drop = []
    
    for column in df.columns:
        if pattern.match(column):
            correlation, p_value = stats.spearmanr(df[column], df[target_to_analyze])
            if abs(correlation) >= threshold:
                columns_to_drop.append(column)
                print(f"Removing {column} with Spearman correlation coefficient: {correlation}")
                
    df = df.drop(columns=columns_to_drop)
    
    return df


def cross_correlation_lag_filter(feature_series, target_series, max_lag, alpha):

    """
    This function identifies significant lags where the cross-correlation between a feature and target series exceeds a statistical threshold.
    """
    
    cross_corr_values = ccf(target_series, feature_series, unbiased=False)[:max_lag+1]
    significant_lags = []

    threshold = stats.norm.ppf(1 - alpha / 2) / np.sqrt(len(target_series))

    for i in range(1, len(cross_corr_values)):
        if abs(cross_corr_values[i]) > threshold:
            significant_lags.append(i)

    print('Significant lags:', significant_lags)
    
    return significant_lags


def partial_auto_correlation_function_filter(target_series, max_lag, alpha):
    
    """
    This function determines significant lags in the target time series based on the partial autocorrelation function up to a specified maximum lag, considering a given significance level alpha.
    """
    
    pacf_values, confint = pacf(target_series, nlags=max_lag, alpha=alpha)
    significant_lags = []

    for i in range(1, len(confint)):
        lower_bound = confint[i][0]
        upper_bound = confint[i][1]
        
        if lower_bound > 0 or upper_bound < 0:
            significant_lags.append(i)
    
    
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_pacf(target_series, ax=ax, lags=max_lag, alpha=alpha)
    for lag in significant_lags:
        ax.axvline(x=lag, color='red', linestyle='--')
    plt.show()
    
    
    print('Significant lags:', significant_lags)
    return significant_lags

