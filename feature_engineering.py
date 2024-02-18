#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:11:00 2023

@author: shahab-nasiri
"""
from processing_cleaning_outliers_filters import *

"""""""""""""""""""""""""""""
TIME AND SEASONALITY FEATURES
"""""""""""""""""""""""""""""

def add_time_features(df, start_date='2020-01-05', add_year=True, add_month_of_year=True, 
                      add_week_of_year=True, add_quarter=True, add_season=True):
    """
    This function fills the DataFrame with time-related features derived from a sequence of dates.
    """
    
    df['Date'] = pd.date_range(start=start_date, periods=len(df), freq='W')
    
    if add_year:
        df['Year'] = df['Date'].dt.year
    
    if add_month_of_year:
        df['Month_of_Year'] = df['Date'].dt.month
    
    if add_week_of_year:
        df['Week_of_Year'] = df['Date'].dt.isocalendar().week
    
    if add_quarter:
        df['Quarter'] = df['Date'].dt.quarter
    
    if add_season:
        df['Season'] = df['Date'].dt.month % 12 // 3 + 1
    
    df = df.drop('Date', axis=1)
    
    return df


def add_cyclical_features(df, add_month_of_year=True, add_week_of_year=True, add_quarter=True, add_season=True):
    
    """
    This function transforms specified cyclical features into their sine and cosine components, facilitating the capture of cyclical patterns in models.
    """
    
    if add_month_of_year:
        max_value = 12
        df['Month_of_Year_sin'] = np.sin(2 * np.pi * df['Month_of_Year'] / max_value)
        df['Month_of_Year_cos'] = np.cos(2 * np.pi * df['Month_of_Year'] / max_value)

    if add_week_of_year:
        max_value = 52
        df['Week_of_Year_sin'] = np.sin(2 * np.pi * df['Week_of_Year'] / max_value)
        df['Week_of_Year_cos'] = np.cos(2 * np.pi * df['Week_of_Year'] / max_value)

    if add_quarter:
        max_value = 4
        df['Quarter_sin'] = np.sin(2 * np.pi * df['Quarter'] / max_value)
        df['Quarter_cos'] = np.cos(2 * np.pi * df['Quarter'] / max_value)

    if add_season:
        max_value = 4
        df['Season_sin'] = np.sin(2 * np.pi * df['Season'] / max_value)
        df['Season_cos'] = np.cos(2 * np.pi * df['Season'] / max_value)

    return df
            

"""""""""""""""""""""""""""""""""""""""
STATISTICAL AND DECOMPOSITION FEATURES
"""""""""""""""""""""""""""""""""""""""

def auto_decompositions_feature_creation_wiz(df, target_column, alpha):
    
    """
    This function automatically decomposes non-stationary features that are significantly correlated with the target variable.
    """
    
    pattern = re.compile(r'^Feature \d+$')
    
    correlations = df.corr(method='spearman')[target_column]
    
    
    for column in df.columns:
        if pattern.match(column) and not is_stationary(df[column]) and abs(correlations[column]) > alpha:  
            print(f"Decomposing {column} due to significant correlation with the target...")
            df = add_decompositions(df, column)
            
    
    return df


def auto_lag_feature_creation_wiz(df, target_column, max_lag, cross_correlation_alpha, auto_correlation_alpha):
    
    """
    This function generates new lagged features for stationary series in the dataset that are significantly correlated with the target variable, up to a specified maximum lag.
    """
    
    pattern = re.compile(r'^Feature \d+$')
    
    for column in df.columns:
        if pattern.match(column) and is_stationary(df[column]):
            print(f"Processing feature {column} with respect to target...")
            feature_significant_lags = cross_correlation_lag_filter(df[column].dropna(), df[target_column].dropna(), max_lag, cross_correlation_alpha)
            df = add_lag_features(df, column, feature_significant_lags)
    
    target_significant_lags = partial_auto_correlation_function_filter(df[target_column].dropna(), max_lag, auto_correlation_alpha)
    df = add_lag_features(df, target_column, target_significant_lags)

    return df


def add_decompositions(df, column_name, add_trend=True, add_seasonal=True, add_residual=True):
    
    series = df[column_name].dropna()
    result = seasonal_decompose(series, model='additive', period=52)  # Assuming weekly data, hence period=52
    
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 8), sharex=True)
    plt.suptitle(f'Seasonal Decomposition of {column_name}')
    
    if add_trend:
        trend_col_name = f"{column_name}_trend"
        df[trend_col_name] = None
        df.loc[df.index.isin(series.index), trend_col_name] = result.trend.values
        axes[0].plot(result.trend, label='Trend')
        axes[0].legend(loc='best')

    if add_seasonal:
        seasonal_col_name = f"{column_name}_seasonal"
        df[seasonal_col_name] = None
        df.loc[df.index.isin(series.index), seasonal_col_name] = result.seasonal.values
        axes[1].plot(result.seasonal, label='Seasonal')
        axes[1].legend(loc='best')

    if add_residual:
        residual_col_name = f"{column_name}_residual"
        df[residual_col_name] = None
        df.loc[df.index.isin(series.index), residual_col_name] = result.resid.values
        axes[2].plot(result.resid, label='Residual')
        axes[2].legend(loc='best')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()
    
    return df


def add_lag_features(df, column_name, lags):
    
    for lag in lags:
        lag_col_name = f"{column_name}_lag{lag}"
        df[lag_col_name] = df[column_name].shift(lag)
    return df


def add_rolling_statistics_features(df, column_name, window_sizes, statistics):
    
    """
    This function adds rolling statistical features (mean, variance, standard deviation, minimum, and maximum) for the specified window sizes.
    """
  
    for window_size in window_sizes:
        for stat in statistics:
            stat_col_name = f"{column_name}_rolling_{stat}_{window_size}"
            if stat == 'mean':
                df[stat_col_name] = df[column_name].rolling(window=window_size).mean()
            elif stat == 'var':
                df[stat_col_name] = df[column_name].rolling(window=window_size).var()
            elif stat == 'std':
                df[stat_col_name] = df[column_name].rolling(window=window_size).std()
            elif stat == 'min':
                df[stat_col_name] = df[column_name].rolling(window=window_size).min()
            elif stat == 'max':
                df[stat_col_name] = df[column_name].rolling(window=window_size).max()
            else:
                raise ValueError(f"Statistic '{stat}' not recognized. Please choose from 'mean', 'var', 'std', 'min', 'max'.")
                
    return df

"""""""""""""""""
FOURIER FEATURES
"""""""""""""""""

def add_fourier_features(df, periods, harmonics, waves_to_add):
    
    """
    This function adds Fourier series features (sine and cosine components) for specified periods and harmonics.
    """

    for period in periods:
        for k in harmonics:
            if 'sin' in waves_to_add:
                df[f'sin_{period}_{k}'] = np.sin((2 * np.pi * k * df.index) / period)
            if 'cos' in waves_to_add:
                df[f'cos_{period}_{k}'] = np.cos((2 * np.pi * k * df.index) / period)

    return df