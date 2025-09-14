"""
Feature engineering utilities for MMM analysis.
"""

import numpy as np
import pandas as pd


def apply_adstock_transformation(media_data, decay_rate=0.6, max_lags=8):
    """
    Apply adstock transformation to media data to capture carryover effects.
    
    Args:
        media_data: Array-like media spend data
        decay_rate: Decay parameter (0-1), higher = more carryover
        max_lags: Maximum number of lags to consider
        
    Returns:
        Adstocked media data
    """
    media_array = np.array(media_data)
    adstocked_data = np.zeros_like(media_array)
    
    for i in range(len(media_array)):
        for lag in range(0, min(max_lags + 1, i + 1)):
            adstocked_data[i] += media_array[i - lag] * (decay_rate ** lag)
    
    return adstocked_data


def apply_saturation_curve(media_data, saturation_point):
    """
    Apply saturation curve to capture diminishing returns.
    
    Args:
        media_data: Array-like adstocked media data
        saturation_point: Half-saturation point
        
    Returns:
        Saturated media data
    """
    return media_data / (media_data + saturation_point)


def create_seasonality_features(df, date_col='week'):
    """
    Create trend and seasonality features.
    
    Args:
        df: DataFrame with date column
        date_col: Name of date column
        
    Returns:
        DataFrame with additional features
    """
    df_processed = df.copy()
    
    # Linear trend
    df_processed['trend'] = np.arange(len(df_processed))
    
    # Week of year
    df_processed['week_of_year'] = df_processed[date_col].dt.isocalendar().week.astype(int)
    
    # Seasonal components
    df_processed['sin_yearly'] = np.sin(2 * np.pi * df_processed['week_of_year'] / 52)
    df_processed['cos_yearly'] = np.cos(2 * np.pi * df_processed['week_of_year'] / 52)
    df_processed['sin_biannual'] = np.sin(2 * np.pi * df_processed['week_of_year'] / 26)
    
    return df_processed


def process_media_channels(df, media_channels, decay_rate=0.6, max_lags=8):
    """
    Process all media channels with adstock and saturation.
    
    Args:
        df: DataFrame containing media spend data
        media_channels: List of media channel column names
        decay_rate: Adstock decay parameter
        max_lags: Maximum adstock lags
        
    Returns:
        DataFrame with processed media features
    """
    df_processed = df.copy()
    
    # Handle zero values with forward fill
    for channel in media_channels:
        df_processed[channel] = df_processed[channel].replace(0, np.nan).ffill().fillna(0)
    
    # Apply adstock and saturation
    for channel in media_channels:
        # Adstock transformation
        adstock_col = f"{channel}_adstocked"
        df_processed[adstock_col] = apply_adstock_transformation(
            df_processed[channel], decay_rate, max_lags
        )
        
        # Calculate saturation point (70th percentile of non-zero values)
        non_zero_values = df_processed[adstock_col][df_processed[adstock_col] > 0]
        saturation_point = np.percentile(non_zero_values, 70) if len(non_zero_values) > 0 else 1
        
        # Apply saturation
        saturated_col = f"{channel}_transformed"
        df_processed[saturated_col] = apply_saturation_curve(df_processed[adstock_col], saturation_point)
    
    return df_processed


def prepare_model_features(df, media_channels, control_features):
    """
    Prepare final feature set for modeling.
    
    Args:
        df: DataFrame with all processed features
        media_channels: List of original media channel names
        control_features: List of control variable names
        
    Returns:
        Lists of feature names for different model components
    """
    # Social media transformed features (excluding Google)
    social_media_features = [
        f"{channel}_transformed" for channel in media_channels 
        if 'google' not in channel.lower()
    ]
    
    # All transformed media features
    media_transformed_features = [f"{channel}_transformed" for channel in media_channels]
    
    return social_media_features, media_transformed_features, control_features