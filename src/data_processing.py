"""
Data processing utilities for MMM analysis.
"""

import pandas as pd
import numpy as np


def load_and_validate_data(file_path, date_col='week'):
    """
    Load and validate the dataset.
    
    Args:
        file_path: Path to the CSV file
        date_col: Name of the date column
        
    Returns:
        Validated and sorted DataFrame
    """
    # Load data
    df = pd.read_csv(file_path, parse_dates=[date_col])
    
    # Sort by date
    df = df.sort_values(date_col).reset_index(drop=True)
    
    # Basic validation
    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
    print(f"Missing values:\n{df.isnull().sum()}")
    
    return df


def clean_media_data(df, media_channels):
    """
    Clean media spend data by handling zeros and outliers.
    
    Args:
        df: DataFrame containing media data
        media_channels: List of media channel column names
        
    Returns:
        DataFrame with cleaned media data
    """
    df_clean = df.copy()
    
    for channel in media_channels:
        # Log initial statistics
        print(f"\n{channel} - Zero values: {(df_clean[channel] == 0).sum()}")
        
        # Replace zeros with NaN and forward fill
        df_clean[channel] = df_clean[channel].replace(0, np.nan).ffill().fillna(0)
    
    return df_clean


def clean_control_variables(df, control_vars):
    """
    Clean control variables.
    
    Args:
        df: DataFrame containing control variables
        control_vars: List of control variable names
        
    Returns:
        DataFrame with cleaned control variables
    """
    df_clean = df.copy()
    
    for var in control_vars:
        if var in df_clean.columns:
            # Handle specific cleaning for each variable type
            if 'followers' in var.lower():
                df_clean[var] = df_clean[var].replace(0, np.nan).ffill().fillna(0)
            
            # Log statistics
            print(f"{var} - Range: [{df_clean[var].min():.2f}, {df_clean[var].max():.2f}]")
    
    return df_clean


def create_dataset_splits(df, date_col='week', train_ratio=0.7, val_ratio=0.15):
    """
    Create time-based train/validation/test splits.
    
    Args:
        df: DataFrame to split
        date_col: Date column name
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        
    Returns:
        Dictionary with train/val/test DataFrames and indices
    """
    n_total = len(df)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_idx = slice(0, n_train)
    val_idx = slice(n_train, n_train + n_val)
    test_idx = slice(n_train + n_val, n_total)
    
    splits = {
        'train': df.iloc[train_idx].copy(),
        'val': df.iloc[val_idx].copy(),
        'test': df.iloc[test_idx].copy(),
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx
    }
    
    print(f"Train: {len(splits['train'])} samples")
    print(f"Validation: {len(splits['val'])} samples") 
    print(f"Test: {len(splits['test'])} samples")
    
    return splits