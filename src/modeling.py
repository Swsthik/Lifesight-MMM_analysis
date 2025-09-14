# ML model training and evaluation
"""
Machine learning models for MMM analysis.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import joblib


def create_elasticnet_model(cv_folds=5, l1_ratios=None, n_alphas=50, random_state=42):
    """
    Create ElasticNet model with cross-validation.
    
    Args:
        cv_folds: Number of CV folds
        l1_ratios: L1 ratios to try
        n_alphas: Number of alpha values
        random_state: Random seed
        
    Returns:
        Pipeline with scaler and ElasticNet
    """
    if l1_ratios is None:
        l1_ratios = [0.1, 0.5, 0.9]
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('elasticnet', ElasticNetCV(
            cv=cv_folds,
            l1_ratio=l1_ratios,
            n_alphas=n_alphas,
            random_state=random_state,
            max_iter=10000
        ))
    ])
    
    return pipeline


def create_random_forest_model(n_estimators=300, max_depth=8, random_state=42):
    """
    Create Random Forest model.
    
    Args:
        n_estimators: Number of trees
        max_depth: Maximum depth
        random_state: Random seed
        
    Returns:
        RandomForestRegressor
    """
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )


def train_model_with_timeseries_cv(model, X, y, n_splits=5):
    """
    Train model using time series cross-validation.
    
    Args:
        model: Sklearn model or pipeline
        X: Features
        y: Target
        n_splits: Number of time series splits
        
    Returns:
        Trained model and out-of-fold predictions
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof_predictions = np.zeros(len(y))
    oof_indices = np.zeros(len(y), dtype=bool)
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Predict
        val_predictions = model.predict(X_val)
        oof_predictions[val_idx] = val_predictions
        oof_indices[val_idx] = True
        
        # Log fold performance
        fold_r2 = r2_score(y_val, val_predictions)
        print(f"Fold {fold + 1} R²: {fold_r2:.4f}")
    
    # Final fit on all data
    model.fit(X, y)
    
    return model, oof_predictions, oof_indices


def evaluate_model_performance(y_true, y_pred, model_name="Model"):
    """
    Evaluate model performance with multiple metrics.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        model_name: Name for reporting
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'r2_score': r2_score(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mape': mean_absolute_percentage_error(y_true, y_pred),
        'mae': np.mean(np.abs(y_true - y_pred))
    }
    
    print(f"\n=== {model_name} Performance ===")
    print(f"R² Score: {metrics['r2_score']:.4f}")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"MAPE: {metrics['mape']:.2%}")
    print(f"MAE: {metrics['mae']:.2f}")
    
    return metrics


def extract_model_coefficients(model, feature_names):
    """
    Extract and format model coefficients.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        
    Returns:
        DataFrame with coefficients
    """
    if hasattr(model, 'named_steps'):
        # Pipeline model
        if 'elasticnet' in model.named_steps:
            coefs = model.named_steps['elasticnet'].coef_
        else:
            raise ValueError("Unknown pipeline structure")
    elif hasattr(model, 'coef_'):
        # Direct model with coefficients
        coefs = model.coef_
    elif hasattr(model, 'feature_importances_'):
        # Tree-based model
        coefs = model.feature_importances_
    else:
        raise ValueError("Model type not supported for coefficient extraction")
    
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefs
    }).sort_values('coefficient', key=abs, ascending=False)
    
    return coef_df


def save_model_artifacts(model, model_metrics, coefficients_df, 
                        predictions_df, save_dir):
    """
    Save all model artifacts.
    
    Args:
        model: Trained model
        model_metrics: Performance metrics
        coefficients_df: Model coefficients
        predictions_df: Predictions and residuals
        save_dir: Directory to save artifacts
    """
    import os
    os.makedirs(f"{save_dir}/models", exist_ok=True)
    os.makedirs(f"{save_dir}/tables", exist_ok=True)
    
    # Save model
    joblib.dump(model, f"{save_dir}/models/trained_model.pkl")
    
    # Save metrics
    metrics_df = pd.DataFrame([model_metrics])
    metrics_df.to_csv(f"{save_dir}/tables/model_metrics.csv", index=False)
    
    # Save coefficients
    coefficients_df.to_csv(f"{save_dir}/tables/model_coefficients.csv", index=False)
    
    # Save predictions
    predictions_df.to_csv(f"{save_dir}/tables/model_predictions.csv", index=False)
    
    print(f"Model artifacts saved to {save_dir}")


def calculate_feature_importance_scores(model, X, y, feature_names, n_permutations=10):
    """
    Calculate permutation feature importance.
    
    Args:
        model: Trained model
        X: Feature matrix
        y: Target values  
        feature_names: List of feature names
        n_permutations: Number of permutation rounds
        
    Returns:
        DataFrame with importance scores
    """
    from sklearn.inspection import permutation_importance
    
    # Calculate permutation importance
    perm_importance = permutation_importance(
        model, X, y, n_repeats=n_permutations, random_state=42
    )
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    return importance_df