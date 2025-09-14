"""
Mediation analysis for Google as mediator between social channels and revenue.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm


def fit_mediator_model(df, predictor_features, mediator_col='google_spend'):
    """
    Fit Stage 1 model: Mediator ~ Predictors (Social channels -> Google spend).
    
    Args:
        df: DataFrame with features
        predictor_features: List of social media feature names
        mediator_col: Name of mediator variable (Google spend)
        
    Returns:
        Fitted model and DataFrame with residuals
    """
    # Prepare features
    X_predictors = sm.add_constant(df[predictor_features])
    y_mediator = df[mediator_col]
    
    # Fit OLS model
    stage1_model = sm.OLS(y_mediator, X_predictors).fit()
    
    # Generate predictions and residuals
    df_with_residuals = df.copy()
    df_with_residuals[f'{mediator_col}_predicted'] = stage1_model.predict(X_predictors)
    df_with_residuals[f'{mediator_col}_residual'] = y_mediator - df_with_residuals[f'{mediator_col}_predicted']
    
    print("=== Stage 1: Social Media -> Google Spend ===")
    print(stage1_model.summary())
    
    return stage1_model, df_with_residuals


def fit_outcome_model(df, predictor_features, mediator_residual_col, 
                     control_features, target_col='revenue'):
    """
    Fit Stage 2 model: Outcome ~ Predictors + Mediator_residual + Controls.
    
    Args:
        df: DataFrame with all features including residuals
        predictor_features: List of social media features
        mediator_residual_col: Name of mediator residual column
        control_features: List of control variables
        target_col: Name of target variable
        
    Returns:
        Feature names and target for downstream modeling
    """
    # Combine all features
    all_features = predictor_features + [mediator_residual_col] + control_features
    
    print(f"=== Stage 2 Features ===")
    print(f"Social media features: {len(predictor_features)}")
    print(f"Mediator residual: {mediator_residual_col}")
    print(f"Control features: {len(control_features)}")
    print(f"Total features: {len(all_features)}")
    
    return all_features


def compute_mediation_effects(stage1_model, stage2_model, social_features, 
                             mediator_predicted_col='google_spend_predicted'):
    """
    Compute direct, indirect, and total effects for mediation analysis.
    
    Args:
        stage1_model: Fitted Stage 1 model (Social -> Google)
        stage2_model: Fitted Stage 2 model (Social + Google_resid -> Revenue)
        social_features: List of social media feature names
        mediator_predicted_col: Name of predicted mediator column
        
    Returns:
        DataFrame with mediation effects
    """
    stage1_coefs = stage1_model.params
    
    # Extract coefficients from stage2_model
    if hasattr(stage2_model, 'coef_'):
        # For sklearn models, need to map coefficients to feature names
        feature_names = stage2_model.feature_names_in_
        stage2_coefs = dict(zip(feature_names, stage2_model.coef_))
    else:
        # For statsmodels
        stage2_coefs = stage2_model.params
    
    mediation_results = []
    
    for social_feature in social_features:
        # Direct effect (Social -> Revenue)
        direct_effect = stage2_coefs.get(social_feature, 0)
        
        # Indirect effect (Social -> Google -> Revenue)
        social_to_google = stage1_coefs.get(social_feature, 0)
        google_to_revenue = stage2_coefs.get(mediator_predicted_col, 0)
        indirect_effect = social_to_google * google_to_revenue
        
        # Total effect
        total_effect = direct_effect + indirect_effect
        
        mediation_results.append({
            'channel': social_feature,
            'direct_effect': direct_effect,
            'indirect_effect': indirect_effect, 
            'total_effect': total_effect,
            'mediation_ratio': indirect_effect / total_effect if total_effect != 0 else 0
        })
    
    return pd.DataFrame(mediation_results)


def analyze_causal_paths(df, social_features, mediator_col='google_spend', 
                        target_col='revenue', control_features=None):
    """
    Complete mediation analysis workflow.
    
    Args:
        df: DataFrame with all variables
        social_features: List of social media features
        mediator_col: Mediator variable name
        target_col: Target variable name
        control_features: List of control variables
        
    Returns:
        Dictionary with all mediation analysis results
    """
    if control_features is None:
        control_features = []
    
    # Stage 1: Social -> Google
    stage1_model, df_with_residuals = fit_mediator_model(
        df, social_features, mediator_col
    )
    
    # Stage 2: Prepare features for ML model
    mediator_residual_col = f'{mediator_col}_residual'
    all_features = fit_outcome_model(
        df_with_residuals, social_features, mediator_residual_col, 
        control_features, target_col
    )
    
    return {
        'stage1_model': stage1_model,
        'processed_data': df_with_residuals,
        'stage2_features': all_features,
        'mediator_residual_col': mediator_residual_col
    }