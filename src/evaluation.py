"""
Model evaluation and business insights for MMM analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import joblib


def create_prediction_plots(dates, y_actual, y_predicted, save_path=None, title="Model Performance"):
    """
    Create prediction vs actual plots.
    
    Args:
        dates: Date values
        y_actual: Actual target values
        y_predicted: Predicted values
        save_path: Path to save plot
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Time series plot
    ax1.plot(dates, y_actual, label="Actual", color="black", linewidth=2)
    ax1.plot(dates, y_predicted, label="Predicted", color="red", alpha=0.8)
    ax1.set_title(f"{title} - Time Series")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Scatter plot
    ax2.scatter(y_actual, y_predicted, alpha=0.6, s=30)
    ax2.plot([y_actual.min(), y_actual.max()], 
             [y_actual.min(), y_actual.max()], 
             color='red', linestyle='--', alpha=0.8)
    ax2.set_xlabel("Actual Revenue")
    ax2.set_ylabel("Predicted Revenue")
    ax2.set_title(f"{title} - Predicted vs Actual")
    ax2.grid(True, alpha=0.3)
    
    # Add R² to scatter plot
    r2 = r2_score(y_actual, y_predicted)
    ax2.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax2.transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_residual_analysis(dates, residuals, save_path=None):
    """
    Create residual analysis plots.
    
    Args:
        dates: Date values
        residuals: Model residuals
        save_path: Path to save plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Residuals over time
    ax1.plot(dates, residuals, color="blue", alpha=0.7)
    ax1.axhline(0, color="red", linestyle="--", alpha=0.8)
    ax1.set_title("Residuals Over Time")
    ax1.set_ylabel("Residuals")
    ax1.grid(True, alpha=0.3)
    
    # Residual distribution
    ax2.hist(residuals, bins=30, alpha=0.7, color="blue", edgecolor="black")
    ax2.axvline(0, color="red", linestyle="--", alpha=0.8)
    ax2.set_title("Residual Distribution")
    ax2.set_xlabel("Residuals")
    ax2.set_ylabel("Frequency")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def rolling_window_validation(df, model, feature_columns, target_col="revenue", 
                             date_col="week", window_size_ratio=0.7, 
                             test_size_ratio=0.1, step_weeks=4, save_path=None):
    """
    Perform rolling window validation for time series stability.
    
    Args:
        df: DataFrame with all data
        model: Trained model object
        feature_columns: List of feature column names
        target_col: Target column name
        date_col: Date column name
        window_size_ratio: Training window size as ratio of total
        test_size_ratio: Test size as ratio of total
        step_weeks: Number of weeks to step forward
        save_path: Path to save results
        
    Returns:
        DataFrame with rolling validation results
    """
    X = df[feature_columns].values
    y = df[target_col].values
    dates = df[date_col].values
    
    train_size = int(len(df) * window_size_ratio)
    test_size = int(len(df) * test_size_ratio)
    
    rolling_results = []
    
    for start_idx in range(0, len(df) - (train_size + test_size), step_weeks):
        train_indices = range(start_idx, start_idx + train_size)
        test_indices = range(start_idx + train_size, start_idx + train_size + test_size)
        
        # Extract training and test data
        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]
        
        # Clone and train model
        model_clone = joblib.load("temp_model.pkl") if hasattr(model, 'save') else model
        model_clone.fit(X_train, y_train)
        
        # Predict and evaluate
        y_pred = model_clone.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        
        rolling_results.append({
            "start_week": dates[train_indices[0]],
            "end_week": dates[test_indices[-1]],
            "r2_score": r2,
            "mape": mape,
            "train_size": len(train_indices),
            "test_size": len(test_indices)
        })
    
    results_df = pd.DataFrame(rolling_results)
    
    if save_path:
        results_df.to_csv(save_path, index=False)
    
    return results_df


def calculate_marketing_roas(df, model, feature_columns, media_channels, 
                           spend_increase_pct=0.05, base_revenue_col="revenue"):
    """
    Calculate Return on Ad Spend (ROAS) for each marketing channel.
    
    Args:
        df: DataFrame with processed features
        model: Trained model
        feature_columns: List of all feature columns
        media_channels: List of media channel names (original spend columns)
        spend_increase_pct: Percentage increase in spend for ROAS calculation
        base_revenue_col: Base revenue column name
        
    Returns:
        DataFrame with ROAS calculations
    """
    # Get baseline prediction
    X_baseline = df[feature_columns]
    baseline_revenue = model.predict(X_baseline).mean()
    
    roas_results = []
    
    for channel in media_channels:
        # Find corresponding transformed feature
        transformed_feature = f"{channel}_transformed"
        
        if transformed_feature in feature_columns:
            # Create scenario with increased spend
            df_scenario = df.copy()
            df_scenario[transformed_feature] *= (1 + spend_increase_pct)
            
            # Predict with increased spend
            X_scenario = df_scenario[feature_columns]
            scenario_revenue = model.predict(X_scenario).mean()
            
            # Calculate incremental revenue and spend
            incremental_revenue = scenario_revenue - baseline_revenue
            incremental_spend = df[channel.replace("_transformed", "")].mean() * spend_increase_pct
            
            # Calculate ROAS
            marginal_roas = incremental_revenue / incremental_spend if incremental_spend > 0 else 0
            
            roas_results.append({
                "channel": channel,
                "baseline_revenue": baseline_revenue,
                "scenario_revenue": scenario_revenue,
                "incremental_revenue": incremental_revenue,
                "incremental_spend": incremental_spend,
                "marginal_roas": marginal_roas
            })
    
    return pd.DataFrame(roas_results)


def analyze_price_sensitivity(df, model, feature_columns, price_col="average_price",
                             price_changes=[-0.1, -0.05, 0.05, 0.1]):
    """
    Analyze price sensitivity and elasticity.
    
    Args:
        df: DataFrame with features
        model: Trained model
        feature_columns: List of feature columns
        price_col: Price column name
        price_changes: List of price change percentages
        
    Returns:
        DataFrame with price sensitivity results
    """
    X_baseline = df[feature_columns]
    baseline_revenue = model.predict(X_baseline).mean()
    baseline_price = df[price_col].mean()
    
    sensitivity_results = []
    
    for price_change in price_changes:
        df_scenario = df.copy()
        df_scenario[price_col] *= (1 + price_change)
        
        X_scenario = df_scenario[feature_columns]
        scenario_revenue = model.predict(X_scenario).mean()
        
        revenue_change_pct = (scenario_revenue - baseline_revenue) / baseline_revenue
        price_elasticity = revenue_change_pct / price_change if price_change != 0 else 0
        
        sensitivity_results.append({
            "price_change_pct": price_change,
            "revenue_change_pct": revenue_change_pct,
            "price_elasticity": price_elasticity,
            "baseline_revenue": baseline_revenue,
            "scenario_revenue": scenario_revenue
        })
    
    return pd.DataFrame(sensitivity_results)


def analyze_promotion_impact(df, model, feature_columns, promotion_col="promotions"):
    """
    Analyze promotion impact on revenue.
    
    Args:
        df: DataFrame with features
        model: Trained model
        feature_columns: List of feature columns
        promotion_col: Promotion column name
        
    Returns:
        Dictionary with promotion analysis results
    """
    # Scenario without promotions
    df_no_promo = df.copy()
    df_no_promo[promotion_col] = 0
    X_no_promo = df_no_promo[feature_columns]
    revenue_no_promo = model.predict(X_no_promo).mean()
    
    # Scenario with promotions
    df_with_promo = df.copy()
    df_with_promo[promotion_col] = 1
    X_with_promo = df_with_promo[feature_columns]
    revenue_with_promo = model.predict(X_with_promo).mean()
    
    # Calculate promotion lift
    promotion_lift_pct = (revenue_with_promo - revenue_no_promo) / revenue_no_promo
    
    return {
        "revenue_no_promotion": revenue_no_promo,
        "revenue_with_promotion": revenue_with_promo,
        "promotion_lift_pct": promotion_lift_pct,
        "incremental_revenue": revenue_with_promo - revenue_no_promo
    }


def create_coefficient_importance_plot(coefficients_df, top_n=15, save_path=None):
    """
    Create coefficient/importance visualization.
    
    Args:
        coefficients_df: DataFrame with feature coefficients
        top_n: Number of top features to show
        save_path: Path to save plot
    """
    # Get top features by absolute coefficient value
    top_features = coefficients_df.head(top_n)
    
    plt.figure(figsize=(10, 8))
    colors = ['red' if x < 0 else 'blue' for x in top_features['coefficient']]
    
    bars = plt.barh(range(len(top_features)), top_features['coefficient'], color=colors, alpha=0.7)
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Coefficient Value')
    plt.title(f'Top {top_n} Feature Coefficients')
    plt.axvline(0, color='black', linestyle='-', alpha=0.5)
    plt.grid(True, alpha=0.3)
    
    # Invert y-axis to show highest coefficients at top
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_business_summary_report(model_metrics, roas_df, price_sensitivity_df, 
                                 promotion_results, coefficients_df):
    """
    Create comprehensive business summary report.
    
    Args:
        model_metrics: Model performance metrics
        roas_df: ROAS analysis results
        price_sensitivity_df: Price sensitivity results
        promotion_results: Promotion impact results
        coefficients_df: Model coefficients
        
    Returns:
        Formatted summary string
    """
    report = f"""
=== MARKETING MIX MODEL - BUSINESS SUMMARY ===

MODEL PERFORMANCE:
- R² Score: {model_metrics['r2_score']:.3f}
- MAPE: {model_metrics['mape']:.2%}
- Model explains {model_metrics['r2_score']*100:.1f}% of revenue variance

TOP REVENUE DRIVERS:
"""
    
    # Add top 5 coefficients
    top_drivers = coefficients_df.head(5)
    for _, row in top_drivers.iterrows():
        report += f"- {row['feature']}: {row['coefficient']:.2f}\n"
    
    report += f"""
MARKETING CHANNEL ROAS:
"""
    
    # Add ROAS for each channel
    for _, row in roas_df.iterrows():
        report += f"- {row['channel']}: ${row['marginal_roas']:.2f} per $1 spent\n"
    
    report += f"""
PRICE ELASTICITY:
- Price elasticity at 5% increase: {price_sensitivity_df[price_sensitivity_df['price_change_pct']==0.05]['price_elasticity'].iloc[0]:.2f}
- Price elasticity at 10% increase: {price_sensitivity_df[price_sensitivity_df['price_change_pct']==0.1]['price_elasticity'].iloc[0]:.2f}

PROMOTION IMPACT:
- Revenue lift from promotions: {promotion_results['promotion_lift_pct']:.2%}
- Incremental revenue: ${promotion_results['incremental_revenue']:.2f}

RECOMMENDATIONS:
1. Focus investment on highest ROAS channels
2. Monitor price elasticity for optimal pricing strategy
3. Strategic use of promotions for revenue boost
4. Continue monitoring model performance with rolling validation
"""
    
    return report