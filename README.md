# Marketing Mix Model with Causal Mediation Analysis

A comprehensive implementation of Marketing Mix Modeling (MMM) with explicit treatment of Google as a mediator between social media channels and revenue.

## 🎯 Project Overview

This project implements a two-stage marketing mix model that accounts for the causal relationship where social media advertising (Facebook, TikTok, Instagram, Snapchat) influences Google search spend, which then drives revenue. The model provides actionable insights for marketing budget allocation and strategic decision-making.

### Key Features
- **Causal Mediation Framework**: Two-stage modeling approach treating Google as mediator
- **Advanced Feature Engineering**: Adstock and saturation transformations for media channels
- **Time Series Validation**: Proper temporal validation preventing data leakage
- **Business Insights**: ROAS analysis, price elasticity, and promotion impact
- **Comprehensive Diagnostics**: Model stability and performance analysis

## 📁 Project Structure

```
mmm_project/
├── data/
│   └── weekly_data.csv              # Input dataset (2 years weekly data)
├── notebooks/
│   └── mmm_analysis.ipynb           # Main analysis notebook
├── src/
│   ├── __init__.py
│   ├── data_processing.py           # Data loading and cleaning utilities
│   ├── feature_engineering.py      # Adstock, saturation, seasonality features
│   ├── mediation_analysis.py       # Two-stage mediation modeling
│   ├── modeling.py                  # ML model training and evaluation
│   └── evaluation.py                # Business insights and diagnostics
├── results/
│   ├── models/                      # Trained model artifacts
│   ├── figures/                     # All visualizations
│   ├── tables/                      # Analysis results in CSV format
│   └── business_summary_report.txt  # Executive summary
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Clone the repository
git clone <repository-url>
cd mmm_project

# Create virtual environment
python -m venv mmm_env
source mmm_env/bin/activate  # On Windows: mmm_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation
Place your `weekly_data.csv` file in the `data/` directory. The dataset should contain:
- `week`: Date column
- `revenue`: Target variable
- Media spend columns: `facebook_spend`, `google_spend`, `tiktok_spend`, `instagram_spend`, `snapchat_spend`
- Control variables: `social_followers`, `average_price`, `promotions`, `emails_send`, `sms_send`

### 3. Run Analysis
```bash
cd notebooks
jupyter notebook mmm_analysis.ipynb
```

Run all cells sequentially to perform the complete analysis.

## 🧠 Methodology

### Causal Framework
The model assumes the following causal structure:
```
Social Media Channels → Google Spend → Revenue
                    ↘               ↗
                      → Revenue (direct)
```

### Two-Stage Approach

**Stage 1: Mediator Model**
```
Google Spend = α + β₁×Facebook + β₂×TikTok + β₃×Instagram + β₄×Snapchat + ε₁
```

**Stage 2: Outcome Model**
```
Revenue = γ + δ₁×Social_Channels + δ₂×Google_Residual + δ₃×Controls + ε₂
```

### Feature Engineering
- **Adstock Transformation**: Captures carryover effects with geometric decay
- **Saturation Curves**: Models diminishing returns using half-saturation points
- **Seasonality**: Sine/cosine transformations for weekly and bi-annual patterns
- **Trend**: Linear time trend component

### Model Selection
- **ElasticNet Regression**: Balances Ridge and Lasso regularization
- **Time Series CV**: 5-fold time series cross-validation
- **Hyperparameter Tuning**: Grid search over α and l1_ratio parameters

## 📊 Key Outputs

### 1. Model Performance
- R² Score and MAPE for model fit assessment
- Out-of-fold validation results
- Residual analysis and diagnostics

### 2. Marketing Insights
- **ROAS Analysis**: Revenue return per dollar spent by channel
- **Mediation Effects**: Direct vs. indirect effects through Google
- **Price Elasticity**: Revenue sensitivity to price changes
- **Promotion Impact**: Incremental revenue from promotional activities

### 3. Visualizations
- Executive dashboard with key metrics
- Channel performance comparisons
- Model fit and residual plots
- Business insight summaries

## 📈 Business Applications

### Strategic Planning
- **Budget Allocation**: Optimize spend across channels based on ROAS
- **Pricing Strategy**: Use elasticity insights for price optimization
- **Campaign Timing**: Leverage seasonality patterns

### Tactical Decisions
- **Channel Mix**: Balance direct and mediated effects
- **Promotion Planning**: Quantify promotional lift
- **Performance Monitoring**: Track model stability over time

## 🔧 Technical Implementation

### Data Processing
- Handles zero-spend periods with forward filling
- Robust outlier detection and treatment
- Missing value imputation strategies

### Feature Engineering
- Configurable adstock decay rates (default: 0.6)
- Adaptive saturation points (70th percentile)
- Multiple seasonality frequencies

### Model Validation
- Time-aware cross-validation
- Rolling window stability checks
- Durbin-Watson autocorrelation tests

## 📋 Results Interpretation

### Coefficient Analysis
- Positive coefficients indicate revenue-driving factors
- Magnitude reflects impact per unit increase
- Sign changes may indicate multicollinearity issues

### ROAS Thresholds
- ROAS > 1.0: Profitable channels
- ROAS < 1.0: Potentially unprofitable
- Consider confidence intervals for decision-making

### Mediation Ratios
- High ratios indicate strong Google mediation
- Low ratios suggest direct revenue impact
- Informs channel strategy and measurement

## ⚠️ Limitations & Considerations

### Model Assumptions
- Linear relationships between transformed features
- Stationary residuals assumption
- No unmeasured confounding variables

### Data Requirements
- Sufficient historical data (recommended: 2+ years)
- Consistent measurement across channels
- Regular data quality monitoring

### Business Context
- External factors not captured in model
- Competitive dynamics and market changes
- Incrementality vs. correlation considerations

## 🔄 Model Maintenance

### Regular Updates
- Retrain quarterly with new data
- Monitor coefficient stability
- Validate ROAS calculations with experiments

### Performance Monitoring
- Track prediction accuracy over time
- Check for model drift
- Update feature engineering as needed

## 📚 References

- Bayesian Methods for Media Mix Modeling (Google)
- Marketing Mix Modeling at Lyft (Lyft Engineering)
- Causal Inference in Marketing Mix Models (various academic sources)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make improvements with proper documentation
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Contact**: For questions or support, please open an issue in the repository.

**Version**: 1.0.0  
**Last Updated**: December 2024

![Project Flowchart](flowchart.png)