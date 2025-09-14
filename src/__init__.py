"""
Marketing Mix Model with Causal Mediation Analysis

A comprehensive MMM implementation with explicit treatment of Google as mediator
between social media channels and revenue.

Author: MMM Analytics Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "MMM Analytics Team"

# Import main modules for easy access
from . import data_processing
from . import feature_engineering
from . import mediation_analysis
from . import modeling
from . import evaluation

__all__ = [
    'data_processing',
    'feature_engineering', 
    'mediation_analysis',
    'modeling',
    'evaluation'
]