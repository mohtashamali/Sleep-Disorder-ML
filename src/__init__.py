"""
Sleep Disorder Detection Package

A machine learning package for detecting sleep disorders using 
physiological and behavioral data.
"""

__version__ = "1.0.0"
__author__ = "Mohd Mohtasham Ali"
__email__ = "mohtashamali@example.com"

from .data_preprocessing import pre_processing
from .feature_engineering import feature_engineering
from .model import train_model

__all__ = [
    'pre_processing',
    'feature_engineering', 
    'train_model',
    '__version__',
    '__author__',
    '__email__'
]