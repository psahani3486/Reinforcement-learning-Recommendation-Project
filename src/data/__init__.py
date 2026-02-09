"""
Data module for RL Recommendation System.
Handles data download, preprocessing, and loading.
"""

from .download import download_movielens
from .preprocess import preprocess_data, load_processed_data

__all__ = [
    'download_movielens',
    'preprocess_data',
    'load_processed_data'
]
