"""
Data preprocessing for MovieLens dataset.

This module handles:
- Loading raw data
- Creating user-item interaction matrices
- Train/val/test splits
- Converting ratings to implicit feedback
"""

import os
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    """Preprocessor for MovieLens dataset."""
    
    def __init__(
        self,
        min_user_interactions: int = 20,
        min_item_interactions: int = 10,
        test_ratio: float = 0.2,
        val_ratio: float = 0.1,
        random_state: int = 42
    ):
        """
        Initialize preprocessor.
        
        Args:
            min_user_interactions: Minimum interactions per user to keep
            min_item_interactions: Minimum interactions per item to keep
            test_ratio: Ratio of data for testing
            val_ratio: Ratio of training data for validation
            random_state: Random seed for reproducibility
        """
        self.min_user_interactions = min_user_interactions
        self.min_item_interactions = min_item_interactions
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.random_state = random_state
        
        # Mappings (will be populated during preprocessing)
        self.user_to_idx: Dict[int, int] = {}
        self.idx_to_user: Dict[int, int] = {}
        self.item_to_idx: Dict[int, int] = {}
        self.idx_to_item: Dict[int, int] = {}
        
        # Statistics
        self.n_users: int = 0
        self.n_items: int = 0
        
    def load_movielens_1m(self, data_path: str) -> pd.DataFrame:
        """
        Load MovieLens 1M dataset.
        
        Args:
            data_path: Path to the ml-1m directory
            
        Returns:
            DataFrame with columns: user_id, item_id, rating, timestamp
        """
        ratings_file = os.path.join(data_path, 'ratings.dat')
        
        # MovieLens 1M uses :: as separator
        df = pd.read_csv(
            ratings_file,
            sep='::',
            engine='python',
            names=['user_id', 'item_id', 'rating', 'timestamp'],
            encoding='latin-1'
        )
        
        print(f"Loaded {len(df)} ratings from {len(df['user_id'].unique())} users "
              f"and {len(df['item_id'].unique())} items")
        
        return df
    
    def load_movies_metadata(self, data_path: str) -> pd.DataFrame:
        """
        Load movie metadata.
        
        Args:
            data_path: Path to the ml-1m directory
            
        Returns:
            DataFrame with movie metadata
        """
        movies_file = os.path.join(data_path, 'movies.dat')
        
        df = pd.read_csv(
            movies_file,
            sep='::',
            engine='python',
            names=['item_id', 'title', 'genres'],
            encoding='latin-1'
        )
        
        return df
    
    def filter_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter users and items with too few interactions.
        
        Args:
            df: Raw ratings DataFrame
            
        Returns:
            Filtered DataFrame
        """
        print(f"Before filtering: {len(df)} interactions")
        
        # Iteratively filter until convergence
        prev_len = len(df) + 1
        while len(df) < prev_len:
            prev_len = len(df)
            
            # Filter users
            user_counts = df['user_id'].value_counts()
            valid_users = user_counts[user_counts >= self.min_user_interactions].index
            df = df[df['user_id'].isin(valid_users)]
            
            # Filter items
            item_counts = df['item_id'].value_counts()
            valid_items = item_counts[item_counts >= self.min_item_interactions].index
            df = df[df['item_id'].isin(valid_items)]
        
        print(f"After filtering: {len(df)} interactions")
        print(f"Users: {len(df['user_id'].unique())}, Items: {len(df['item_id'].unique())}")
        
        return df
    
    def create_mappings(self, df: pd.DataFrame) -> None:
        """
        Create user and item ID mappings to continuous indices.
        
        Args:
            df: Filtered ratings DataFrame
        """
        unique_users = sorted(df['user_id'].unique())
        unique_items = sorted(df['item_id'].unique())
        
        self.user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
        self.idx_to_user = {idx: uid for uid, idx in self.user_to_idx.items()}
        
        self.item_to_idx = {iid: idx for idx, iid in enumerate(unique_items)}
        self.idx_to_item = {idx: iid for iid, idx in self.item_to_idx.items()}
        
        self.n_users = len(unique_users)
        self.n_items = len(unique_items)
        
        print(f"Created mappings: {self.n_users} users, {self.n_items} items")
    
    def convert_to_implicit(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert ratings to implicit feedback signals.
        
        Rating → Feedback Conversion:
        - Rating >= 4: Purchase/High engagement (+5)
        - Rating == 3: Click with dwell (+2)
        - Rating <= 2: Skip (-1)
        
        Args:
            df: Ratings DataFrame
            
        Returns:
            DataFrame with feedback column
        """
        df = df.copy()
        
        def rating_to_feedback(rating):
            if rating >= 4:
                return 5  # Purchase
            elif rating == 3:
                return 2  # Click with dwell
            else:
                return -1  # Skip
        
        df['feedback'] = df['rating'].apply(rating_to_feedback)
        
        # Create binary interaction indicator
        df['interaction'] = (df['rating'] >= 3).astype(int)
        
        return df
    
    def split_data(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/val/test sets.
        
        Uses time-based splitting for more realistic evaluation.
        
        Args:
            df: Processed DataFrame
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Split by time
        n_total = len(df)
        test_start = int(n_total * (1 - self.test_ratio))
        val_start = int(test_start * (1 - self.val_ratio))
        
        train_df = df.iloc[:val_start]
        val_df = df.iloc[val_start:test_start]
        test_df = df.iloc[test_start:]
        
        print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def create_interaction_matrix(
        self,
        df: pd.DataFrame,
        use_feedback: bool = False
    ) -> sparse.csr_matrix:
        """
        Create sparse user-item interaction matrix.
        
        Args:
            df: DataFrame with user_idx, item_idx columns
            use_feedback: If True, use feedback values; else binary
            
        Returns:
            Sparse CSR matrix of shape (n_users, n_items)
        """
        user_indices = df['user_idx'].values
        item_indices = df['item_idx'].values
        
        if use_feedback:
            values = df['feedback'].values.astype(np.float32)
        else:
            values = np.ones(len(df), dtype=np.float32)
        
        matrix = sparse.csr_matrix(
            (values, (user_indices, item_indices)),
            shape=(self.n_users, self.n_items)
        )
        
        return matrix
    
    def preprocess(self, data_path: str, output_path: str) -> Dict:
        """
        Run full preprocessing pipeline.
        
        Args:
            data_path: Path to raw data directory
            output_path: Path to save processed data
            
        Returns:
            Dictionary with all processed data
        """
        # Load data
        print("Loading data...")
        df = self.load_movielens_1m(data_path)
        movies_df = self.load_movies_metadata(data_path)
        
        # Filter
        print("\nFiltering interactions...")
        df = self.filter_interactions(df)
        
        # Create mappings
        print("\nCreating mappings...")
        self.create_mappings(df)
        
        # Add index columns
        df['user_idx'] = df['user_id'].map(self.user_to_idx)
        df['item_idx'] = df['item_id'].map(self.item_to_idx)
        
        # Convert to implicit feedback
        print("\nConverting to implicit feedback...")
        df = self.convert_to_implicit(df)
        
        # Split data
        print("\nSplitting data...")
        train_df, val_df, test_df = self.split_data(df)
        
        # Create interaction matrices
        print("\nCreating interaction matrices...")
        train_matrix = self.create_interaction_matrix(train_df)
        val_matrix = self.create_interaction_matrix(val_df)
        test_matrix = self.create_interaction_matrix(test_df)
        
        # Prepare output
        output = {
            'train_df': train_df,
            'val_df': val_df,
            'test_df': test_df,
            'train_matrix': train_matrix,
            'val_matrix': val_matrix,
            'test_matrix': test_matrix,
            'movies_df': movies_df,
            'user_to_idx': self.user_to_idx,
            'idx_to_user': self.idx_to_user,
            'item_to_idx': self.item_to_idx,
            'idx_to_item': self.idx_to_item,
            'n_users': self.n_users,
            'n_items': self.n_items
        }
        
        # Save
        print(f"\nSaving to {output_path}...")
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save DataFrames as CSV
        train_df.to_csv(output_dir / 'train.csv', index=False)
        val_df.to_csv(output_dir / 'val.csv', index=False)
        test_df.to_csv(output_dir / 'test.csv', index=False)
        
        # Save matrices
        sparse.save_npz(output_dir / 'train_matrix.npz', train_matrix)
        sparse.save_npz(output_dir / 'val_matrix.npz', val_matrix)
        sparse.save_npz(output_dir / 'test_matrix.npz', test_matrix)
        
        # Save mappings
        with open(output_dir / 'mappings.pkl', 'wb') as f:
            pickle.dump({
                'user_to_idx': self.user_to_idx,
                'idx_to_user': self.idx_to_user,
                'item_to_idx': self.item_to_idx,
                'idx_to_item': self.idx_to_item,
                'n_users': self.n_users,
                'n_items': self.n_items
            }, f)
        
        print("Preprocessing complete!")
        
        return output


def preprocess_data(
    data_path: str,
    output_path: str,
    min_user_interactions: int = 20,
    min_item_interactions: int = 10
) -> Dict:
    """
    Convenience function to preprocess data.
    
    Args:
        data_path: Path to raw data
        output_path: Path to save processed data
        min_user_interactions: Minimum interactions per user
        min_item_interactions: Minimum interactions per item
        
    Returns:
        Dictionary with processed data
    """
    preprocessor = DataPreprocessor(
        min_user_interactions=min_user_interactions,
        min_item_interactions=min_item_interactions
    )
    return preprocessor.preprocess(data_path, output_path)


def load_processed_data(processed_path: str) -> Dict:
    """
    Load preprocessed data.
    
    Args:
        processed_path: Path to processed data directory
        
    Returns:
        Dictionary with all data
    """
    processed_dir = Path(processed_path)
    
    # Load DataFrames
    train_df = pd.read_csv(processed_dir / 'train.csv')
    val_df = pd.read_csv(processed_dir / 'val.csv')
    test_df = pd.read_csv(processed_dir / 'test.csv')
    
    # Load matrices
    train_matrix = sparse.load_npz(processed_dir / 'train_matrix.npz')
    val_matrix = sparse.load_npz(processed_dir / 'val_matrix.npz')
    test_matrix = sparse.load_npz(processed_dir / 'test_matrix.npz')
    
    # Load mappings
    with open(processed_dir / 'mappings.pkl', 'rb') as f:
        mappings = pickle.load(f)
    
    return {
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df,
        'train_matrix': train_matrix,
        'val_matrix': val_matrix,
        'test_matrix': test_matrix,
        **mappings
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess MovieLens data')
    parser.add_argument('--data-path', type=str, default='data/raw/ml-1m',
                        help='Path to raw data')
    parser.add_argument('--output-path', type=str, default='data/processed',
                        help='Path to save processed data')
    parser.add_argument('--min-user', type=int, default=20,
                        help='Minimum user interactions')
    parser.add_argument('--min-item', type=int, default=10,
                        help='Minimum item interactions')
    
    args = parser.parse_args()
    
    preprocess_data(
        args.data_path,
        args.output_path,
        args.min_user,
        args.min_item
    )
