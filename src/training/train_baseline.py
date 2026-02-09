"""
Train baseline Matrix Factorization model.

This script trains and evaluates the collaborative filtering baseline
using SVD-based matrix factorization.
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from typing import Dict

import numpy as np
from scipy import sparse

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess import load_processed_data
from src.models.matrix_factorization import MatrixFactorization
from src.evaluation.metrics import (
    compute_precision_at_k,
    compute_recall_at_k,
    compute_ndcg_at_k,
    compute_mrr
)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_baseline(
    data_path: str,
    model_save_path: str,
    embedding_dim: int = 64,
    method: str = 'svd'
) -> MatrixFactorization:
    """
    Train baseline matrix factorization model.
    
    Args:
        data_path: Path to processed data directory
        model_save_path: Path to save trained model
        embedding_dim: Embedding dimension
        method: 'svd' or 'ncf'
        
    Returns:
        Trained MatrixFactorization model
    """
    print("=" * 50)
    print("Training Baseline Model (Matrix Factorization)")
    print("=" * 50)
    
    # Load data
    print("\nLoading processed data...")
    data = load_processed_data(data_path)
    
    train_matrix = data['train_matrix']
    val_df = data['val_df']
    n_users = data['n_users']
    n_items = data['n_items']
    
    print(f"Users: {n_users}, Items: {n_items}")
    print(f"Training interactions: {train_matrix.nnz}")
    
    # Initialize model
    print(f"\nInitializing {method.upper()} model with {embedding_dim} factors...")
    model = MatrixFactorization(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=embedding_dim,
        method=method
    )
    
    # Train
    print("\nTraining model...")
    if method == 'svd':
        model.fit(train_matrix)
    else:
        model.fit(
            train_matrix,
            train_df=data['train_df'],
            epochs=50,
            lr=0.001
        )
    
    # Save model
    model_dir = Path(model_save_path).parent
    model_dir.mkdir(parents=True, exist_ok=True)
    
    if method == 'svd':
        model.save(model_save_path + '.npz')
    else:
        model.save(model_save_path + '.pt')
    
    print(f"\nModel saved to {model_save_path}")
    
    return model


def evaluate_baseline(
    model: MatrixFactorization,
    data: Dict,
    k_values: list = [5, 10, 20]
) -> Dict:
    """
    Evaluate baseline model.
    
    Args:
        model: Trained model
        data: Data dictionary
        k_values: List of K values for metrics
        
    Returns:
        Dictionary of evaluation metrics
    """
    print("\n" + "=" * 50)
    print("Evaluating Baseline Model")
    print("=" * 50)
    
    test_df = data['test_df']
    train_df = data['train_df']
    n_users = data['n_users']
    
    # Group test interactions by user
    test_by_user = test_df.groupby('user_idx')['item_idx'].apply(list).to_dict()
    train_by_user = train_df.groupby('user_idx')['item_idx'].apply(list).to_dict()
    
    # Evaluate
    all_precisions = {k: [] for k in k_values}
    all_recalls = {k: [] for k in k_values}
    all_ndcgs = {k: [] for k in k_values}
    all_mrrs = []
    
    sample_users = list(test_by_user.keys())[:500]  # Sample for speed
    
    print(f"\nEvaluating on {len(sample_users)} users...")
    
    for user_idx in sample_users:
        if user_idx not in train_by_user:
            continue
            
        actual_items = set(test_by_user[user_idx])
        exclude_items = train_by_user.get(user_idx, [])
        
        # Get recommendations
        max_k = max(k_values)
        recommended = model.recommend(user_idx, n_items=max_k, exclude_items=exclude_items)
        
        # Compute metrics
        for k in k_values:
            rec_k = recommended[:k]
            
            precision = compute_precision_at_k(rec_k, actual_items, k)
            recall = compute_recall_at_k(rec_k, actual_items, k)
            ndcg = compute_ndcg_at_k(rec_k, actual_items, k)
            
            all_precisions[k].append(precision)
            all_recalls[k].append(recall)
            all_ndcgs[k].append(ndcg)
        
        mrr = compute_mrr(recommended, actual_items)
        all_mrrs.append(mrr)
    
    # Aggregate results
    results = {}
    
    print("\n" + "-" * 40)
    print("Results:")
    print("-" * 40)
    
    for k in k_values:
        precision = np.mean(all_precisions[k])
        recall = np.mean(all_recalls[k])
        ndcg = np.mean(all_ndcgs[k])
        
        results[f'precision@{k}'] = precision
        results[f'recall@{k}'] = recall
        results[f'ndcg@{k}'] = ndcg
        
        print(f"Precision@{k}: {precision:.4f}")
        print(f"Recall@{k}: {recall:.4f}")
        print(f"NDCG@{k}: {ndcg:.4f}")
        print()
    
    mrr = np.mean(all_mrrs)
    results['mrr'] = mrr
    print(f"MRR: {mrr:.4f}")
    
    return results


def main():
    """Main function to train and evaluate baseline."""
    parser = argparse.ArgumentParser(description='Train baseline model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--data-path', type=str, default='data/processed',
                        help='Path to processed data')
    parser.add_argument('--model-path', type=str, default='results/models/baseline',
                        help='Path to save model')
    parser.add_argument('--embedding-dim', type=int, default=64,
                        help='Embedding dimension')
    parser.add_argument('--method', type=str, default='svd',
                        choices=['svd', 'ncf'],
                        help='MF method to use')
    
    args = parser.parse_args()
    
    # Load config if exists
    config_path = PROJECT_ROOT / args.config
    if config_path.exists():
        config = load_config(str(config_path))
        embedding_dim = config.get('model', {}).get('user_embedding_dim', args.embedding_dim)
    else:
        embedding_dim = args.embedding_dim
    
    # Train
    data_path = PROJECT_ROOT / args.data_path
    model_path = PROJECT_ROOT / args.model_path
    
    model = train_baseline(
        data_path=str(data_path),
        model_save_path=str(model_path),
        embedding_dim=embedding_dim,
        method=args.method
    )
    
    # Evaluate
    data = load_processed_data(str(data_path))
    results = evaluate_baseline(model, data)
    
    # Save results
    results_path = PROJECT_ROOT / 'results' / 'baseline_results.yaml'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, 'w') as f:
        yaml.dump(results, f)
    
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
