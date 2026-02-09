"""
Evaluation metrics for recommendation systems.

This module implements standard recommendation metrics:
- Precision@K
- Recall@K
- NDCG@K
- MRR (Mean Reciprocal Rank)
- Hit Rate
- CTR (Click-Through Rate)
- Cumulative Reward
"""

import numpy as np
from typing import List, Set, Union, Optional


def compute_precision_at_k(
    recommended: Union[List[int], np.ndarray],
    relevant: Set[int],
    k: int
) -> float:
    """
    Compute Precision@K.
    
    Precision@K = |Recommended ∩ Relevant| / K
    
    Args:
        recommended: List of recommended item indices
        relevant: Set of relevant item indices
        k: Number of top items to consider
        
    Returns:
        Precision@K value
    """
    if k <= 0:
        return 0.0
    
    recommended_k = recommended[:k] if len(recommended) > k else recommended
    hits = sum(1 for item in recommended_k if item in relevant)
    
    return hits / k


def compute_recall_at_k(
    recommended: Union[List[int], np.ndarray],
    relevant: Set[int],
    k: int
) -> float:
    """
    Compute Recall@K.
    
    Recall@K = |Recommended ∩ Relevant| / |Relevant|
    
    Args:
        recommended: List of recommended item indices
        relevant: Set of relevant item indices
        k: Number of top items to consider
        
    Returns:
        Recall@K value
    """
    if len(relevant) == 0:
        return 0.0
    
    recommended_k = recommended[:k] if len(recommended) > k else recommended
    hits = sum(1 for item in recommended_k if item in relevant)
    
    return hits / len(relevant)


def compute_dcg_at_k(
    recommended: Union[List[int], np.ndarray],
    relevant: Set[int],
    k: int
) -> float:
    """
    Compute DCG@K (Discounted Cumulative Gain).
    
    DCG@K = Σ rel_i / log2(i + 2) for i in [0, k)
    
    Args:
        recommended: List of recommended item indices
        relevant: Set of relevant item indices
        k: Number of top items to consider
        
    Returns:
        DCG@K value
    """
    recommended_k = recommended[:k] if len(recommended) > k else recommended
    
    dcg = 0.0
    for i, item in enumerate(recommended_k):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)  # +2 because i starts at 0
    
    return dcg


def compute_ndcg_at_k(
    recommended: Union[List[int], np.ndarray],
    relevant: Set[int],
    k: int
) -> float:
    """
    Compute NDCG@K (Normalized Discounted Cumulative Gain).
    
    NDCG@K = DCG@K / IDCG@K
    
    Args:
        recommended: List of recommended item indices
        relevant: Set of relevant item indices
        k: Number of top items to consider
        
    Returns:
        NDCG@K value in [0, 1]
    """
    dcg = compute_dcg_at_k(recommended, relevant, k)
    
    # IDCG: DCG of ideal ranking (all relevant items first)
    ideal_k = min(k, len(relevant))
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_k))
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def compute_mrr(
    recommended: Union[List[int], np.ndarray],
    relevant: Set[int]
) -> float:
    """
    Compute MRR (Mean Reciprocal Rank).
    
    MRR = 1 / rank of first relevant item
    
    Args:
        recommended: List of recommended item indices
        relevant: Set of relevant item indices
        
    Returns:
        Reciprocal rank (0 if no relevant item found)
    """
    for i, item in enumerate(recommended):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0


def compute_hit_rate(
    recommended: Union[List[int], np.ndarray],
    relevant: Set[int],
    k: int
) -> float:
    """
    Compute Hit Rate@K.
    
    Hit Rate = 1 if any relevant item in top K, else 0
    
    Args:
        recommended: List of recommended item indices
        relevant: Set of relevant item indices
        k: Number of top items to consider
        
    Returns:
        1.0 if hit, 0.0 otherwise
    """
    recommended_k = recommended[:k] if len(recommended) > k else recommended
    return 1.0 if any(item in relevant for item in recommended_k) else 0.0


def compute_ctr(clicks: List[int], impressions: List[int]) -> float:
    """
    Compute Click-Through Rate.
    
    CTR = total clicks / total impressions
    
    Args:
        clicks: List of click counts
        impressions: List of impression counts
        
    Returns:
        CTR value
    """
    total_clicks = sum(clicks)
    total_impressions = sum(impressions)
    
    if total_impressions == 0:
        return 0.0
    
    return total_clicks / total_impressions


def compute_average_precision(
    recommended: Union[List[int], np.ndarray],
    relevant: Set[int]
) -> float:
    """
    Compute Average Precision (AP).
    
    AP = Σ (P@i * rel_i) / |Relevant|
    
    Args:
        recommended: List of recommended item indices
        relevant: Set of relevant item indices
        
    Returns:
        AP value
    """
    if len(relevant) == 0:
        return 0.0
    
    hits = 0
    sum_precisions = 0.0
    
    for i, item in enumerate(recommended):
        if item in relevant:
            hits += 1
            precision_at_i = hits / (i + 1)
            sum_precisions += precision_at_i
    
    return sum_precisions / len(relevant)


def compute_coverage(
    all_recommendations: List[List[int]],
    total_items: int
) -> float:
    """
    Compute Catalog Coverage.
    
    Coverage = |unique recommended items| / |total items|
    
    Args:
        all_recommendations: List of recommendation lists
        total_items: Total number of items in catalog
        
    Returns:
        Coverage value in [0, 1]
    """
    unique_items = set()
    for rec_list in all_recommendations:
        unique_items.update(rec_list)
    
    return len(unique_items) / total_items


def compute_diversity(
    recommended: Union[List[int], np.ndarray],
    item_embeddings: np.ndarray
) -> float:
    """
    Compute Intra-List Diversity (ILD).
    
    Diversity = average pairwise distance of recommended items
    
    Args:
        recommended: List of recommended item indices
        item_embeddings: Item embedding matrix
        
    Returns:
        Diversity score
    """
    if len(recommended) < 2:
        return 0.0
    
    embeddings = item_embeddings[recommended]
    
    # Compute pairwise cosine distances
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normalized = embeddings / norms
    
    similarities = np.dot(normalized, normalized.T)
    
    # Get upper triangle (excluding diagonal)
    n = len(recommended)
    distances = 1 - similarities
    upper_indices = np.triu_indices(n, k=1)
    
    return np.mean(distances[upper_indices])


def compute_novelty(
    recommended: Union[List[int], np.ndarray],
    item_popularity: np.ndarray
) -> float:
    """
    Compute Novelty (inverse popularity).
    
    Novelty = average of -log(popularity)
    
    Args:
        recommended: List of recommended item indices
        item_popularity: Array of item popularity scores
        
    Returns:
        Novelty score
    """
    popularities = item_popularity[recommended]
    
    # Avoid log(0)
    popularities = np.clip(popularities, 1e-10, 1.0)
    
    return float(np.mean(-np.log(popularities)))


class MetricsComputer:
    """
    Utility class to compute multiple metrics at once.
    """
    
    def __init__(
        self,
        k_values: Optional[List[int]] = None,
        item_embeddings: Optional[np.ndarray] = None,
        item_popularity: Optional[np.ndarray] = None
    ):
        """
        Initialize metrics computer.
        
        Args:
            k_values: List of K values for K-based metrics
            item_embeddings: Item embeddings for diversity
            item_popularity: Item popularity for novelty
        """
        self.k_values = k_values if k_values is not None else [5, 10, 20]
        self.item_embeddings = item_embeddings
        self.item_popularity = item_popularity
    
    def compute_all(
        self,
        recommended: Union[List[int], np.ndarray],
        relevant: Set[int]
    ) -> dict:
        """
        Compute all metrics.
        
        Args:
            recommended: Recommended items
            relevant: Relevant items
            
        Returns:
            Dictionary of metric name -> value
        """
        results = {}
        
        for k in self.k_values:
            results[f'precision@{k}'] = compute_precision_at_k(recommended, relevant, k)
            results[f'recall@{k}'] = compute_recall_at_k(recommended, relevant, k)
            results[f'ndcg@{k}'] = compute_ndcg_at_k(recommended, relevant, k)
            results[f'hit_rate@{k}'] = compute_hit_rate(recommended, relevant, k)
        
        results['mrr'] = compute_mrr(recommended, relevant)
        results['ap'] = compute_average_precision(recommended, relevant)
        
        if self.item_embeddings is not None:
            results['diversity'] = compute_diversity(recommended, self.item_embeddings)
        
        if self.item_popularity is not None:
            results['novelty'] = compute_novelty(recommended, self.item_popularity)
        
        return results
