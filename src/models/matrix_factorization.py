"""
Matrix Factorization baseline model for collaborative filtering.

This module implements:
- SVD-based matrix factorization
- Neural collaborative filtering
- Embedding extraction for RL state representation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import sparse
from scipy.sparse.linalg import svds
from typing import Tuple, Optional, List, Union, Any
from tqdm import tqdm


class SVDMatrixFactorization:
    """
    SVD-based Matrix Factorization for collaborative filtering.
    
    This is a simple but effective baseline that learns latent factors
    for users and items using Singular Value Decomposition.
    """
    
    def __init__(self, n_factors: int = 64):
        """
        Initialize SVD model.
        
        Args:
            n_factors: Number of latent factors (embedding dimension)
        """
        self.n_factors = n_factors
        self.user_embeddings: Optional[np.ndarray] = None
        self.item_embeddings: Optional[np.ndarray] = None
        self.user_bias: Optional[np.ndarray] = None
        self.item_bias: Optional[np.ndarray] = None
        self.global_mean: float = 0.0
        
    def fit(self, interaction_matrix: sparse.csr_matrix) -> 'SVDMatrixFactorization':
        """
        Fit the model using SVD.
        
        Args:
            interaction_matrix: Sparse user-item interaction matrix
            
        Returns:
            Self
        """
        # Compute global mean
        self.global_mean = interaction_matrix.data.mean() if interaction_matrix.nnz > 0 else 0.0
        
        # For implicit feedback (mostly 0s and 1s), skip centering to avoid numerical issues
        # Use the original matrix for SVD
        matrix_for_svd = interaction_matrix.astype(np.float64)
        
        # Perform truncated SVD with random starting vector to avoid zero-vector issues
        print(f"Performing SVD with {self.n_factors} factors...")
        rng = np.random.RandomState(42)
        v0 = rng.randn(min(interaction_matrix.shape))
        
        U, sigma, Vt = svds(matrix_for_svd, k=self.n_factors, v0=v0)
        
        # Store embeddings
        # U: (n_users, n_factors), Vt: (n_factors, n_items)
        # Cast to ensure proper types (svds returns ndarray)
        U = np.asarray(U)
        sigma = np.asarray(sigma)
        Vt = np.asarray(Vt)
        
        sigma_sqrt = np.sqrt(sigma)
        self.user_embeddings = U * sigma_sqrt  # (n_users, n_factors)
        self.item_embeddings = Vt.T * sigma_sqrt  # (n_items, n_factors)
        
        # Compute biases
        user_sums = np.array(interaction_matrix.sum(axis=1)).flatten()
        # Create boolean mask for non-zero entries and convert to float for summing
        user_counts_sparse = interaction_matrix.copy()
        user_counts_sparse.data = np.ones_like(user_counts_sparse.data)
        user_counts = np.array(user_counts_sparse.sum(axis=1)).flatten()
        self.user_bias = np.divide(user_sums, user_counts, 
                                    out=np.zeros_like(user_sums), 
                                    where=user_counts != 0) - self.global_mean
        
        item_sums = np.array(interaction_matrix.sum(axis=0)).flatten()
        item_counts_sparse = interaction_matrix.copy()
        item_counts_sparse.data = np.ones_like(item_counts_sparse.data)
        item_counts = np.array(item_counts_sparse.sum(axis=0)).flatten()
        self.item_bias = np.divide(item_sums, item_counts,
                                    out=np.zeros_like(item_sums),
                                    where=item_counts != 0) - self.global_mean
        
        print("SVD fitting complete!")
        return self
    
    def _check_fitted(self) -> None:
        """Check if model is fitted and raise error if not."""
        if self.user_embeddings is None or self.item_embeddings is None:
            raise ValueError("Model not fitted. Call fit() first.")
        if self.user_bias is None or self.item_bias is None:
            raise ValueError("Model not fitted. Call fit() first.")
    
    def predict(self, user_idx: int, item_idx: int) -> float:
        """
        Predict rating for a user-item pair.
        
        Args:
            user_idx: User index
            item_idx: Item index
            
        Returns:
            Predicted rating
        """
        self._check_fitted()
        assert self.user_embeddings is not None
        assert self.item_embeddings is not None
        assert self.user_bias is not None
        assert self.item_bias is not None
            
        score = (self.global_mean + 
                 self.user_bias[user_idx] + 
                 self.item_bias[item_idx] +
                 np.dot(self.user_embeddings[user_idx], self.item_embeddings[item_idx]))
        
        return float(score)
    
    def predict_batch(self, user_idx: int, item_indices: np.ndarray) -> np.ndarray:
        """
        Predict ratings for a user and multiple items.
        
        Args:
            user_idx: User index
            item_indices: Array of item indices
            
        Returns:
            Array of predicted ratings
        """
        self._check_fitted()
        assert self.user_embeddings is not None
        assert self.item_embeddings is not None
        assert self.user_bias is not None
        assert self.item_bias is not None
            
        user_emb = self.user_embeddings[user_idx]  # (n_factors,)
        item_embs = self.item_embeddings[item_indices]  # (n_items, n_factors)
        
        scores = (self.global_mean +
                  self.user_bias[user_idx] +
                  self.item_bias[item_indices] +
                  np.dot(item_embs, user_emb))
        
        return scores
    
    def recommend(self, user_idx: int, n_items: int = 10, 
                  exclude_items: Optional[List[int]] = None) -> np.ndarray:
        """
        Get top-N recommendations for a user.
        
        Args:
            user_idx: User index
            n_items: Number of recommendations
            exclude_items: Items to exclude (e.g., already interacted)
            
        Returns:
            Array of recommended item indices
        """
        self._check_fitted()
        assert self.item_embeddings is not None
            
        # Compute all scores
        all_scores = self.predict_batch(user_idx, np.arange(self.item_embeddings.shape[0]))
        
        # Exclude items
        if exclude_items is not None:
            all_scores[exclude_items] = -np.inf
        
        # Get top items
        top_items = np.argsort(all_scores)[::-1][:n_items]
        
        return top_items
    
    def get_user_embedding(self, user_idx: int) -> np.ndarray:
        """Get embedding for a user."""
        self._check_fitted()
        assert self.user_embeddings is not None
        return self.user_embeddings[user_idx]
    
    def get_item_embedding(self, item_idx: int) -> np.ndarray:
        """Get embedding for an item."""
        self._check_fitted()
        assert self.item_embeddings is not None
        return self.item_embeddings[item_idx]
    
    def save(self, path: str) -> None:
        """Save model to file."""
        self._check_fitted()
        assert self.user_embeddings is not None
        assert self.item_embeddings is not None
        assert self.user_bias is not None
        assert self.item_bias is not None
        
        np.savez(
            path,
            user_embeddings=self.user_embeddings,
            item_embeddings=self.item_embeddings,
            user_bias=self.user_bias,
            item_bias=self.item_bias,
            global_mean=self.global_mean,
            n_factors=self.n_factors
        )
    
    def load(self, path: str) -> 'SVDMatrixFactorization':
        """Load model from file."""
        data = np.load(path)
        self.user_embeddings = data['user_embeddings']
        self.item_embeddings = data['item_embeddings']
        self.user_bias = data['user_bias']
        self.item_bias = data['item_bias']
        self.global_mean = float(data['global_mean'])
        self.n_factors = int(data['n_factors'])
        return self


class NeuralCollaborativeFiltering(nn.Module):
    """
    Neural Collaborative Filtering (NCF) model.
    
    Combines matrix factorization with neural network for more
    expressive user-item interactions.
    """
    
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        hidden_layers: Optional[List[int]] = None,
        dropout: float = 0.2
    ):
        """
        Initialize NCF model.
        
        Args:
            n_users: Number of users
            n_items: Number of items
            embedding_dim: Dimension of embeddings
            hidden_layers: List of hidden layer sizes
            dropout: Dropout probability
        """
        super().__init__()
        
        if hidden_layers is None:
            hidden_layers = [128, 64]
        
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        
        # MF embeddings
        self.user_embedding_mf = nn.Embedding(n_users, embedding_dim)
        self.item_embedding_mf = nn.Embedding(n_items, embedding_dim)
        
        # MLP embeddings
        self.user_embedding_mlp = nn.Embedding(n_users, embedding_dim)
        self.item_embedding_mlp = nn.Embedding(n_items, embedding_dim)
        
        # MLP layers
        mlp_input_size = embedding_dim * 2
        layers: List[nn.Module] = []
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(mlp_input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            mlp_input_size = hidden_size
        self.mlp = nn.Sequential(*layers)
        
        # Final prediction layer
        # MF output (embedding_dim) + MLP output (last hidden layer)
        self.fc_out = nn.Linear(embedding_dim + hidden_layers[-1], 1)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self) -> None:
        """Initialize embeddings with small random values."""
        nn.init.normal_(self.user_embedding_mf.weight, std=0.01)
        nn.init.normal_(self.item_embedding_mf.weight, std=0.01)
        nn.init.normal_(self.user_embedding_mlp.weight, std=0.01)
        nn.init.normal_(self.item_embedding_mlp.weight, std=0.01)
        
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            user_ids: Tensor of user indices (batch_size,)
            item_ids: Tensor of item indices (batch_size,)
            
        Returns:
            Predicted scores (batch_size,)
        """
        # MF part
        user_mf = self.user_embedding_mf(user_ids)
        item_mf = self.item_embedding_mf(item_ids)
        mf_output = user_mf * item_mf  # Element-wise product
        
        # MLP part
        user_mlp = self.user_embedding_mlp(user_ids)
        item_mlp = self.item_embedding_mlp(item_ids)
        mlp_input = torch.cat([user_mlp, item_mlp], dim=-1)
        mlp_output = self.mlp(mlp_input)
        
        # Combine
        combined = torch.cat([mf_output, mlp_output], dim=-1)
        output = self.fc_out(combined).squeeze(-1)
        
        return output
    
    def get_user_embedding(self, user_idx: int) -> np.ndarray:
        """Get combined embedding for a user."""
        with torch.no_grad():
            user_tensor = torch.tensor([user_idx])
            emb_mf = self.user_embedding_mf(user_tensor)
            emb_mlp = self.user_embedding_mlp(user_tensor)
            combined = torch.cat([emb_mf, emb_mlp], dim=-1)
            return combined.squeeze(0).numpy()
    
    def get_item_embedding(self, item_idx: int) -> np.ndarray:
        """Get combined embedding for an item."""
        with torch.no_grad():
            item_tensor = torch.tensor([item_idx])
            emb_mf = self.item_embedding_mf(item_tensor)
            emb_mlp = self.item_embedding_mlp(item_tensor)
            combined = torch.cat([emb_mf, emb_mlp], dim=-1)
            return combined.squeeze(0).numpy()


class MatrixFactorization:
    """
    Unified Matrix Factorization interface.
    
    Supports both SVD and Neural Collaborative Filtering.
    """
    
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        method: str = 'svd',
        device: str = 'cpu'
    ):
        """
        Initialize MF model.
        
        Args:
            n_users: Number of users
            n_items: Number of items
            embedding_dim: Embedding dimension
            method: 'svd' or 'ncf'
            device: Device to use for neural model
        """
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.method = method
        self.device = device
        
        self.svd_model: Optional[SVDMatrixFactorization] = None
        self.ncf_model: Optional[NeuralCollaborativeFiltering] = None
        
        if method == 'svd':
            self.svd_model = SVDMatrixFactorization(n_factors=embedding_dim)
        elif method == 'ncf':
            self.ncf_model = NeuralCollaborativeFiltering(
                n_users=n_users,
                n_items=n_items,
                embedding_dim=embedding_dim
            ).to(device)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def fit(
        self,
        interaction_matrix: sparse.csr_matrix,
        train_df: Any = None,
        epochs: int = 50,
        lr: float = 0.001,
        batch_size: int = 256
    ) -> 'MatrixFactorization':
        """
        Fit the model.
        
        Args:
            interaction_matrix: Sparse interaction matrix (for SVD)
            train_df: Training DataFrame (for NCF)
            epochs: Number of epochs (for NCF)
            lr: Learning rate (for NCF)
            batch_size: Batch size (for NCF)
            
        Returns:
            Self
        """
        if self.method == 'svd' and self.svd_model is not None:
            self.svd_model.fit(interaction_matrix)
        elif self.method == 'ncf' and self.ncf_model is not None:
            self._train_ncf(train_df, epochs, lr, batch_size)
        
        return self
    
    def _train_ncf(
        self,
        train_df: Any,
        epochs: int,
        lr: float,
        batch_size: int
    ) -> None:
        """Train NCF model."""
        if self.ncf_model is None:
            raise ValueError("NCF model not initialized")
            
        optimizer = optim.Adam(self.ncf_model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # Prepare data
        users = torch.tensor(train_df['user_idx'].values, dtype=torch.long)
        items = torch.tensor(train_df['item_idx'].values, dtype=torch.long)
        ratings = torch.tensor(train_df['rating'].values, dtype=torch.float32)
        
        dataset = torch.utils.data.TensorDataset(users, items, ratings)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        self.ncf_model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_users, batch_items, batch_ratings in tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}'):
                batch_users = batch_users.to(self.device)
                batch_items = batch_items.to(self.device)
                batch_ratings = batch_ratings.to(self.device)
                
                optimizer.zero_grad()
                predictions = self.ncf_model(batch_users, batch_items)
                loss = criterion(predictions, batch_ratings)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
    
    def predict(self, user_idx: int, item_idx: int) -> float:
        """Predict score for user-item pair."""
        if self.method == 'svd' and self.svd_model is not None:
            return self.svd_model.predict(user_idx, item_idx)
        elif self.method == 'ncf' and self.ncf_model is not None:
            self.ncf_model.eval()
            with torch.no_grad():
                user = torch.tensor([user_idx], dtype=torch.long).to(self.device)
                item = torch.tensor([item_idx], dtype=torch.long).to(self.device)
                return float(self.ncf_model(user, item).item())
        raise ValueError("Model not initialized")
    
    def recommend(self, user_idx: int, n_items: int = 10,
                  exclude_items: Optional[List[int]] = None) -> np.ndarray:
        """Get top-N recommendations."""
        if self.method == 'svd' and self.svd_model is not None:
            return self.svd_model.recommend(user_idx, n_items, exclude_items)
        elif self.method == 'ncf' and self.ncf_model is not None:
            self.ncf_model.eval()
            with torch.no_grad():
                user = torch.tensor([user_idx] * self.n_items, dtype=torch.long).to(self.device)
                items = torch.arange(self.n_items, dtype=torch.long).to(self.device)
                scores = self.ncf_model(user, items).cpu().numpy()
                
                if exclude_items is not None:
                    scores[exclude_items] = -np.inf
                
                return np.argsort(scores)[::-1][:n_items]
        raise ValueError("Model not initialized")
    
    def get_user_embedding(self, user_idx: int) -> np.ndarray:
        """Get user embedding."""
        if self.method == 'svd' and self.svd_model is not None:
            return self.svd_model.get_user_embedding(user_idx)
        elif self.method == 'ncf' and self.ncf_model is not None:
            return self.ncf_model.get_user_embedding(user_idx)
        raise ValueError("Model not initialized")
    
    def get_item_embedding(self, item_idx: int) -> np.ndarray:
        """Get item embedding."""
        if self.method == 'svd' and self.svd_model is not None:
            return self.svd_model.get_item_embedding(item_idx)
        elif self.method == 'ncf' and self.ncf_model is not None:
            return self.ncf_model.get_item_embedding(item_idx)
        raise ValueError("Model not initialized")
    
    def get_all_user_embeddings(self) -> np.ndarray:
        """Get all user embeddings."""
        if self.method == 'svd' and self.svd_model is not None:
            if self.svd_model.user_embeddings is None:
                raise ValueError("Model not fitted")
            return self.svd_model.user_embeddings
        elif self.method == 'ncf' and self.ncf_model is not None:
            with torch.no_grad():
                users = torch.arange(self.n_users, dtype=torch.long)
                emb_mf = self.ncf_model.user_embedding_mf(users)
                emb_mlp = self.ncf_model.user_embedding_mlp(users)
                return torch.cat([emb_mf, emb_mlp], dim=-1).numpy()
        raise ValueError("Model not initialized")
    
    def get_all_item_embeddings(self) -> np.ndarray:
        """Get all item embeddings."""
        if self.method == 'svd' and self.svd_model is not None:
            if self.svd_model.item_embeddings is None:
                raise ValueError("Model not fitted")
            return self.svd_model.item_embeddings
        elif self.method == 'ncf' and self.ncf_model is not None:
            with torch.no_grad():
                items = torch.arange(self.n_items, dtype=torch.long)
                emb_mf = self.ncf_model.item_embedding_mf(items)
                emb_mlp = self.ncf_model.item_embedding_mlp(items)
                return torch.cat([emb_mf, emb_mlp], dim=-1).numpy()
        raise ValueError("Model not initialized")
    
    def save(self, path: str) -> None:
        """Save model."""
        if self.method == 'svd' and self.svd_model is not None:
            self.svd_model.save(path)
        elif self.method == 'ncf' and self.ncf_model is not None:
            torch.save(self.ncf_model.state_dict(), path)
    
    def load(self, path: str) -> 'MatrixFactorization':
        """Load model."""
        if self.method == 'svd' and self.svd_model is not None:
            self.svd_model.load(path)
        elif self.method == 'ncf' and self.ncf_model is not None:
            self.ncf_model.load_state_dict(torch.load(path))
        return self
