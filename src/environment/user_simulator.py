"""
User Simulator for generating synthetic user feedback.

This module simulates realistic user behavior based on:
- User-item embedding similarity
- User preference patterns
- Session fatigue modeling
- Probabilistic feedback generation

Interview Key Point:
"We use a probabilistic user simulator based on embedding similarity 
to approximate real user feedback for online RL training."
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass


@dataclass
class FeedbackConfig:
    """Configuration for feedback generation."""
    purchase_threshold: float = 0.7
    click_threshold: float = 0.4
    
    # Reward values
    reward_purchase: float = 5.0
    reward_click_dwell: float = 2.0
    reward_click: float = 1.0
    reward_skip: float = -1.0
    
    # Noise and randomness
    noise_std: float = 0.1
    click_noise: float = 0.15
    
    # Fatigue modeling
    enable_fatigue: bool = True
    fatigue_decay: float = 0.95


class UserSimulator:
    """
    Simulates user behavior for training RL recommendation agents.
    
    The simulator uses embedding similarity to determine user preferences
    and generates probabilistic feedback signals (clicks, dwell time, purchases).
    
    Key Features:
    - Embedding-based preference modeling
    - Probabilistic feedback with controlled noise
    - User fatigue modeling (engagement decays over session)
    - Session-level behavior patterns
    """
    
    def __init__(
        self,
        user_embeddings: np.ndarray,
        item_embeddings: np.ndarray,
        config: Optional[FeedbackConfig] = None,
        random_state: int = 42
    ):
        """
        Initialize user simulator.
        
        Args:
            user_embeddings: Matrix of user embeddings (n_users, embedding_dim)
            item_embeddings: Matrix of item embeddings (n_items, embedding_dim)
            config: Feedback configuration
            random_state: Random seed for reproducibility
        """
        self.user_embeddings = user_embeddings
        self.item_embeddings = item_embeddings
        self.config = config or FeedbackConfig()
        self.rng = np.random.RandomState(random_state)
        
        # Normalize embeddings for cosine similarity
        self.user_embeddings_norm = self._normalize(user_embeddings)
        self.item_embeddings_norm = self._normalize(item_embeddings)
        
        # Session state
        self.current_user: Optional[int] = None
        self.session_step: int = 0
        self.fatigue_factor: float = 1.0
        self.interaction_history: List[int] = []
        
    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings for cosine similarity computation."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        return embeddings / norms
    
    def reset(self, user_idx: Optional[int] = None) -> int:
        """
        Reset simulator for a new user session.
        
        Args:
            user_idx: Specific user index, or None for random user
            
        Returns:
            Selected user index
        """
        if user_idx is None:
            user_idx = self.rng.randint(0, len(self.user_embeddings))
        
        self.current_user = user_idx
        self.session_step = 0
        self.fatigue_factor = 1.0
        self.interaction_history = []
        
        return user_idx
    
    def compute_similarity(self, user_idx: int, item_idx: int) -> float:
        """
        Compute cosine similarity between user and item embeddings.
        
        Args:
            user_idx: User index
            item_idx: Item index
            
        Returns:
            Cosine similarity score in [-1, 1]
        """
        user_emb = self.user_embeddings_norm[user_idx]
        item_emb = self.item_embeddings_norm[item_idx]
        return np.dot(user_emb, item_emb)
    
    def compute_similarity_batch(self, user_idx: int, item_indices: np.ndarray) -> np.ndarray:
        """
        Compute similarities for multiple items.
        
        Args:
            user_idx: User index
            item_indices: Array of item indices
            
        Returns:
            Array of similarity scores
        """
        user_emb = self.user_embeddings_norm[user_idx]
        item_embs = self.item_embeddings_norm[item_indices]
        return np.dot(item_embs, user_emb)
    
    def get_feedback(self, item_idx: int) -> Tuple[float, Dict]:
        """
        Generate feedback for a recommended item.
        
        This is the core function that simulates user response.
        
        Logic:
        1. Compute embedding similarity
        2. Add noise for realism
        3. Apply fatigue decay
        4. Generate probabilistic feedback
        
        Args:
            item_idx: Index of recommended item
            
        Returns:
            Tuple of (reward, info_dict)
        """
        if self.current_user is None:
            raise ValueError("No active session. Call reset() first.")
        
        # Compute base similarity
        similarity = self.compute_similarity(self.current_user, item_idx)
        
        # Add noise for realistic variation
        noisy_similarity = similarity + self.rng.normal(0, self.config.noise_std)
        noisy_similarity = np.clip(noisy_similarity, -1, 1)
        
        # Apply fatigue factor
        effective_similarity = noisy_similarity * self.fatigue_factor
        
        # Determine feedback based on thresholds
        reward, feedback_type, dwell_time = self._generate_feedback(effective_similarity)
        
        # Update session state
        self.session_step += 1
        self.interaction_history.append(item_idx)
        
        if self.config.enable_fatigue:
            self.fatigue_factor *= self.config.fatigue_decay
        
        info = {
            'similarity': similarity,
            'noisy_similarity': noisy_similarity,
            'effective_similarity': effective_similarity,
            'feedback_type': feedback_type,
            'dwell_time': dwell_time,
            'session_step': self.session_step,
            'fatigue_factor': self.fatigue_factor
        }
        
        return reward, info
    
    def _generate_feedback(self, similarity: float) -> Tuple[float, str, float]:
        """
        Generate feedback based on similarity score.
        
        Args:
            similarity: Effective similarity score
            
        Returns:
            Tuple of (reward, feedback_type, dwell_time)
        """
        # Add click noise for probabilistic behavior
        click_roll = self.rng.random()
        
        if similarity > self.config.purchase_threshold:
            # High similarity → Purchase (with high probability)
            if click_roll < 0.85:  # 85% chance of purchase
                return self.config.reward_purchase, 'purchase', 60.0
            else:
                return self.config.reward_click_dwell, 'click_dwell', 30.0
                
        elif similarity > self.config.click_threshold:
            # Medium similarity → Click (possibly with dwell)
            if click_roll < 0.6:  # 60% chance of meaningful click
                dwell_time = self.rng.uniform(10, 30)
                if dwell_time > 20:
                    return self.config.reward_click_dwell, 'click_dwell', dwell_time
                else:
                    return self.config.reward_click, 'click', dwell_time
            else:
                return self.config.reward_skip, 'skip', 0.0
                
        else:
            # Low similarity → Mostly skip
            if click_roll < 0.1:  # 10% chance of exploration click
                return self.config.reward_click, 'click', self.rng.uniform(1, 5)
            else:
                return self.config.reward_skip, 'skip', 0.0
    
    def should_continue_session(self) -> bool:
        """
        Determine if user should continue the session.
        
        Users may leave early if they're not engaged.
        
        Returns:
            True if session should continue
        """
        # Probability of leaving increases with fatigue
        leave_prob = (1 - self.fatigue_factor) * 0.3
        return self.rng.random() > leave_prob
    
    def get_user_preference_vector(self, user_idx: int) -> np.ndarray:
        """
        Get the preference vector for a user.
        
        This represents the user's latent preferences and can be used
        as part of the RL state representation.
        
        Args:
            user_idx: User index
            
        Returns:
            User embedding vector
        """
        return self.user_embeddings[user_idx]
    
    def get_session_history_embedding(
        self,
        history_length: int = 10
    ) -> np.ndarray:
        """
        Get aggregated embedding of recent interaction history.
        
        Args:
            history_length: Number of recent items to consider
            
        Returns:
            Aggregated embedding vector
        """
        if not self.interaction_history:
            return np.zeros(self.item_embeddings.shape[1])
        
        recent_items = self.interaction_history[-history_length:]
        embeddings = self.item_embeddings[recent_items]
        
        # Use mean pooling (could also use attention)
        return np.mean(embeddings, axis=0)
    
    def get_candidate_items(
        self,
        user_idx: int,
        n_candidates: int = 100,
        exclude_history: bool = True
    ) -> np.ndarray:
        """
        Get candidate items for recommendation.
        
        In practice, this would be from a candidate generation stage.
        Here we simulate by sampling items weighted by similarity.
        
        Args:
            user_idx: User index
            n_candidates: Number of candidates
            exclude_history: Whether to exclude already interacted items
            
        Returns:
            Array of candidate item indices
        """
        n_items = len(self.item_embeddings)
        
        # Compute all similarities
        similarities = self.compute_similarity_batch(user_idx, np.arange(n_items))
        
        # Convert to probabilities (softmax-like)
        # Mix of relevant and random items for diversity
        probs = np.exp(similarities * 2)  # Temperature scaling
        probs = probs / probs.sum()
        
        # Mix with uniform for exploration
        uniform = np.ones(n_items) / n_items
        probs = 0.7 * probs + 0.3 * uniform
        
        if exclude_history:
            probs[self.interaction_history] = 0
            probs = probs / probs.sum()
        
        # Sample candidates
        candidates = self.rng.choice(
            n_items, 
            size=min(n_candidates, n_items - len(self.interaction_history)),
            replace=False,
            p=probs
        )
        
        return candidates


class RealisticUserSimulator(UserSimulator):
    """
    Extended user simulator with more realistic behavior patterns.
    
    Additional features:
    - User mood variations
    - Time-of-day effects
    - Novelty seeking behavior
    - Category preferences
    """
    
    def __init__(
        self,
        user_embeddings: np.ndarray,
        item_embeddings: np.ndarray,
        item_categories: Optional[np.ndarray] = None,
        config: Optional[FeedbackConfig] = None,
        random_state: int = 42
    ):
        """
        Initialize realistic simulator.
        
        Args:
            user_embeddings: User embedding matrix
            item_embeddings: Item embedding matrix
            item_categories: Category labels for items
            config: Feedback configuration
            random_state: Random seed
        """
        super().__init__(user_embeddings, item_embeddings, config, random_state)
        
        self.item_categories = item_categories
        self.user_mood: float = 1.0  # 1.0 = normal, <1 = picky, >1 = generous
        self.novelty_preference: float = 0.5  # Balance of familiar vs new
        
    def reset(self, user_idx: Optional[int] = None) -> int:
        """Reset with random mood."""
        user_idx = super().reset(user_idx)
        
        # Random mood for this session
        self.user_mood = self.rng.uniform(0.8, 1.2)
        self.novelty_preference = self.rng.uniform(0.3, 0.7)
        
        return user_idx
    
    def _generate_feedback(self, similarity: float) -> Tuple[float, str, float]:
        """Generate feedback with mood adjustment."""
        # Adjust similarity based on mood
        adjusted_similarity = similarity * self.user_mood
        return super()._generate_feedback(adjusted_similarity)
    
    def get_feedback(self, item_idx: int) -> Tuple[float, Dict]:
        """
        Generate feedback with novelty bonus/penalty.
        
        Args:
            item_idx: Recommended item index
            
        Returns:
            Tuple of (reward, info_dict)
        """
        reward, info = super().get_feedback(item_idx)
        
        # Apply novelty adjustment
        if self.item_categories is not None:
            # Check if category is new
            item_cat = self.item_categories[item_idx]
            history_cats = [self.item_categories[i] for i in self.interaction_history[:-1]]
            
            if item_cat not in history_cats:
                # Novel category - adjust based on preference
                novelty_bonus = (self.novelty_preference - 0.5) * 0.5
                reward += novelty_bonus
                info['novelty_bonus'] = novelty_bonus
        
        info['user_mood'] = self.user_mood
        info['novelty_preference'] = self.novelty_preference
        
        return reward, info


# Factory function for easy instantiation
def create_simulator(
    user_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    simulator_type: str = 'basic',
    **kwargs
) -> UserSimulator:
    """
    Create a user simulator.
    
    Args:
        user_embeddings: User embedding matrix
        item_embeddings: Item embedding matrix
        simulator_type: 'basic' or 'realistic'
        **kwargs: Additional arguments for simulator
        
    Returns:
        Configured UserSimulator instance
    """
    if simulator_type == 'basic':
        return UserSimulator(user_embeddings, item_embeddings, **kwargs)
    elif simulator_type == 'realistic':
        return RealisticUserSimulator(user_embeddings, item_embeddings, **kwargs)
    else:
        raise ValueError(f"Unknown simulator type: {simulator_type}")
