"""
OpenAI Gym-compatible Recommendation Environment.

This module implements a custom Gym environment for training
RL agents on the recommendation task.

Key Components:
- State: User embedding + recent interaction history
- Action: Item index to recommend
- Reward: User feedback (click, dwell, purchase)
- Done: Session ends or max steps reached
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict, Tuple, Any, List

from .user_simulator import UserSimulator, FeedbackConfig


class RecommenderEnv(gym.Env):
    """
    Gym environment for recommendation with RL.
    
    This environment simulates user sessions where the agent
    must recommend items and receives rewards based on user feedback.
    
    State Space:
        - User embedding (embedding_dim)
        - Recent interaction history embedding (embedding_dim)
        - Session step (normalized)
        Total: 2 * embedding_dim + 1
    
    Action Space:
        - Discrete: Choose from candidate items
        - Continuous: Direct item embedding (advanced)
    
    Reward:
        - Purchase: +5
        - Click with dwell: +2
        - Click: +1
        - Skip: -1
        - Session complete bonus: +10
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        user_embeddings: np.ndarray,
        item_embeddings: np.ndarray,
        max_steps: int = 20,
        num_candidates: int = 100,
        history_length: int = 10,
        config: Optional[FeedbackConfig] = None,
        reward_config: Optional[Dict] = None,
        random_state: int = 42
    ):
        """
        Initialize the recommendation environment.
        
        Args:
            user_embeddings: Matrix of user embeddings (n_users, embedding_dim)
            item_embeddings: Matrix of item embeddings (n_items, embedding_dim)
            max_steps: Maximum recommendations per episode
            num_candidates: Number of candidate items per step
            history_length: Number of recent interactions to track
            config: User simulator configuration
            reward_config: Custom reward values
            random_state: Random seed
        """
        super().__init__()
        
        self.user_embeddings = user_embeddings
        self.item_embeddings = item_embeddings
        self.n_users = len(user_embeddings)
        self.n_items = len(item_embeddings)
        self.embedding_dim = user_embeddings.shape[1]
        
        self.max_steps = max_steps
        self.num_candidates = num_candidates
        self.history_length = history_length
        self.rng = np.random.RandomState(random_state)
        
        # Initialize user simulator
        self.simulator = UserSimulator(
            user_embeddings=user_embeddings,
            item_embeddings=item_embeddings,
            config=config,
            random_state=random_state
        )
        
        # Reward configuration
        self.reward_config = reward_config or {
            'session_complete_bonus': 10.0
        }
        
        # Define observation space
        # State = [user_embedding, history_embedding, normalized_step]
        state_dim = self.embedding_dim * 2 + 1
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )
        
        # Action space = index into candidate list
        self.action_space = spaces.Discrete(num_candidates)
        
        # Episode state
        self.current_user: Optional[int] = None
        self.current_step: int = 0
        self.candidates: Optional[np.ndarray] = None
        self.episode_rewards: List[float] = []
        self.interaction_history: List[int] = []
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment for new episode.
        
        Args:
            seed: Random seed
            options: Additional options (e.g., specific user)
            
        Returns:
            Tuple of (initial_state, info_dict)
        """
        if seed is not None:
            self.rng = np.random.RandomState(seed)
            self.simulator.rng = np.random.RandomState(seed)
        
        # Get user from options or sample randomly
        user_idx = None
        if options and 'user_idx' in options:
            user_idx = options['user_idx']
        
        # Reset simulator and get user
        self.current_user = self.simulator.reset(user_idx)
        self.current_step = 0
        self.episode_rewards = []
        self.interaction_history = []
        
        # Generate initial candidates
        self._generate_candidates()
        
        # Get initial state
        state = self._get_state()
        
        # Candidates are guaranteed to exist after _generate_candidates()
        assert self.candidates is not None
        
        info = {
            'user_idx': self.current_user,
            'candidates': self.candidates.copy(),
            'step': self.current_step
        }
        
        return state, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Index into candidate list (0 to num_candidates-1)
            
        Returns:
            Tuple of (next_state, reward, terminated, truncated, info)
        """
        if self.current_user is None:
            raise ValueError("Environment not reset. Call reset() first.")
        
        if self.candidates is None:
            raise ValueError("No candidates available. Call reset() first.")
        
        # Map action to item index
        if action < 0 or action >= len(self.candidates):
            # Invalid action - penalty
            reward = -2.0
            info = {'feedback_type': 'invalid_action', 'item_idx': -1}
        else:
            item_idx = self.candidates[action]
            
            # Get feedback from simulator
            reward, feedback_info = self.simulator.get_feedback(item_idx)
            
            # Update history
            self.interaction_history.append(item_idx)
            
            info = {
                'item_idx': item_idx,
                **feedback_info
            }
        
        # Update episode state
        self.current_step += 1
        self.episode_rewards.append(reward)
        
        # Check termination conditions
        terminated = False
        truncated = False
        
        # Episode ends if max steps reached
        if self.current_step >= self.max_steps:
            truncated = True
            # Add completion bonus
            reward += self.reward_config.get('session_complete_bonus', 0)
            info['session_complete'] = True
        
        # Episode ends if user leaves (simulator decides)
        if not self.simulator.should_continue_session():
            terminated = True
            info['user_left'] = True
        
        # Generate new candidates if episode continues
        if not (terminated or truncated):
            self._generate_candidates()
        
        # Get next state
        next_state = self._get_state()
        
        # Add episode statistics to info
        info.update({
            'step': self.current_step,
            'cumulative_reward': sum(self.episode_rewards),
            'candidates': self.candidates.copy() if self.candidates is not None else None
        })
        
        return next_state, reward, terminated, truncated, info
    
    def _get_state(self) -> np.ndarray:
        """
        Construct state representation.
        
        State components:
        1. User embedding
        2. History embedding (mean of recent items)
        3. Normalized step counter
        
        Returns:
            State vector
        """
        # User embedding
        if self.current_user is None:
            raise ValueError("No current user. Call reset() first.")
        user_emb = self.user_embeddings[self.current_user]
        
        # History embedding
        if self.interaction_history:
            recent = self.interaction_history[-self.history_length:]
            history_emb = np.mean(self.item_embeddings[recent], axis=0)
        else:
            history_emb = np.zeros(self.embedding_dim)
        
        # Normalized step
        norm_step = np.array([self.current_step / self.max_steps])
        
        # Concatenate
        state = np.concatenate([user_emb, history_emb, norm_step]).astype(np.float32)
        
        return state
    
    def _generate_candidates(self) -> None:
        """Generate candidate items for current step."""
        if self.current_user is None:
            raise ValueError("No current user. Call reset() first.")
            
        self.candidates = self.simulator.get_candidate_items(
            user_idx=self.current_user,
            n_candidates=self.num_candidates,
            exclude_history=True
        )
        
        # Pad if fewer candidates available
        if len(self.candidates) < self.num_candidates:
            padding = self.rng.choice(
                self.n_items,
                size=self.num_candidates - len(self.candidates),
                replace=True
            )
            self.candidates = np.concatenate([self.candidates, padding])
    
    def get_candidate_embeddings(self) -> np.ndarray:
        """
        Get embeddings for current candidates.
        
        Returns:
            Matrix of candidate item embeddings (num_candidates, embedding_dim)
        """
        return self.item_embeddings[self.candidates]
    
    def render(self, mode: str = 'human') -> None:
        """Render current state (for debugging)."""
        print(f"\n=== Step {self.current_step}/{self.max_steps} ===")
        print(f"User: {self.current_user}")
        print(f"History length: {len(self.interaction_history)}")
        print(f"Cumulative reward: {sum(self.episode_rewards):.2f}")
        print(f"Fatigue factor: {self.simulator.fatigue_factor:.3f}")
        print(f"Candidates: {len(self.candidates) if self.candidates is not None else 0} items")
    
    def close(self) -> None:
        """Cleanup."""
        pass


class SlateRecommenderEnv(RecommenderEnv):
    """
    Extended environment for slate-based recommendations.
    
    Instead of recommending one item at a time, the agent
    recommends a slate of K items.
    """
    
    def __init__(
        self,
        user_embeddings: np.ndarray,
        item_embeddings: np.ndarray,
        slate_size: int = 5,
        max_steps: int = 20,
        num_candidates: int = 100,
        **kwargs
    ):
        """
        Initialize slate recommendation environment.
        
        Args:
            user_embeddings: User embedding matrix
            item_embeddings: Item embedding matrix
            slate_size: Number of items per recommendation slate
            max_steps: Max recommendations per episode
            num_candidates: Candidate pool size
            **kwargs: Additional arguments
        """
        super().__init__(
            user_embeddings=user_embeddings,
            item_embeddings=item_embeddings,
            max_steps=max_steps,
            num_candidates=num_candidates,
            **kwargs
        )
        
        self.slate_size = slate_size
        
        # Update action space for slate
        # Action is now choosing slate_size items from candidates
        self.action_space = spaces.MultiDiscrete([num_candidates] * slate_size)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take a step with a slate of recommendations.
        
        Args:
            action: Array of candidate indices for slate
            
        Returns:
            Tuple of (next_state, reward, terminated, truncated, info)
        """
        if self.current_user is None:
            raise ValueError("Environment not reset. Call reset() first.")
        
        if self.candidates is None:
            raise ValueError("No candidates available. Call reset() first.")
        
        total_reward = 0.0
        feedback_list = []
        
        # Get feedback for each item in slate
        for item_action in action:
            if item_action < 0 or item_action >= len(self.candidates):
                continue
            
            item_idx = int(self.candidates[item_action])
            reward, feedback_info = self.simulator.get_feedback(item_idx)
            
            total_reward += reward
            feedback_list.append(feedback_info)
            self.interaction_history.append(item_idx)
        
        # Update episode state
        self.current_step += 1
        self.episode_rewards.append(total_reward)
        
        # Check termination
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        if truncated:
            total_reward += self.reward_config.get('session_complete_bonus', 0)
        
        if not self.simulator.should_continue_session():
            terminated = True
        
        if not (terminated or truncated):
            self._generate_candidates()
        
        next_state = self._get_state()
        
        info = {
            'step': self.current_step,
            'cumulative_reward': sum(self.episode_rewards),
            'slate_feedback': feedback_list,
            'candidates': self.candidates.copy() if self.candidates is not None else None
        }
        
        return next_state, total_reward, terminated, truncated, info


# Factory function
def create_env(
    user_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    env_type: str = 'single',
    **kwargs
) -> RecommenderEnv:
    """
    Create a recommendation environment.
    
    Args:
        user_embeddings: User embedding matrix
        item_embeddings: Item embedding matrix
        env_type: 'single' for single-item, 'slate' for slate recommendations
        **kwargs: Additional environment arguments
        
    Returns:
        Configured environment instance
    """
    if env_type == 'single':
        return RecommenderEnv(user_embeddings, item_embeddings, **kwargs)
    elif env_type == 'slate':
        return SlateRecommenderEnv(user_embeddings, item_embeddings, **kwargs)
    else:
        raise ValueError(f"Unknown env_type: {env_type}")
