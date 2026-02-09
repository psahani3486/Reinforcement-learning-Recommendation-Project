"""
Deep Q-Network (DQN) Agent for Recommendation.

This module implements:
- DQN with experience replay
- Target network for stable learning
- Epsilon-greedy exploration
- Double DQN variant

Interview Key Points:
- "We use DQN because it handles discrete action spaces well"
- "Experience replay breaks correlation between samples"
- "Target network prevents oscillations during training"
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
from typing import List, Tuple, Optional, Dict
import random


# Experience tuple for replay buffer
Experience = namedtuple('Experience', 
    ['state', 'action', 'reward', 'next_state', 'done', 'candidate_embeddings'])


class ReplayBuffer:
    """
    Experience Replay Buffer for DQN.
    
    Stores transitions and allows random sampling for training.
    This breaks correlation between consecutive experiences.
    """
    
    def __init__(self, capacity: int = 100000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
        
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        candidate_embeddings: Optional[np.ndarray] = None
    ) -> None:
        """
        Add experience to buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Whether episode ended
            candidate_embeddings: Embeddings of candidate items
        """
        self.buffer.append(Experience(
            state, action, reward, next_state, done, candidate_embeddings
        ))
    
    def sample(self, batch_size: int) -> List[Experience]:
        """
        Sample a batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            List of Experience tuples
        """
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        return len(self.buffer)


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay.
    
    Samples experiences based on TD-error priority,
    focusing learning on surprising transitions.
    """
    
    def __init__(self, capacity: int = 100000, alpha: float = 0.6):
        """
        Initialize prioritized buffer.
        
        Args:
            capacity: Maximum capacity
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
        """
        super().__init__(capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha
        
    def push(self, *args, priority: float = 1.0, **kwargs) -> None:
        """Add experience with priority."""
        super().push(*args, **kwargs)
        self.priorities.append(priority ** self.alpha)
    
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """
        Sample with importance weights.
        
        Args:
            batch_size: Batch size
            beta: Importance sampling exponent
            
        Returns:
            Tuple of (experiences, indices, importance_weights)
        """
        priorities = np.array(self.priorities)
        probs = priorities / priorities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        experiences = [self.buffer[idx] for idx in indices]
        
        # Importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights = weights / weights.max()
        
        return experiences, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update priorities for sampled experiences."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority ** self.alpha


class QNetwork(nn.Module):
    """
    Q-Network for value estimation.
    
    Architecture:
    - State encoder (MLP)
    - Candidate encoder (shared)
    - Q-value head for each candidate
    """
    
    def __init__(
        self,
        state_dim: int,
        embedding_dim: int,
        hidden_layers: List[int] = [256, 128],
        dropout: float = 0.2
    ):
        """
        Initialize Q-Network.
        
        Args:
            state_dim: Dimension of state vector
            embedding_dim: Dimension of item embeddings
            hidden_layers: List of hidden layer sizes
            dropout: Dropout probability
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.embedding_dim = embedding_dim
        
        # State encoder
        state_layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_layers:
            state_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        self.state_encoder = nn.Sequential(*state_layers)
        
        # Candidate encoder (projects item embeddings)
        self.candidate_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_layers[-1]),
            nn.ReLU()
        )
        
        # Q-value computation (dot product attention style)
        self.q_head = nn.Sequential(
            nn.Linear(hidden_layers[-1] * 2, hidden_layers[-1]),
            nn.ReLU(),
            nn.Linear(hidden_layers[-1], 1)
        )
        
    def forward(
        self,
        state: torch.Tensor,
        candidate_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Q-values for all candidates.
        
        Args:
            state: State tensor (batch_size, state_dim)
            candidate_embeddings: Candidate item embeddings 
                                  (batch_size, num_candidates, embedding_dim)
        
        Returns:
            Q-values (batch_size, num_candidates)
        """
        batch_size = state.shape[0]
        num_candidates = candidate_embeddings.shape[1]
        
        # Encode state
        state_encoded = self.state_encoder(state)  # (batch, hidden)
        
        # Encode candidates
        candidate_flat = candidate_embeddings.view(-1, self.embedding_dim)
        candidates_encoded = self.candidate_encoder(candidate_flat)
        candidates_encoded = candidates_encoded.view(batch_size, num_candidates, -1)
        
        # Expand state for each candidate
        state_expanded = state_encoded.unsqueeze(1).expand(-1, num_candidates, -1)
        
        # Concatenate and compute Q-values
        combined = torch.cat([state_expanded, candidates_encoded], dim=-1)
        q_values = self.q_head(combined).squeeze(-1)  # (batch, num_candidates)
        
        return q_values


class DuelingQNetwork(nn.Module):
    """
    Dueling DQN architecture.
    
    Separates value and advantage streams for better learning.
    Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
    """
    
    def __init__(
        self,
        state_dim: int,
        embedding_dim: int,
        hidden_layers: List[int] = [256, 128],
        dropout: float = 0.2
    ):
        """Initialize Dueling Q-Network."""
        super().__init__()
        
        self.state_dim = state_dim
        self.embedding_dim = embedding_dim
        
        # Shared encoder
        shared_layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_layers[:-1]:
            shared_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        self.shared_encoder = nn.Sequential(*shared_layers)
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(prev_dim, hidden_layers[-1]),
            nn.ReLU(),
            nn.Linear(hidden_layers[-1], 1)
        )
        
        # Advantage stream
        self.candidate_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_layers[-1]),
            nn.ReLU()
        )
        
        self.advantage_head = nn.Sequential(
            nn.Linear(prev_dim + hidden_layers[-1], hidden_layers[-1]),
            nn.ReLU(),
            nn.Linear(hidden_layers[-1], 1)
        )
        
    def forward(
        self,
        state: torch.Tensor,
        candidate_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Compute Q-values using dueling architecture."""
        batch_size = state.shape[0]
        num_candidates = candidate_embeddings.shape[1]
        
        # Shared encoding
        shared = self.shared_encoder(state)  # (batch, hidden)
        
        # Value stream
        value = self.value_stream(shared)  # (batch, 1)
        
        # Advantage stream
        candidate_flat = candidate_embeddings.view(-1, self.embedding_dim)
        candidates_encoded = self.candidate_encoder(candidate_flat)
        candidates_encoded = candidates_encoded.view(batch_size, num_candidates, -1)
        
        shared_expanded = shared.unsqueeze(1).expand(-1, num_candidates, -1)
        combined = torch.cat([shared_expanded, candidates_encoded], dim=-1)
        advantages = self.advantage_head(combined).squeeze(-1)  # (batch, num_candidates)
        
        # Combine: Q = V + A - mean(A)
        q_values = value + advantages - advantages.mean(dim=1, keepdim=True)
        
        return q_values


class DQNAgent:
    """
    DQN Agent for recommendation.
    
    Features:
    - Experience replay
    - Target network
    - Epsilon-greedy exploration
    - Optional Double DQN
    - Optional Dueling architecture
    """
    
    def __init__(
        self,
        state_dim: int,
        embedding_dim: int,
        num_candidates: int = 100,
        hidden_layers: List[int] = [256, 128],
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        double_dqn: bool = True,
        dueling: bool = False,
        device: str = 'auto'
    ):
        """
        Initialize DQN agent.
        
        Args:
            state_dim: Dimension of state
            embedding_dim: Dimension of item embeddings
            num_candidates: Number of candidate items
            hidden_layers: Hidden layer sizes
            learning_rate: Learning rate
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Epsilon decay rate
            buffer_size: Replay buffer capacity
            batch_size: Training batch size
            target_update_freq: Steps between target updates
            double_dqn: Use Double DQN
            dueling: Use Dueling architecture
            device: Device to use
        """
        self.state_dim = state_dim
        self.embedding_dim = embedding_dim
        self.num_candidates = num_candidates
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.double_dqn = double_dqn
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Create networks
        NetworkClass = DuelingQNetwork if dueling else QNetwork
        self.policy_net = NetworkClass(
            state_dim=state_dim,
            embedding_dim=embedding_dim,
            hidden_layers=hidden_layers
        ).to(self.device)
        
        self.target_net = NetworkClass(
            state_dim=state_dim,
            embedding_dim=embedding_dim,
            hidden_layers=hidden_layers
        ).to(self.device)
        
        # Initialize target network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)
        
        # Training stats
        self.train_step = 0
        self.losses = []
        
    def select_action(
        self,
        state: np.ndarray,
        candidate_embeddings: np.ndarray,
        training: bool = True
    ) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            candidate_embeddings: Embeddings of candidate items
            training: Whether in training mode
            
        Returns:
            Index of selected candidate
        """
        # Epsilon-greedy exploration
        if training and random.random() < self.epsilon:
            return random.randrange(len(candidate_embeddings))
        
        # Greedy action
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            candidates_tensor = torch.FloatTensor(candidate_embeddings).unsqueeze(0).to(self.device)
            
            q_values = self.policy_net(state_tensor, candidates_tensor)
            action = q_values.argmax(dim=1).item()
            
        return action
    
    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        candidate_embeddings: np.ndarray
    ) -> None:
        """Store experience in replay buffer."""
        self.replay_buffer.push(
            state, action, reward, next_state, done, candidate_embeddings
        )
    
    def train(self) -> Optional[float]:
        """
        Train on a batch from replay buffer.
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        
        # Unpack batch
        states = torch.FloatTensor(np.array([e.state for e in batch])).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor(np.array([e.next_state for e in batch])).to(self.device)
        dones = torch.FloatTensor([e.done for e in batch]).to(self.device)
        
        # Use stored candidate embeddings if available, otherwise fall back to random
        if batch[0].candidate_embeddings is not None:
            candidate_embs = torch.FloatTensor(
                np.array([e.candidate_embeddings for e in batch])
            ).to(self.device)
        else:
            candidate_embs = torch.randn(
                self.batch_size, self.num_candidates, self.embedding_dim
            ).to(self.device)
        
        # Compute current Q-values
        current_q = self.policy_net(states, candidate_embs)
        current_q = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: use policy net for action selection
                next_q_policy = self.policy_net(next_states, candidate_embs)
                next_actions = next_q_policy.argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_states, candidate_embs)
                next_q = next_q.gather(1, next_actions).squeeze(1)
            else:
                # Standard DQN
                next_q = self.target_net(next_states, candidate_embs)
                next_q = next_q.max(dim=1)[0]
            
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss and update
        loss = F.smooth_l1_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network periodically
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        loss_value = loss.item()
        self.losses.append(loss_value)
        
        return loss_value
    
    def save(self, path: str) -> None:
        """Save agent state."""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_step': self.train_step,
            'losses': self.losses
        }, path)
    
    def load(self, path: str) -> None:
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.train_step = checkpoint['train_step']
        self.losses = checkpoint['losses']


class ActorCriticAgent:
    """
    Actor-Critic agent (for future extension).
    
    Combines policy gradient (actor) with value estimation (critic)
    for potentially better sample efficiency.
    """
    
    def __init__(self, state_dim: int, embedding_dim: int, **kwargs):
        """Placeholder for Actor-Critic implementation."""
        raise NotImplementedError("Actor-Critic not yet implemented. Use DQNAgent.")
