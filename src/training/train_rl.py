"""
Train RL Agent for recommendation.

This script trains the DQN agent using the user simulator
and recommendation environment.
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess import load_processed_data
from src.models.matrix_factorization import MatrixFactorization
from src.models.dqn_agent import DQNAgent
from src.environment.recommender_env import RecommenderEnv


class TrainingLogger:
    """Logger for training progress."""
    
    def __init__(self, log_dir: str, use_tensorboard: bool = True):
        """Initialize logger."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_tensorboard = use_tensorboard
        self.writer = None
        
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=str(self.log_dir))
            except ImportError:
                print("TensorBoard not available. Logging to files only.")
                self.use_tensorboard = False
        
        # Training history
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.epsilons = []
        
    def log_episode(
        self,
        episode: int,
        reward: float,
        length: int,
        loss: Optional[float],
        epsilon: float
    ) -> None:
        """Log episode metrics."""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        if loss is not None:
            self.losses.append(loss)
        self.epsilons.append(epsilon)
        
        if self.use_tensorboard and self.writer:
            self.writer.add_scalar('Episode/Reward', reward, episode)
            self.writer.add_scalar('Episode/Length', length, episode)
            if loss is not None:
                self.writer.add_scalar('Training/Loss', loss, episode)
            self.writer.add_scalar('Training/Epsilon', epsilon, episode)
    
    def log_evaluation(
        self,
        episode: int,
        metrics: Dict
    ) -> None:
        """Log evaluation metrics."""
        if self.use_tensorboard and self.writer:
            for name, value in metrics.items():
                self.writer.add_scalar(f'Eval/{name}', value, episode)
    
    def save_history(self, filename: str = 'training_history.npz') -> None:
        """Save training history."""
        np.savez(
            self.log_dir / filename,
            episode_rewards=self.episode_rewards,
            episode_lengths=self.episode_lengths,
            losses=self.losses,
            epsilons=self.epsilons
        )
    
    def close(self) -> None:
        """Close logger."""
        if self.writer:
            self.writer.close()


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_rl_agent(
    data_path: str,
    baseline_model_path: str,
    config: Dict,
    save_dir: str,
    log_dir: str,
    device: str = 'auto'
) -> DQNAgent:
    """
    Train the RL agent.
    
    Args:
        data_path: Path to processed data
        baseline_model_path: Path to trained baseline model
        config: Configuration dictionary
        save_dir: Directory to save checkpoints
        log_dir: Directory for logs
        device: Device to use
        
    Returns:
        Trained DQNAgent
    """
    print("=" * 60)
    print("Training RL Agent (DQN)")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    data = load_processed_data(data_path)
    n_users = data['n_users']
    n_items = data['n_items']
    
    # Load baseline model for embeddings
    print("\nLoading baseline model for embeddings...")
    embedding_dim = config['model']['user_embedding_dim']
    
    baseline = MatrixFactorization(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=embedding_dim,
        method='svd'
    )
    baseline.load(baseline_model_path)
    
    # Get embeddings
    user_embeddings = baseline.get_all_user_embeddings()
    item_embeddings = baseline.get_all_item_embeddings()
    
    print(f"User embeddings: {user_embeddings.shape}")
    print(f"Item embeddings: {item_embeddings.shape}")
    
    # Create environment
    print("\nCreating recommendation environment...")
    env_config = config.get('environment', {})
    
    env = RecommenderEnv(
        user_embeddings=user_embeddings,
        item_embeddings=item_embeddings,
        max_steps=env_config.get('session_length', 20),
        num_candidates=env_config.get('num_candidates', 100),
        history_length=config['model'].get('history_length', 10)
    )
    
    # Initialize DQN agent
    print("\nInitializing DQN agent...")
    train_config = config.get('training', {})
    
    assert env.observation_space.shape is not None, "Observation space shape must be defined"
    state_dim = env.observation_space.shape[0]
    
    agent = DQNAgent(
        state_dim=state_dim,
        embedding_dim=embedding_dim,
        num_candidates=env_config.get('num_candidates', 100),
        hidden_layers=config['model'].get('hidden_layers', [256, 128]),
        learning_rate=train_config.get('learning_rate', 0.001),
        gamma=train_config.get('gamma', 0.99),
        epsilon_start=train_config.get('epsilon_start', 1.0),
        epsilon_end=train_config.get('epsilon_end', 0.01),
        epsilon_decay=train_config.get('epsilon_decay', 0.995),
        buffer_size=train_config.get('replay_buffer_size', 100000),
        batch_size=train_config.get('batch_size', 64),
        target_update_freq=train_config.get('target_update_freq', 100),
        double_dqn=True,
        device=device
    )
    
    print(f"Agent device: {agent.device}")
    
    # Initialize logger
    logger = TrainingLogger(log_dir, use_tensorboard=config.get('logging', {}).get('tensorboard', True))
    
    # Training loop
    print("\nStarting training...")
    episodes = train_config.get('episodes', 10000)
    save_freq = train_config.get('save_freq', 500)
    log_freq = train_config.get('log_freq', 100)
    min_replay_size = train_config.get('min_replay_size', 1000)
    
    best_reward = float('-inf')
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    for episode in tqdm(range(1, episodes + 1), desc="Training"):
        # Reset environment
        state, info = env.reset()
        candidates = info['candidates']
        
        episode_reward = 0
        episode_loss = None
        
        done = False
        while not done:
            # Get candidate embeddings
            candidate_embeddings = env.get_candidate_embeddings()
            
            # Select action
            action = agent.select_action(state, candidate_embeddings, training=True)
            
            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store experience
            agent.store_experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                candidate_embeddings=candidate_embeddings
            )
            
            # Train
            if len(agent.replay_buffer) >= min_replay_size:
                loss = agent.train()
                if loss is not None:
                    episode_loss = loss
            
            # Update state
            state = next_state
            if not done:
                candidates = info.get('candidates', candidates)
            
            episode_reward += reward
        
        # Log episode
        episode_length = info.get('step', env.max_steps) if info else env.max_steps
        logger.log_episode(
            episode=episode,
            reward=episode_reward,
            length=episode_length,
            loss=episode_loss,
            epsilon=agent.epsilon
        )
        
        # Periodic logging
        if episode % log_freq == 0:
            avg_reward = np.mean(logger.episode_rewards[-log_freq:])
            avg_length = np.mean(logger.episode_lengths[-log_freq:])
            
            print(f"\nEpisode {episode}/{episodes}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Length: {avg_length:.1f}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            print(f"  Buffer Size: {len(agent.replay_buffer)}")
        
        # Save checkpoint
        if episode % save_freq == 0:
            checkpoint_path = save_path / f'agent_ep{episode}.pt'
            agent.save(str(checkpoint_path))
            
            # Save best model
            avg_reward = np.mean(logger.episode_rewards[-100:])
            if avg_reward > best_reward:
                best_reward = avg_reward
                agent.save(str(save_path / 'agent_best.pt'))
    
    # Save final model
    agent.save(str(save_path / 'agent_final.pt'))
    logger.save_history()
    logger.close()
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best Average Reward: {best_reward:.2f}")
    print(f"Models saved to: {save_path}")
    print("=" * 60)
    
    return agent


def main():
    """Main function to train RL agent."""
    parser = argparse.ArgumentParser(description='Train RL agent')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--data-path', type=str, default='data/processed',
                        help='Path to processed data')
    parser.add_argument('--baseline-path', type=str, default='results/models/baseline.npz',
                        help='Path to baseline model')
    parser.add_argument('--save-dir', type=str, default='results/models/rl',
                        help='Directory to save checkpoints')
    parser.add_argument('--log-dir', type=str, default='results/logs',
                        help='Directory for logs')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Load config
    config_path = PROJECT_ROOT / args.config
    if config_path.exists():
        config = load_config(str(config_path))
    else:
        # Default config
        config = {
            'model': {
                'user_embedding_dim': 64,
                'hidden_layers': [256, 128],
                'history_length': 10
            },
            'training': {
                'episodes': 10000,
                'batch_size': 64,
                'learning_rate': 0.001,
                'gamma': 0.99,
                'epsilon_start': 1.0,
                'epsilon_end': 0.01,
                'epsilon_decay': 0.995,
                'replay_buffer_size': 100000,
                'min_replay_size': 1000,
                'target_update_freq': 100,
                'save_freq': 500,
                'log_freq': 100
            },
            'environment': {
                'session_length': 20,
                'num_candidates': 100
            },
            'logging': {
                'tensorboard': True
            }
        }
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = PROJECT_ROOT / args.save_dir / timestamp
    log_dir = PROJECT_ROOT / args.log_dir / timestamp
    
    # Train
    agent = train_rl_agent(
        data_path=str(PROJECT_ROOT / args.data_path),
        baseline_model_path=str(PROJECT_ROOT / args.baseline_path),
        config=config,
        save_dir=str(save_dir),
        log_dir=str(log_dir),
        device=args.device
    )


if __name__ == '__main__':
    main()
