"""
Evaluation script for RL recommendation agent.

This script evaluates both baseline and RL models and
generates comparison plots.
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocess import load_processed_data
from src.models.matrix_factorization import MatrixFactorization
from src.models.dqn_agent import DQNAgent
from src.environment.recommender_env import RecommenderEnv
from src.evaluation.metrics import MetricsComputer


def evaluate_baseline(
    model: MatrixFactorization,
    data: Dict,
    k_values: List[int] = [5, 10, 20],
    num_users: int = 500
) -> Dict:
    """
    Evaluate baseline model on standard metrics.
    
    Args:
        model: Trained baseline model
        data: Data dictionary
        k_values: K values for metrics
        num_users: Number of users to evaluate on
        
    Returns:
        Dictionary of metrics
    """
    print("\nEvaluating Baseline Model...")
    
    test_df = data['test_df']
    train_df = data['train_df']
    
    test_by_user = test_df.groupby('user_idx')['item_idx'].apply(list).to_dict()
    train_by_user = train_df.groupby('user_idx')['item_idx'].apply(list).to_dict()
    
    metrics_computer = MetricsComputer(k_values=k_values)
    
    all_results = {f'precision@{k}': [] for k in k_values}
    all_results.update({f'recall@{k}': [] for k in k_values})
    all_results.update({f'ndcg@{k}': [] for k in k_values})
    all_results['mrr'] = []
    
    sample_users = list(test_by_user.keys())[:num_users]
    
    for user_idx in tqdm(sample_users, desc="Baseline Evaluation"):
        if user_idx not in train_by_user:
            continue
        
        actual_items = set(test_by_user[user_idx])
        exclude_items = train_by_user.get(user_idx, [])
        
        recommended = model.recommend(user_idx, n_items=max(k_values), exclude_items=exclude_items)
        
        results = metrics_computer.compute_all(recommended, actual_items)
        
        for key in all_results:
            if key in results:
                all_results[key].append(results[key])
    
    # Aggregate
    final_results = {key: np.mean(values) for key, values in all_results.items()}
    
    return final_results


def evaluate_agent(
    agent: DQNAgent,
    env: RecommenderEnv,
    num_episodes: int = 100
) -> Dict:
    """
    Evaluate RL agent on the environment.
    
    Args:
        agent: Trained DQN agent
        env: Recommendation environment
        num_episodes: Number of evaluation episodes
        
    Returns:
        Dictionary of metrics
    """
    print("\nEvaluating RL Agent...")
    
    episode_rewards = []
    episode_lengths = []
    episode_clicks = []
    episode_purchases = []
    
    for _ in tqdm(range(num_episodes), desc="RL Evaluation"):
        state, info = env.reset()
        
        total_reward = 0
        clicks = 0
        purchases = 0
        
        done = False
        while not done:
            candidate_embeddings = env.get_candidate_embeddings()
            action = agent.select_action(state, candidate_embeddings, training=False)
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            
            feedback_type = info.get('feedback_type', '')
            if 'click' in feedback_type or 'purchase' in feedback_type:
                clicks += 1
            if 'purchase' in feedback_type:
                purchases += 1
            
            state = next_state
        
        episode_rewards.append(total_reward)
        episode_lengths.append(info.get('step', env.max_steps))
        episode_clicks.append(clicks)
        episode_purchases.append(purchases)
    
    results = {
        'avg_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'avg_length': np.mean(episode_lengths),
        'avg_clicks': np.mean(episode_clicks),
        'avg_purchases': np.mean(episode_purchases),
        'ctr': np.sum(episode_clicks) / (np.sum(episode_lengths) + 1e-10),
        'purchase_rate': np.sum(episode_purchases) / (np.sum(episode_lengths) + 1e-10)
    }
    
    return results


def evaluate_random_policy(
    env: RecommenderEnv,
    num_episodes: int = 100
) -> Dict:
    """
    Evaluate random policy for comparison.
    
    Args:
        env: Recommendation environment
        num_episodes: Number of episodes
        
    Returns:
        Dictionary of metrics
    """
    print("\nEvaluating Random Policy...")
    
    episode_rewards = []
    episode_lengths = []
    
    for _ in tqdm(range(num_episodes), desc="Random Policy"):
        state, info = env.reset()
        
        total_reward = 0
        done = False
        
        while not done:
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            state = next_state
        
        episode_rewards.append(total_reward)
        episode_lengths.append(info.get('step', env.max_steps))
    
    return {
        'avg_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'avg_length': np.mean(episode_lengths)
    }


def compare_models(
    baseline_results: Dict,
    rl_results: Dict,
    random_results: Dict,
    save_path: Optional[str] = None
) -> None:
    """
    Generate comparison visualizations.
    
    Args:
        baseline_results: Baseline model metrics
        rl_results: RL agent metrics
        random_results: Random policy metrics
        save_path: Path to save plots
    """
    print("\nGenerating Comparison Plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Cumulative Reward Comparison
    ax1 = axes[0, 0]
    models = ['Random', 'Baseline (MF)', 'DQN (RL)']
    rewards = [
        random_results['avg_reward'],
        baseline_results.get('avg_reward', 0),  # Simulated
        rl_results['avg_reward']
    ]
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
    
    bars = ax1.bar(models, rewards, color=colors, edgecolor='black')
    ax1.set_ylabel('Average Cumulative Reward', fontsize=12)
    ax1.set_title('Episode Reward Comparison', fontsize=14, fontweight='bold')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Add value labels
    for bar, reward in zip(bars, rewards):
        height = bar.get_height()
        ax1.annotate(f'{reward:.2f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=11)
    
    # Plot 2: Precision/Recall/NDCG
    ax2 = axes[0, 1]
    k = 10
    metrics = ['Precision', 'Recall', 'NDCG']
    baseline_vals = [
        baseline_results.get(f'precision@{k}', 0),
        baseline_results.get(f'recall@{k}', 0),
        baseline_results.get(f'ndcg@{k}', 0)
    ]
    
    x = np.arange(len(metrics))
    ax2.bar(x, baseline_vals, color='#4ecdc4', edgecolor='black')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title(f'Baseline Offline Metrics @{k}', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, max(baseline_vals) * 1.3)
    
    for i, v in enumerate(baseline_vals):
        ax2.annotate(f'{v:.3f}', xy=(i, v), xytext=(0, 3),
                     textcoords="offset points", ha='center', fontsize=11)
    
    # Plot 3: RL Agent CTR and Purchase Rate
    ax3 = axes[1, 0]
    rl_metrics = ['CTR', 'Purchase Rate']
    rl_vals = [rl_results['ctr'] * 100, rl_results['purchase_rate'] * 100]
    
    bars = ax3.bar(rl_metrics, rl_vals, color=['#45b7d1', '#96ceb4'], edgecolor='black')
    ax3.set_ylabel('Percentage (%)', fontsize=12)
    ax3.set_title('RL Agent Engagement Metrics', fontsize=14, fontweight='bold')
    
    for bar, val in zip(bars, rl_vals):
        ax3.annotate(f'{val:.1f}%',
                     xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, 3),
                     textcoords="offset points", ha='center', fontsize=11)
    
    # Plot 4: Learning Summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = """
    📊 EVALUATION SUMMARY
    ═══════════════════════════════════════
    
    ✅ RL Agent vs Random Policy:
       Reward Improvement: {:.1f}x
    
    ✅ RL Agent Performance:
       Avg Episode Reward: {:.2f} ± {:.2f}
       Avg Session Length: {:.1f} steps
       Click-Through Rate: {:.1f}%
       Purchase Rate: {:.1f}%
    
    ✅ Baseline (Matrix Factorization):
       Precision@10: {:.3f}
       NDCG@10: {:.3f}
       MRR: {:.3f}
    
    ═══════════════════════════════════════
    """.format(
        rl_results['avg_reward'] / (random_results['avg_reward'] + 1e-10),
        rl_results['avg_reward'],
        rl_results['std_reward'],
        rl_results['avg_length'],
        rl_results['ctr'] * 100,
        rl_results['purchase_rate'] * 100,
        baseline_results.get('precision@10', 0),
        baseline_results.get('ndcg@10', 0),
        baseline_results.get('mrr', 0)
    )
    
    ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes,
             fontsize=11, verticalalignment='center',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_learning_curve(
    history_path: str,
    save_path: Optional[str] = None
) -> None:
    """
    Plot learning curve from training history.
    
    Args:
        history_path: Path to training history file
        save_path: Path to save plot
    """
    data = np.load(history_path)
    
    episode_rewards = data['episode_rewards']
    
    # Smooth with moving average
    window = 100
    smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(episode_rewards, alpha=0.3, color='blue', label='Raw')
    ax.plot(range(window-1, len(episode_rewards)), smoothed, color='blue', linewidth=2, label=f'Smoothed (window={window})')
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Cumulative Reward', fontsize=12)
    ax.set_title('RL Agent Learning Curve', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate models')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--data-path', type=str, default='data/processed')
    parser.add_argument('--baseline-path', type=str, default='results/models/baseline.npz')
    parser.add_argument('--agent-path', type=str, default='results/models/rl/agent_best.pt')
    parser.add_argument('--num-episodes', type=int, default=100)
    parser.add_argument('--output-dir', type=str, default='results/plots')
    
    args = parser.parse_args()
    
    # Load config
    config_path = PROJECT_ROOT / args.config
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {
            'model': {'user_embedding_dim': 64, 'hidden_layers': [256, 128]},
            'environment': {'session_length': 20, 'num_candidates': 100}
        }
    
    # Load data
    print("Loading data...")
    data = load_processed_data(str(PROJECT_ROOT / args.data_path))
    
    # Load baseline
    print("Loading baseline model...")
    baseline = MatrixFactorization(
        n_users=data['n_users'],
        n_items=data['n_items'],
        embedding_dim=config['model']['user_embedding_dim'],
        method='svd'
    )
    baseline.load(str(PROJECT_ROOT / args.baseline_path))
    
    # Get embeddings
    user_embeddings = baseline.get_all_user_embeddings()
    item_embeddings = baseline.get_all_item_embeddings()
    
    # Create environment
    env = RecommenderEnv(
        user_embeddings=user_embeddings,
        item_embeddings=item_embeddings,
        max_steps=config['environment']['session_length'],
        num_candidates=config['environment']['num_candidates']
    )
    
    # Load RL agent
    agent_path = PROJECT_ROOT / args.agent_path
    assert env.observation_space.shape is not None, "Observation space shape must be defined"
    state_dim = env.observation_space.shape[0]
    embedding_dim = config['model']['user_embedding_dim']
    
    agent = DQNAgent(
        state_dim=state_dim,
        embedding_dim=embedding_dim,
        num_candidates=config['environment']['num_candidates'],
        hidden_layers=config['model']['hidden_layers']
    )
    
    if agent_path.exists():
        print("Loading trained RL agent...")
        agent.load(str(agent_path))
    else:
        print("Warning: No trained agent found, using untrained agent")
    
    # Evaluate
    baseline_results = evaluate_baseline(baseline, data)
    rl_results = evaluate_agent(agent, env, num_episodes=args.num_episodes)
    random_results = evaluate_random_policy(env, num_episodes=args.num_episodes)
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    print("\nBaseline (Matrix Factorization):")
    for key, value in baseline_results.items():
        print(f"  {key}: {value:.4f}")
    
    print("\nRL Agent (DQN):")
    for key, value in rl_results.items():
        print(f"  {key}: {value:.4f}")
    
    print("\nRandom Policy:")
    for key, value in random_results.items():
        print(f"  {key}: {value:.4f}")
    
    # Generate plots
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    compare_models(
        baseline_results,
        rl_results,
        random_results,
        save_path=str(output_dir / 'model_comparison.png')
    )
    
    # Save results
    results = {
        'baseline': baseline_results,
        'rl_agent': rl_results,
        'random': random_results
    }
    
    with open(output_dir / 'evaluation_results.yaml', 'w') as f:
        yaml.dump(results, f)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
