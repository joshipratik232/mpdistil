"""Curriculum learning utilities for MPDistil.

This module provides reward computation and task sampling for
reinforcement learning-based curriculum learning.
"""

from typing import Dict, List, Tuple
import random
import numpy as np
import torch
from torch.distributions import Categorical


def discount_rewards(rewards: List[float], gamma: float = 0.99) -> np.ndarray:
    """Discount and normalize rewards for REINFORCE algorithm.
    
    Args:
        rewards: List of rewards
        gamma: Discount factor (default: 0.99)
        
    Returns:
        Discounted and normalized rewards
    """
    r = np.array([gamma**i * rewards[i] for i in range(len(rewards))])
    # Reverse, cumsum, reverse back
    r = r[::-1].cumsum()[::-1]
    # Normalize (subtract mean)
    return r - r.mean()


class RewardCalculator:
    """Computes rewards for curriculum learning.
    
    Reward is based on how much better the student performs compared
    to the teacher on the main task.
    
    Args:
        reward_type: 'binary' or 'real'
    """
    
    def __init__(self, reward_type: str = 'binary'):
        if reward_type not in ['binary', 'real']:
            raise ValueError("reward_type must be 'binary' or 'real'")
        self.reward_type = reward_type
    
    def compute_reward(
        self,
        student_out: torch.Tensor,
        teacher_out: torch.Tensor,
        labels: torch.Tensor,
        task_name: str,
        is_regression: bool = False
    ) -> torch.Tensor:
        """Compute reward based on student vs teacher performance.
        
        Args:
            student_out: Student logits [batch_size, num_labels] or values
            teacher_out: Teacher logits [batch_size, num_labels] or values
            labels: Ground truth labels [batch_size]
            task_name: Name of task
            is_regression: Whether this is a regression task
            
        Returns:
            Scalar reward tensor
        """
        if not is_regression:
            # Classification task
            teacher_out = torch.nn.Softmax(-1)(teacher_out)
            student_out = torch.nn.Softmax(-1)(student_out)
            
            # Get probabilities for correct class
            teacher_out = teacher_out.gather(
                dim=1, index=labels.long().view(-1, 1)
            ).squeeze()
            student_out = student_out.gather(
                dim=1, index=labels.long().view(-1, 1)
            ).squeeze()
            
            if self.reward_type == 'real':
                reward = (student_out - teacher_out).float().sum()
            elif self.reward_type == 'binary':
                reward = (student_out > teacher_out).float().sum()
        else:
            # Regression task
            teacher_out = teacher_out[:, 0]
            student_out = student_out[:, 0]
            
            if self.reward_type == 'real':
                reward = (
                    torch.abs(labels - teacher_out) - 
                    torch.abs(labels - student_out)
                ).float().sum()
            elif self.reward_type == 'binary':
                reward = (
                    torch.abs(labels - teacher_out) > 
                    torch.abs(labels - student_out)
                ).float().sum()
        
        return reward


class CurriculumSampler:
    """Manages task sampling for curriculum learning.
    
    Pre-loads all held batches into memory for efficient random access.
    
    Args:
        task_names: List of task names
        task_loaders: Dict of task loaders with 'held' key
    """
    
    def __init__(self, task_names: List[str], task_loaders: Dict):
        self.task_names = task_names
        self.all_held_batches = {}
        
        # Pre-load all held batches into memory
        for task_name in task_names:
            self.all_held_batches[task_name] = {}
            held_loader = task_loaders[task_name]['held']['loader']
            for idx, batch in enumerate(held_loader):
                self.all_held_batches[task_name][idx] = batch
    
    def sample_batch(self, task_name: str) -> Tuple:
        """Sample a random batch from the specified task.
        
        Args:
            task_name: Name of task to sample from
            
        Returns:
            Batch tuple (input_ids, attention_mask, token_type_ids, labels)
        """
        idx = random.choice(list(self.all_held_batches[task_name].keys()))
        return self.all_held_batches[task_name][idx]
    
    def get_num_batches(self, task_name: str) -> int:
        """Get number of batches for a task.
        
        Args:
            task_name: Name of task
            
        Returns:
            Number of batches
        """
        return len(self.all_held_batches[task_name])


class ActionSelector:
    """Selects next task using action model policy.
    
    Args:
        action_model: ActionPredictor model
        task_names: List of task names
    """
    
    def __init__(self, action_model, task_names: List[str]):
        self.action_model = action_model
        self.task_names = task_names
    
    def select_action(self, state: torch.Tensor) -> Tuple[str, torch.Tensor, torch.Tensor]:
        """Select next task based on current state.
        
        Args:
            state: Current model state [1, d_model]
            
        Returns:
            Tuple of (task_name, action_index, action_probs)
        """
        # Get action probabilities
        action_probs = self.action_model(state)[0]  # [num_actions]
        
        # Sample action
        action_idx = Categorical(action_probs).sample()  # scalar tensor
        
        # Get task name
        task_name = self.task_names[action_idx.cpu().numpy()]
        
        return task_name, action_idx, action_probs
    
    def compute_policy_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor
    ) -> torch.Tensor:
        """Compute policy gradient loss for REINFORCE.
        
        Args:
            states: Batch of states [num_steps, d_model]
            actions: Batch of action indices [num_steps, 1]
            rewards: Discounted rewards [num_steps]
            
        Returns:
            Policy gradient loss (scalar)
        """
        # Get log probabilities
        logprob = torch.log(self.action_model.forward(states))
        
        # Select log probs for taken actions
        selected_logprobs = rewards * torch.gather(logprob, 1, actions).squeeze()
        
        # Loss is negative expected reward
        loss = -selected_logprobs.mean()
        
        return loss
