"""
Task-Level Self-Paced Learning (SPL) for multi-task learning.
Dynamically adjusts task contributions based on learning difficulty.

This is adapted from self_paced_learning.py to ensure device compatibility
using register_buffer while maintaining the exact same logic.
"""
from dataclasses import dataclass
from typing import Dict, List
import torch
import torch.nn as nn
import numpy as np


@dataclass
class SPLConfig:
    """Configuration for Task-Level Self-Paced Learning."""
    # Pace function type
    pace_function: str = "linear"  # linear, log, self_paced, mixture
    
    # Initial threshold (lambda) - only tasks with loss < initial_lambda are used
    initial_lambda: float = 0.5
    
    # Final threshold - at the end of training
    final_lambda: float = 2.0
    
    # Growth rate for lambda
    lambda_growth: str = "linear"  # linear, exponential, step
    
    # For step growth
    step_epochs: List[int] = None
    
    # For mixture of easy and hard tasks
    easy_ratio: float = 0.7  # Start with 70% easy tasks
    
    # Regularization
    use_regularization: bool = False
    reg_weight: float = 0.1
    
    #
    min_task_ratio: float = 0.25

    # Minimum weight for each task (prevent complete exclusion)
    min_task_weight: float = 0.0  # Each task gets at least this weight (0.0 = disabled)
    
    # Use balanced weighting (normalize weights to maintain relative importance)
    use_balanced_weighting: bool = False  # If True, normalize weights to sum to num_tasks
    
    def __post_init__(self):
        if self.step_epochs is None:
            self.step_epochs = [2, 4, 6]


class TaskLevelSPL(nn.Module):
    """
    Task-Level Self-Paced Learning
    
    Instead of weighting samples, this weights tasks (items) based on their difficulty.
    Easier tasks are learned first, harder tasks are gradually included.
    
    This implementation uses register_buffer for device compatibility while
    maintaining the exact same logic as the original self_paced_learning.py
    """
    
    def __init__(self, config: SPLConfig, num_tasks: int):
        super().__init__()
        self.config = config
        self.num_tasks = num_tasks
        self.current_lambda = config.initial_lambda
        self.current_epoch = 0
        
        # Track task difficulties (per-task average loss)
        # Use register_buffer to ensure device compatibility
        self.register_buffer('task_losses', torch.zeros(num_tasks))
        self.register_buffer('task_weights', torch.ones(num_tasks))
        self.register_buffer('task_update_count', torch.zeros(num_tasks))
    
    def update_lambda(self, epoch: int, total_epochs: int):
        """Update the threshold lambda based on training progress"""
        self.current_epoch = epoch
        progress = epoch / max(1, total_epochs - 1)
       
        if self.config.lambda_growth == "linear":
            self.current_lambda = (
                self.config.initial_lambda + 
                (self.config.final_lambda - self.config.initial_lambda) * progress
            )
        elif self.config.lambda_growth == "exponential":
            # Avoid division by zero: ensure initial_lambda > 0
            safe_initial = max(self.config.initial_lambda, 1e-6)
            if safe_initial > 0 and self.config.final_lambda > 0:
                self.current_lambda = safe_initial * (
                    (self.config.final_lambda / safe_initial) ** progress
                )
            else:
                # Fallback to linear growth if values are invalid
                self.current_lambda = (
                    safe_initial + 
                    (self.config.final_lambda - safe_initial) * progress
                )
        elif self.config.lambda_growth == "step":
            # Step-wise increase
            if self.config.step_epochs and len(self.config.step_epochs) > 0:
                for step_epoch in self.config.step_epochs:
                    if epoch >= step_epoch:
                        step_progress = self.config.step_epochs.index(step_epoch) + 1
                        self.current_lambda = self.config.initial_lambda + (
                            (self.config.final_lambda - self.config.initial_lambda) * 
                            step_progress / len(self.config.step_epochs)
                        )
            else:
                # Fallback to linear growth if step_epochs is empty
                self.current_lambda = (
                    self.config.initial_lambda + 
                    (self.config.final_lambda - self.config.initial_lambda) * progress
                )
                    
        return self.current_lambda
    
    def update_task_difficulties(
        self,
        per_task_losses: torch.Tensor  # [num_tasks] - average loss per task
    ):
        """
        Update task difficulty estimates based on current losses
        
        Args:
            per_task_losses: Average loss for each task [num_tasks]
        """
        # Ensure per_task_losses is on the same device
        per_task_losses = per_task_losses.to(self.task_losses.device)
        
        # Exponential moving average
        alpha = 0.1  # Smoothing factor
        for task_id in range(self.num_tasks):
            if self.task_update_count[task_id] == 0:
                # First update: directly assign (keep as tensor for register_buffer)
                self.task_losses[task_id] = per_task_losses[task_id]
            else:
                # Subsequent updates: exponential moving average (keep as tensor)
                self.task_losses[task_id] = (
                    alpha * per_task_losses[task_id] + 
                    (1 - alpha) * self.task_losses[task_id]
                )
            self.task_update_count[task_id] += 1
    
    def compute_task_weights(self) -> torch.Tensor:
        """
        Compute weights for each task based on their difficulty
        
        Returns:
            weights: Task weights [num_tasks]
        """
        # Ensure lambda is positive
        safe_lambda = max(self.current_lambda, 1e-6)
        # Convert to tensor for torch operations
        safe_lambda_tensor = torch.tensor(safe_lambda, device=self.task_losses.device, dtype=self.task_losses.dtype)
        
        if self.config.pace_function == "linear":
            # Linear pace: weight = max(0, 1 - loss/lambda)
            weights = torch.clamp(
                1.0 - self.task_losses / safe_lambda_tensor,
                min=0.0,
                max=1.0
            )
        elif self.config.pace_function == "log":
            # Logarithmic pace function for smoother transitions
            # Formula: weight = max(0, 1 - log(1 + loss) / log(1 + lambda))
            # This provides smoother transitions than linear pace
            safe_log_denom = torch.log(1 + safe_lambda_tensor)
            # safe_log_denom is always > 0 when safe_lambda_tensor > 0, but add clamp for safety
            safe_log_denom = torch.clamp(safe_log_denom, min=1e-8)
            weights = torch.clamp(
                1.0 - torch.log(1 + self.task_losses) / safe_log_denom,
                min=0.0
            )
        elif self.config.pace_function == "self_paced":
            # Binary selection: weight = 1 if loss < lambda, else 0
            # Use safe_lambda_tensor for consistency with other pace functions
            weights = (self.task_losses < safe_lambda_tensor).float()
        elif self.config.pace_function == "mixture":
            # Mixture: easier tasks get higher weights
            sorted_indices = torch.argsort(self.task_losses)
            num_easy = int(self.num_tasks * self.config.easy_ratio)
            
            weights = torch.zeros(self.num_tasks, device=self.task_losses.device)
            weights[sorted_indices[:num_easy]] = 1.0
            weights[sorted_indices[num_easy:]] = 0.3
        else:
            weights = torch.ones(self.num_tasks, device=self.task_losses.device)
        
        
        # Ensure minimum samples are used
        weight_sum = weights.sum().item() if isinstance(weights.sum(), torch.Tensor) else weights.sum()
        min_tasks = max(1, int(self.num_tasks * self.config.min_task_ratio))
        
        if weight_sum < min_tasks:
            # Use top-k easiest tasks
            k = min(min_tasks, self.num_tasks)
            top_k_indices = torch.topk(self.task_losses, k, largest=False).indices
            weights = torch.zeros(self.num_tasks, device=self.task_losses.device, dtype=self.task_losses.dtype)
            weights[top_k_indices] = 1.0
        self.task_weights = weights
        return weights
    
    def apply_task_weights(
        self,
        per_task_losses: torch.Tensor  # [batch, num_tasks] or [num_tasks]
    ) -> torch.Tensor:
        """
        Apply task weights to per-task losses
        
        Args:
            per_task_losses: Losses for each task [batch, num_tasks] or [num_tasks]
            
        Returns:
            weighted_losses: Weighted losses with same shape
        """
        if per_task_losses.dim() == 1:
            # [num_tasks]
            weights = self.compute_task_weights().to(per_task_losses.device)
            return per_task_losses * weights
        else:
            # [batch, num_tasks]
            weights = self.compute_task_weights().to(per_task_losses.device)
            return per_task_losses * weights.unsqueeze(0)
    
    def compute_regularization(self) -> torch.Tensor:
        """Compute regularization term for task-level SPL"""
        if not self.config.use_regularization:
            return torch.tensor(0.0, device=self.task_weights.device)
        
        # FIXED: Use mean instead of sum to prevent overwhelming the loss
        weights = self.compute_task_weights()
        reg = -self.config.reg_weight * self.current_lambda * weights.mean()
        return reg
    
    def get_curriculum_statistics(self) -> Dict:
        """Get statistics about the current curriculum"""
        weights = self.compute_task_weights()
        
        return {
            'current_lambda': self.current_lambda,
            'current_epoch': self.current_epoch,
            'num_active_tasks': (weights > 0.0).sum().item(),
            'total_tasks': self.num_tasks,
            'active_ratio': (weights > 0.0).sum().item() / self.num_tasks,
            'mean_weight': weights.mean().item(),
            'mean_task_loss': self.task_losses.mean().item(),
            'median_task_loss': self.task_losses.median().item(),
            'task_weights': weights.tolist(),
            'task_losses': self.task_losses.tolist()
        }

