"""
Early stopping utility for training.
"""
import copy
import numpy as np
import torch


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True, verbose=True):
        """
        Args:
            patience: Number of epochs to wait after last improvement
            min_delta: Minimum change to qualify as an improvement
            restore_best_weights: If True, restore model weights from best epoch
            verbose: If True, print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self._restore_best_weights_flag = restore_best_weights  # Use different name to avoid conflict with method
        self.verbose = verbose
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        self.best_metrics = {}  # Store all best metrics
        self.early_stop = False
    
    def __call__(self, val_loss, model=None, metrics=None):
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            model: Model to save/restore weights from (optional)
            metrics: Dictionary of additional metrics to track (optional)
        
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            if model is not None:
                self.best_weights = copy.deepcopy(model.state_dict())
            # Save best metrics
            if metrics:
                self.best_metrics = copy.deepcopy(metrics)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if model is not None:
                self.best_weights = copy.deepcopy(model.state_dict())
            # Save best metrics
            if metrics:
                self.best_metrics = copy.deepcopy(metrics)
            if self.verbose:
                print(f"EarlyStopping: New best val_loss {self.best_loss:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: No improvement for {self.counter}/{self.patience} epochs")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"EarlyStopping: Stopping training after {self.patience} epochs without improvement")
                if self._restore_best_weights_flag and model is not None and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    if self.verbose:
                        print("EarlyStopping: Restored best model weights")
        
        return self.early_stop
    
    def restore_best_weights(self, model):
        """
        Restore the best model weights.
        
        Args:
            model: Model to restore weights to
        """
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            if self.verbose:
                print("EarlyStopping: Restored best model weights")
        else:
            if self.verbose:
                print("EarlyStopping: No best weights available to restore")
    
    def get_best_metrics(self):
        """Return the best metrics achieved."""
        return self.best_metrics
    
    def reset(self):
        """Reset early stopping state."""
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        self.best_metrics = {}
        self.early_stop = False


