"""
Ordinal loss for ordinal classification.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalLoss(nn.Module):
    """
    Ordinal loss for multi-class ordinal classification.
    Based on the approach of treating ordinal classes as a sequence of binary classifications.
    """
    
    def __init__(self, num_classes=4, reduction='mean'):
        """
        Args:
            num_classes: Number of ordinal classes (e.g., 4 for 0-3)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.num_classes = num_classes
        self.reduction = reduction
    
    def forward(self, predictions, targets):
        """
        Compute ordinal loss.
        
        Args:
            predictions: [batch_size, num_subscales, num_classes] logits
            targets: [batch_size, num_subscales] class indices
        
        Returns:
            loss: Scalar or per-sample loss
        """
        batch_size, num_subscales, num_classes = predictions.shape
        
        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets.long(), num_classes=num_classes).float()
        
        # Compute probabilities
        probs = F.softmax(predictions, dim=-1)
        
        # Compute cumulative probabilities
        cum_probs = torch.cumsum(probs, dim=-1)
        cum_targets = torch.cumsum(targets_one_hot, dim=-1)
        
        # Binary cross-entropy for each threshold
        loss = F.binary_cross_entropy(
            cum_probs[:, :, :-1],  # Exclude last class (always 1)
            cum_targets[:, :, :-1],
            reduction='none'
        )
        
        # Sum over thresholds
        loss = loss.sum(dim=-1)  # [batch_size, num_subscales]
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def ordinal_predict(logits):
    """
    Predict ordinal class from logits.
    
    Args:
        logits: [batch_size, num_subscales, num_classes] logits
    
    Returns:
        predictions: [batch_size, num_subscales] predicted class indices
    """
    probs = F.softmax(logits, dim=-1)
    cum_probs = torch.cumsum(probs, dim=-1)
    
    # Find the first threshold where cumulative probability > 0.5
    predictions = (cum_probs > 0.5).int().argmax(dim=-1)
    
    return predictions

