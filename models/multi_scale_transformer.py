"""
Multi-Scale Transformer for depression assessment.
Supports both PHQ-8 and HAMD-13 scales with regression and ordinal classification.
Optionally includes Task Graph GAT for modeling inter-task relationships.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple, Dict


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer.
    Supports both batch_first=True and batch_first=False formats.
    """
    
    def __init__(self, d_model, max_len=5000, batch_first=True):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.batch_first = batch_first
        if batch_first:
            # For batch_first: [1, max_len, d_model]
            pe = pe.unsqueeze(0)
        else:
            # For batch_first=False: [max_len, 1, d_model]
            pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        if self.batch_first:
            # x: [batch_size, seq_len, d_model]
            # pe: [1, max_len, d_model]
            seq_len = x.size(1)
            return x + self.pe[:, :seq_len, :]
        else:
            # x: [seq_len, batch_size, d_model]
            # pe: [max_len, 1, d_model]
            seq_len = x.size(0)
            return x + self.pe[:seq_len, :, :]


class MultiScaleTransformer(nn.Module):
    """
    Multi-scale transformer for depression subscale prediction.
    Optionally includes Task Graph GAT for modeling inter-task relationships.
    """
    
    def __init__(
        self,
        input_dim: int = 768,  # BERT embedding size
        hidden_dim: int = 200,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3,
        num_subscales: int = 13,  # 8 for PHQ-8, 13 for HAMD-13
        subscale_ranges: List[int] = None,
        prediction_mode: str = "regression",  # "regression" or "ordinal"
        use_normalization: bool = True,
        per_subscale_normalization: bool = False,
        normalization_min: float = 0.0,
        normalization_max: float = 4.0,
        subscale_min_list: Optional[List[float]] = None,
        subscale_max_list: Optional[List[float]] = None,
        output_head_type: str = "linear",  # "linear", "mlp", "mlp_bn"
        # Task Graph GAT parameters
        use_task_graph: bool = False,
        scale_type: str = "PHQ-8",
        task_graph_embed_dim: int = 64,
        task_graph_hidden_dim: Optional[int] = None,  # Will use hidden_dim if None
        task_graph_num_layers: int = 2,
        task_graph_num_heads: int = 4,
        task_graph_dropout: float = 0.1,
        task_graph_omega_intra: float = 1.0,
        task_graph_omega_cross: float = 0.6,
        task_graph_learnable_weights: bool = True,
        task_graph_fusion_type: str = "gate"
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_subscales = num_subscales
        self.subscale_ranges = subscale_ranges if subscale_ranges else [4] * num_subscales
        self.prediction_mode = prediction_mode
        self.use_normalization = use_normalization
        self.per_subscale_normalization = per_subscale_normalization
        self.use_task_graph = use_task_graph
        self.scale_type = scale_type
        
        # Normalization parameters
        self.normalization_min = normalization_min
        self.normalization_max = normalization_max
        if per_subscale_normalization and subscale_min_list and subscale_max_list:
            self.register_buffer('subscale_min', torch.tensor(subscale_min_list))
            self.register_buffer('subscale_max', torch.tensor(subscale_max_list))
        else:
            self.subscale_min = None
            self.subscale_max = None
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding (batch_first=True)
        self.pos_encoder = PositionalEncoding(hidden_dim, batch_first=True)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Task Graph GAT (optional)
        self.task_graph = None
        self.task_fusion_heads = None
        task_graph_hidden = task_graph_hidden_dim if task_graph_hidden_dim else hidden_dim
        
        if use_task_graph:
            from models.task_graph_gat import TaskGraphGAT, TaskGraphFusion
            
            self.task_graph = TaskGraphGAT(
                num_tasks=num_subscales,
                task_embed_dim=task_graph_embed_dim,
                hidden_dim=task_graph_hidden,
                num_gat_layers=task_graph_num_layers,
                num_heads=task_graph_num_heads,
                dropout=task_graph_dropout,
                scale_type=scale_type,
                intra_cluster_weight_init=task_graph_omega_intra,
                cross_cluster_weight_init=task_graph_omega_cross,
                use_learnable_edge_weights=task_graph_learnable_weights,
                shared_embedding_dim=hidden_dim  # Encoder's hidden_dim for projection
            )
            
            # Create fusion heads for each task
            # Determine output dimension based on prediction mode
            if prediction_mode == "ordinal":
                num_classes = max(subscale_ranges) + 1 if subscale_ranges else 5
                output_dim_per_task = num_classes
            else:
                output_dim_per_task = 1
            
            self.task_fusion_heads = nn.ModuleList([
                TaskGraphFusion(
                    shared_dim=hidden_dim,
                    task_dim=task_graph_hidden,
                    output_dim=output_dim_per_task,
                    fusion_type=task_graph_fusion_type,
                    dropout=dropout
                ) for _ in range(num_subscales)
            ])
        
        # Output heads for each subscale (used when task_graph is disabled)
        self.output_head_type = output_head_type
        if prediction_mode == "ordinal":
            # Ordinal classification: each subscale has num_classes outputs
            num_classes = max(subscale_ranges) + 1 if subscale_ranges else 5
            self.output_heads = nn.ModuleList([
                nn.Linear(hidden_dim, num_classes) for _ in range(num_subscales)
            ])
        else:
            # Regression: different output head types
            if output_head_type == "linear":
                # Simple linear layer (default)
                self.output_heads = nn.ModuleList([
                    nn.Linear(hidden_dim, 1) for _ in range(num_subscales)
                ])
            elif output_head_type == "mlp":
                # MLP: Linear -> ReLU -> Linear (2-layer)
                self.output_heads = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim // 2),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim // 2, 1)
                    ) for _ in range(num_subscales)
                ])
            elif output_head_type == "mlp_bn":
                # MLP with BatchNorm: Linear -> BN -> ReLU -> Linear
                self.output_heads = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim // 2),
                        nn.BatchNorm1d(hidden_dim // 2),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim // 2, 1)
                    ) for _ in range(num_subscales)
                ])
            elif output_head_type == "mlp_deep":
                # Deeper MLP: Linear -> ReLU -> Linear -> ReLU -> Linear (3-layer)
                self.output_heads = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim // 2),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim // 2, hidden_dim // 4),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim // 4, 1)
                    ) for _ in range(num_subscales)
                ])
            else:
                raise ValueError(f"Unknown output_head_type: {output_head_type}. "
                               f"Supported: 'linear', 'mlp', 'mlp_bn', 'mlp_deep'")
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, attention_mask=None, return_task_graph_info: bool = False):
        """
        Forward pass.
        
        Args:
            x: [batch_size, seq_len, input_dim] input embeddings
            attention_mask: [batch_size, seq_len] attention mask (optional)
            return_task_graph_info: Whether to return task graph information (edge weights, etc.)
        
        Returns:
            output: [batch_size, num_subscales] or [batch_size, num_subscales, num_classes]
            task_graph_info (optional): Dictionary with task graph information if return_task_graph_info=True
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = self.input_projection(x)  # [batch_size, seq_len, hidden_dim]
        
        # Add positional encoding (batch_first=True format)
        x = self.pos_encoder(x)  # [batch_size, seq_len, hidden_dim]
        
        # Create attention mask if provided
        if attention_mask is not None:
            # src_key_padding_mask should be [batch_size, seq_len] when batch_first=True
            # True for padding tokens, False for valid tokens
            # attention_mask: True=valid, False=padding, so invert it
            mask = ~attention_mask  # True for padding tokens
        else:
            mask = None
        
        # Transformer encoder (batch_first=True, so x is [batch_size, seq_len, hidden_dim])
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Global average pooling -> Shared Embedding
        if mask is not None:
            # Masked average pooling
            # mask is [batch_size, seq_len], x is [batch_size, seq_len, hidden_dim]
            # Convert mask to [batch_size, seq_len, 1] for broadcasting
            mask_expanded = (~mask).unsqueeze(-1).float()  # [batch_size, seq_len, 1]
            shared_embedding = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            # Simple average pooling
            shared_embedding = x.mean(dim=1)  # [batch_size, hidden_dim]
        
        shared_embedding = self.dropout(shared_embedding)
        
        # Task Graph GAT branch (if enabled)
        task_graph_info = None
        if self.use_task_graph and self.task_graph is not None:
            # Get task embeddings from Task Graph GAT
            task_embeddings, edge_weights = self.task_graph(shared_embedding)
            # task_embeddings: [batch_size, num_subscales, task_graph_hidden_dim]
            
            if return_task_graph_info:
                task_graph_info = {
                    'edge_weights': edge_weights,
                    'task_embeddings': task_embeddings,
                    'omega_intra': self.task_graph.omega_intra.item() if hasattr(self.task_graph.omega_intra, 'item') else self.task_graph.omega_intra,
                    'omega_cross': self.task_graph.omega_cross.item() if hasattr(self.task_graph.omega_cross, 'item') else self.task_graph.omega_cross
                }
            
            # Fuse shared embedding with task-specific embeddings for prediction
            if self.prediction_mode == "ordinal":
                outputs = []
                for i, fusion_head in enumerate(self.task_fusion_heads):
                    task_emb = task_embeddings[:, i, :]  # [batch_size, task_graph_hidden_dim]
                    out = fusion_head(shared_embedding, task_emb)  # [batch_size, num_classes]
                    outputs.append(out)
                output = torch.stack(outputs, dim=1)  # [batch_size, num_subscales, num_classes]
            else:
                # Regression mode
                outputs = []
                for i, fusion_head in enumerate(self.task_fusion_heads):
                    task_emb = task_embeddings[:, i, :]  # [batch_size, task_graph_hidden_dim]
                    out = fusion_head(shared_embedding, task_emb)  # [batch_size, 1]
                    outputs.append(out.squeeze(-1))  # [batch_size]
                output = torch.stack(outputs, dim=1)  # [batch_size, num_subscales]
        else:
            # Standard output heads (no task graph)
            if self.prediction_mode == "ordinal":
                outputs = []
                for head in self.output_heads:
                    outputs.append(head(shared_embedding))  # [batch_size, num_classes]
                output = torch.stack(outputs, dim=1)  # [batch_size, num_subscales, num_classes]
            else:
                # Regression mode: support different head types
                outputs = []
                for head in self.output_heads:
                    if self.output_head_type == "mlp_bn":
                        # For BatchNorm1d, need to handle 2D input [batch_size, features]
                        out = head(shared_embedding)  # [batch_size, 1]
                        outputs.append(out.squeeze(-1))  # [batch_size]
                    else:
                        # For linear, mlp, mlp_deep: Sequential modules
                        out = head(shared_embedding)  # [batch_size, 1]
                        outputs.append(out.squeeze(-1))  # [batch_size]
                output = torch.stack(outputs, dim=1)  # [batch_size, num_subscales]
        
        if return_task_graph_info:
            return output, task_graph_info
        return output
    
    def get_task_graph_info(self) -> Optional[Dict]:
        """
        Get task graph information for visualization/debugging.
        
        Returns:
            Dictionary with task graph information, or None if task graph is not used.
        """
        if self.use_task_graph and self.task_graph is not None:
            return self.task_graph.get_graph_info()
        return None
    
    def normalize_labels(self, labels):
        """
        Normalize labels to [0, 1] range.
        
        Args:
            labels: [batch_size, num_subscales] original labels
        
        Returns:
            normalized: [batch_size, num_subscales] normalized labels
        """
        if not self.use_normalization:
            return labels
        
        if self.per_subscale_normalization and self.subscale_min is not None:
            # Per-subscale normalization
            ranges = self.subscale_max - self.subscale_min
            ranges = ranges.clamp(min=1e-8)  # Avoid division by zero
            normalized = (labels - self.subscale_min.unsqueeze(0)) / ranges.unsqueeze(0)
        else:
            # Global normalization
            range_val = self.normalization_max - self.normalization_min
            if range_val > 0:
                normalized = (labels - self.normalization_min) / range_val
            else:
                normalized = labels
        
        return normalized.clamp(0.0, 1.0)
    
    def denormalize_predictions(self, predictions):
        """
        Denormalize predictions from [0, 1] to original range.
        
        Args:
            predictions: [batch_size, num_subscales] normalized predictions
        
        Returns:
            denormalized: [batch_size, num_subscales] denormalized predictions
        """
        if not self.use_normalization:
            return predictions
        
        if self.per_subscale_normalization and self.subscale_min is not None:
            # Per-subscale denormalization
            ranges = self.subscale_max - self.subscale_min
            denormalized = predictions * ranges.unsqueeze(0) + self.subscale_min.unsqueeze(0)
        else:
            # Global denormalization
            range_val = self.normalization_max - self.normalization_min
            denormalized = predictions * range_val + self.normalization_min
        
        return denormalized

