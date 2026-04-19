"""
Task Graph with Graph Attention Network (GAT) for multi-task depression assessment.
Constructs task relationships based on clinically validated symptom clusters.

For PHQ-8:
- Core Depression: Anhedonia (0), Depressed_Mood (1), Worthlessness (5)
- Cognitive Function: Concentration (6), Psychomotor (7)
- Somatic Symptoms: Sleep_Problems (2), Fatigue (3), Appetite_Changes (4)

For HAMD-13:
- Cognitive: Guilt (0), Suicide (1)
- Sleep: Insomnia_Initial (2), Insomnia_Middle (3), Insomnia_Late (4)
- Retardation: Work_Interests (5), Genital_Symptoms (9)
- Anxiety/Somatization: Psychic_Anxiety (6), Hypochondriasis (10), GI_Symptoms (7), Somatic_Symptoms (8), Insight (12)
- Weight: Weight_Loss (11)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import math


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer (GAT) with learnable edge weights.
    
    Implements attention mechanism for graph neural networks with support for
    pre-defined edge weights based on clinical priors.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_nodes: int,
        dropout: float = 0.1,
        alpha: float = 0.2,
        concat: bool = True,
        use_edge_weights: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        self.use_edge_weights = use_edge_weights
        
        # Linear transformation for node features
        self.W = nn.Linear(in_features, out_features, bias=False)
        
        # Attention mechanism parameters
        self.a = nn.Parameter(torch.zeros(1, 2 * out_features))
        nn.init.xavier_uniform_(self.a)
        
        # Learnable edge weight modulation
        if use_edge_weights:
            self.edge_weight_transform = nn.Linear(1, 1, bias=True)
        
        self.leaky_relu = nn.LeakyReLU(self.alpha)
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(
        self,
        h: torch.Tensor,
        adj: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of GAT layer.
        
        Args:
            h: Node features [batch_size, num_nodes, in_features]
            adj: Adjacency matrix [num_nodes, num_nodes]
            edge_weights: Edge weight matrix [num_nodes, num_nodes]
            
        Returns:
            Updated node features [batch_size, num_nodes, out_features]
        """
        batch_size = h.size(0)
        
        # Linear transformation
        Wh = self.W(h)  # [batch_size, num_nodes, out_features]
        
        # Compute attention coefficients
        # Create all pairs of node features
        Wh_i = Wh.unsqueeze(2).expand(-1, -1, self.num_nodes, -1)  # [B, N, N, F]
        Wh_j = Wh.unsqueeze(1).expand(-1, self.num_nodes, -1, -1)  # [B, N, N, F]
        
        # Concatenate and apply attention
        a_input = torch.cat([Wh_i, Wh_j], dim=-1)  # [B, N, N, 2F]
        e = self.leaky_relu(torch.matmul(a_input, self.a.t()).squeeze(-1))  # [B, N, N]
        
        # Mask with adjacency matrix
        zero_vec = -9e15 * torch.ones_like(e)
        adj_expanded = adj.unsqueeze(0).expand(batch_size, -1, -1)  # [B, N, N]
        attention = torch.where(adj_expanded > 0, e, zero_vec)
        
        # Apply edge weights if provided
        if self.use_edge_weights and edge_weights is not None:
            edge_weights_expanded = edge_weights.unsqueeze(0).expand(batch_size, -1, -1)
            attention = attention * edge_weights_expanded
        
        # Softmax normalization
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout_layer(attention)
        
        # Aggregate neighbor features
        h_prime = torch.bmm(attention, Wh)  # [B, N, out_features]
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class MultiHeadGAT(nn.Module):
    """
    Multi-head Graph Attention Network.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_nodes: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        alpha: float = 0.2,
        concat: bool = True,
        use_edge_weights: bool = True
    ):
        super().__init__()
        self.num_heads = num_heads
        self.concat = concat
        
        # Create multiple attention heads
        self.attention_heads = nn.ModuleList([
            GraphAttentionLayer(
                in_features=in_features,
                out_features=out_features,
                num_nodes=num_nodes,
                dropout=dropout,
                alpha=alpha,
                concat=True,
                use_edge_weights=use_edge_weights
            ) for _ in range(num_heads)
        ])
        
        # Output projection if concatenating
        if concat:
            self.out_proj = nn.Linear(num_heads * out_features, out_features)
        
    def forward(
        self,
        h: torch.Tensor,
        adj: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with multiple attention heads.
        
        Args:
            h: Node features [batch_size, num_nodes, in_features]
            adj: Adjacency matrix [num_nodes, num_nodes]
            edge_weights: Edge weight matrix [num_nodes, num_nodes]
            
        Returns:
            Updated node features [batch_size, num_nodes, out_features]
        """
        # Apply each attention head
        head_outputs = [
            head(h, adj, edge_weights) for head in self.attention_heads
        ]
        
        if self.concat:
            # Concatenate and project
            h_cat = torch.cat(head_outputs, dim=-1)  # [B, N, num_heads * out_features]
            return self.out_proj(h_cat)  # [B, N, out_features]
        else:
            # Average
            return torch.stack(head_outputs, dim=0).mean(dim=0)


class TaskGraphGAT(nn.Module):
    """
    Task Graph with GAT for multi-task depression assessment.
    
    Constructs a graph where nodes represent tasks (subscales) and edges
    represent clinical correlations between symptoms.
    
    Features:
    - Learnable task embeddings
    - Clinically-informed graph structure
    - Learnable edge weights (intra-cluster and cross-cluster)
    - Multi-head graph attention for message passing
    """
    
    # PHQ-8 Symptom Clusters (based on clinical validation)
    PHQ8_CLUSTERS = {
        'core_depression': [0, 1, 5],      # Anhedonia, Depressed_Mood, Worthlessness
        'cognitive': [6, 7],                # Concentration, Psychomotor
        'somatic': [2, 3, 4]               # Sleep_Problems, Fatigue, Appetite_Changes
    }
    
    # HAMD-13 Symptom Clusters
    HAMD13_CLUSTERS = {
        'cognitive': [0, 1],               # Guilt, Suicide
        'sleep': [2, 3, 4],                # Insomnia_Initial, Insomnia_Middle, Insomnia_Late
        'retardation': [5, 9],             # Work_Interests, Genital_Symptoms
        'anxiety_somatization': [6, 7, 8, 10, 12],  # Psychic_Anxiety, GI, Somatic, Hypochondriasis, Insight
        'weight': [11]                     # Weight_Loss
    }
    
    # Cross-cluster connections based on clinical correlations
    PHQ8_CROSS_CLUSTER_EDGES = [
        # Core Depression <-> Cognitive Function (strong clinical correlation)
        (0, 6), (0, 7), (1, 6), (1, 7), (5, 6), (5, 7),
        # Core Depression <-> Somatic (moderate correlation)
        (0, 2), (0, 3), (1, 2), (1, 3), (1, 4),
        # Cognitive <-> Somatic (sleep affects concentration)
        (6, 2), (7, 3)
    ]
    
    HAMD13_CROSS_CLUSTER_EDGES = [
        # Cognitive <-> Sleep
        (0, 2), (0, 3), (1, 2), (1, 4),
        # Cognitive <-> Anxiety
        (0, 6), (1, 6),
        # Sleep <-> Anxiety
        (2, 6), (3, 6), (4, 6),
        # Retardation <-> Anxiety
        (5, 6), (5, 7),
        # Anxiety <-> Weight
        (7, 11), (8, 11)
    ]
    
    def __init__(
        self,
        num_tasks: int,
        task_embed_dim: int = 64,
        hidden_dim: int = 128,
        num_gat_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        scale_type: str = "PHQ-8",
        intra_cluster_weight_init: float = 1.0,
        cross_cluster_weight_init: float = 0.6,
        use_learnable_edge_weights: bool = True,
        shared_embedding_dim: Optional[int] = None  # Dimension of shared embedding from encoder
    ):
        """
        Initialize Task Graph GAT.
        
        Args:
            num_tasks: Number of tasks (subscales)
            task_embed_dim: Dimension of task embeddings
            hidden_dim: Hidden dimension for GAT layers
            num_gat_layers: Number of GAT layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            scale_type: "PHQ-8" or "HAMD-13"
            intra_cluster_weight_init: Initial weight for intra-cluster edges
            cross_cluster_weight_init: Initial weight for cross-cluster edges
            use_learnable_edge_weights: Whether edge weights are learnable
            shared_embedding_dim: Dimension of shared embedding from encoder (if different from hidden_dim)
        """
        super().__init__()
        self.num_tasks = num_tasks
        self.task_embed_dim = task_embed_dim
        self.hidden_dim = hidden_dim
        self.scale_type = scale_type
        self.use_learnable_edge_weights = use_learnable_edge_weights
        self.shared_embedding_dim = shared_embedding_dim if shared_embedding_dim else hidden_dim
        
        # Select clusters and cross-cluster edges based on scale type
        if scale_type == "PHQ-8":
            self.clusters = self.PHQ8_CLUSTERS
            self.cross_cluster_edges = self.PHQ8_CROSS_CLUSTER_EDGES
        else:
            self.clusters = self.HAMD13_CLUSTERS
            self.cross_cluster_edges = self.HAMD13_CROSS_CLUSTER_EDGES
        
        # Learnable task embeddings
        self.task_embeddings = nn.Parameter(torch.randn(num_tasks, task_embed_dim))
        nn.init.xavier_uniform_(self.task_embeddings)
        
        # Learnable edge weight hyperparameters
        if use_learnable_edge_weights:
            self.omega_intra = nn.Parameter(torch.tensor(intra_cluster_weight_init))
            self.omega_cross = nn.Parameter(torch.tensor(cross_cluster_weight_init))
        else:
            self.register_buffer('omega_intra', torch.tensor(intra_cluster_weight_init))
            self.register_buffer('omega_cross', torch.tensor(cross_cluster_weight_init))
        
        # Build adjacency matrix and edge type matrix
        self._build_graph_structure()
        
        # Input projection for task embeddings
        self.task_input_proj = nn.Linear(task_embed_dim, hidden_dim)
        
        # Projection for shared embedding (if dimension differs from hidden_dim)
        if self.shared_embedding_dim != hidden_dim:
            self.shared_proj = nn.Linear(self.shared_embedding_dim, hidden_dim)
        else:
            self.shared_proj = nn.Identity()
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        for i in range(num_gat_layers):
            self.gat_layers.append(
                MultiHeadGAT(
                    in_features=hidden_dim,
                    out_features=hidden_dim,
                    num_nodes=num_tasks,
                    num_heads=num_heads,
                    dropout=dropout,
                    use_edge_weights=True
                )
            )
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_gat_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def _build_graph_structure(self):
        """Build adjacency matrix and edge type matrix based on clinical clusters."""
        num_tasks = self.num_tasks
        
        # Initialize adjacency matrix (with self-loops)
        adj = torch.eye(num_tasks)
        
        # Edge type matrix: 0 = no edge, 1 = intra-cluster, 2 = cross-cluster
        edge_types = torch.zeros(num_tasks, num_tasks)
        
        # Add intra-cluster edges (fully connected within each cluster)
        for cluster_name, cluster_indices in self.clusters.items():
            for i in cluster_indices:
                for j in cluster_indices:
                    if i < num_tasks and j < num_tasks:
                        adj[i, j] = 1.0
                        if i != j:
                            edge_types[i, j] = 1  # Intra-cluster
        
        # Add cross-cluster edges
        for i, j in self.cross_cluster_edges:
            if i < num_tasks and j < num_tasks:
                adj[i, j] = 1.0
                adj[j, i] = 1.0  # Symmetric
                edge_types[i, j] = 2  # Cross-cluster
                edge_types[j, i] = 2
        
        self.register_buffer('adj', adj)
        self.register_buffer('edge_types', edge_types)
        
    def get_edge_weights(self) -> torch.Tensor:
        """
        Compute edge weight matrix from learnable parameters.
        
        Returns:
            Edge weight matrix [num_tasks, num_tasks]
        """
        edge_weights = torch.ones_like(self.adj)
        
        # Apply intra-cluster weights
        intra_mask = (self.edge_types == 1)
        edge_weights = torch.where(intra_mask, self.omega_intra * torch.ones_like(edge_weights), edge_weights)
        
        # Apply cross-cluster weights
        cross_mask = (self.edge_types == 2)
        edge_weights = torch.where(cross_mask, self.omega_cross * torch.ones_like(edge_weights), edge_weights)
        
        # Self-loops have weight 1.0
        edge_weights = edge_weights * self.adj
        
        return edge_weights
    
    def forward(
        self,
        shared_embedding: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Task Graph GAT.
        
        Args:
            shared_embedding: Shared representation from encoder [batch_size, shared_embedding_dim]
            
        Returns:
            task_embeddings: Updated task embeddings [batch_size, num_tasks, hidden_dim]
            attention_weights: Attention weights for visualization (optional)
        """
        batch_size = shared_embedding.size(0)
        
        # Project shared embedding to hidden_dim if needed
        shared_projected = self.shared_proj(shared_embedding)  # [batch_size, hidden_dim]
        
        # Initialize task node features
        # Combine learnable task embeddings with shared embedding
        task_h = self.task_input_proj(self.task_embeddings)  # [num_tasks, hidden_dim]
        task_h = task_h.unsqueeze(0).expand(batch_size, -1, -1)  # [B, num_tasks, hidden_dim]
        
        # Add projected shared embedding to all task nodes
        shared_expanded = shared_projected.unsqueeze(1).expand(-1, self.num_tasks, -1)
        task_h = task_h + shared_expanded  # Residual connection with shared embedding
        
        # Get edge weights
        edge_weights = self.get_edge_weights()
        
        # Apply GAT layers
        for i, (gat_layer, layer_norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            task_h_new = gat_layer(task_h, self.adj, edge_weights)
            task_h = layer_norm(task_h + task_h_new)  # Residual connection
            task_h = self.dropout(task_h)
        
        return task_h, edge_weights
    
    def get_graph_info(self) -> Dict:
        """
        Get graph structure information for visualization/debugging.
        
        Returns:
            Dictionary with graph information
        """
        return {
            'num_tasks': self.num_tasks,
            'scale_type': self.scale_type,
            'clusters': self.clusters,
            'adjacency_matrix': self.adj.cpu().numpy(),
            'edge_types': self.edge_types.cpu().numpy(),
            'omega_intra': self.omega_intra.item() if isinstance(self.omega_intra, nn.Parameter) else self.omega_intra.item(),
            'omega_cross': self.omega_cross.item() if isinstance(self.omega_cross, nn.Parameter) else self.omega_cross.item(),
            'edge_weights': self.get_edge_weights().cpu().detach().numpy()
        }


class TaskGraphFusion(nn.Module):
    """
    Fusion module to combine shared embeddings with task-specific embeddings
    from the Task Graph for final prediction.
    """
    
    def __init__(
        self,
        shared_dim: int,
        task_dim: int,
        output_dim: int = 1,
        fusion_type: str = "concat",  # "concat", "add", "gate"
        dropout: float = 0.1
    ):
        """
        Initialize fusion module.
        
        Args:
            shared_dim: Dimension of shared embedding
            task_dim: Dimension of task embedding from GAT
            output_dim: Output dimension (1 for regression, num_classes for classification)
            fusion_type: Type of fusion ("concat", "add", "gate")
            dropout: Dropout rate
        """
        super().__init__()
        self.fusion_type = fusion_type
        
        if fusion_type == "concat":
            self.fusion_proj = nn.Sequential(
                nn.Linear(shared_dim + task_dim, shared_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(shared_dim, output_dim)
            )
        elif fusion_type == "add":
            # Project task_dim to shared_dim if different
            if task_dim != shared_dim:
                self.task_proj = nn.Linear(task_dim, shared_dim)
            else:
                self.task_proj = nn.Identity()
            self.output_proj = nn.Linear(shared_dim, output_dim)
        elif fusion_type == "gate":
            # Gated fusion
            self.gate = nn.Sequential(
                nn.Linear(shared_dim + task_dim, shared_dim),
                nn.Sigmoid()
            )
            if task_dim != shared_dim:
                self.task_proj = nn.Linear(task_dim, shared_dim)
            else:
                self.task_proj = nn.Identity()
            self.output_proj = nn.Linear(shared_dim, output_dim)
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        shared_embedding: torch.Tensor,
        task_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse shared and task embeddings.
        
        Args:
            shared_embedding: [batch_size, shared_dim]
            task_embedding: [batch_size, task_dim]
            
        Returns:
            output: [batch_size, output_dim]
        """
        if self.fusion_type == "concat":
            fused = torch.cat([shared_embedding, task_embedding], dim=-1)
            return self.fusion_proj(fused)
        elif self.fusion_type == "add":
            task_proj = self.task_proj(task_embedding)
            fused = shared_embedding + task_proj
            return self.output_proj(self.dropout(fused))
        elif self.fusion_type == "gate":
            concat = torch.cat([shared_embedding, task_embedding], dim=-1)
            gate = self.gate(concat)
            task_proj = self.task_proj(task_embedding)
            fused = gate * shared_embedding + (1 - gate) * task_proj
            return self.output_proj(self.dropout(fused))


if __name__ == "__main__":
    # Test the Task Graph GAT module
    print("Testing TaskGraphGAT...")
    
    # PHQ-8 test
    batch_size = 4
    hidden_dim = 128
    num_tasks = 8
    
    # Create module
    task_graph = TaskGraphGAT(
        num_tasks=num_tasks,
        task_embed_dim=64,
        hidden_dim=hidden_dim,
        num_gat_layers=2,
        num_heads=4,
        scale_type="PHQ-8"
    )
    
    # Create dummy shared embedding
    shared_emb = torch.randn(batch_size, hidden_dim)
    
    # Forward pass
    task_embeddings, edge_weights = task_graph(shared_emb)
    
    print(f"Input shared embedding shape: {shared_emb.shape}")
    print(f"Output task embeddings shape: {task_embeddings.shape}")
    print(f"Edge weights shape: {edge_weights.shape}")
    
    # Print graph info
    graph_info = task_graph.get_graph_info()
    print(f"\nGraph Info:")
    print(f"  Scale type: {graph_info['scale_type']}")
    print(f"  Num tasks: {graph_info['num_tasks']}")
    print(f"  Clusters: {graph_info['clusters']}")
    print(f"  Omega intra: {graph_info['omega_intra']:.4f}")
    print(f"  Omega cross: {graph_info['omega_cross']:.4f}")
    print(f"  Adjacency matrix:\n{graph_info['adjacency_matrix']}")
    
    # Test fusion module
    print("\n\nTesting TaskGraphFusion...")
    fusion = TaskGraphFusion(
        shared_dim=hidden_dim,
        task_dim=hidden_dim,
        output_dim=1,
        fusion_type="gate"
    )
    
    # Get one task embedding
    task_emb = task_embeddings[:, 0, :]  # [batch_size, hidden_dim]
    output = fusion(shared_emb, task_emb)
    print(f"Fusion output shape: {output.shape}")
    
    print("\nAll tests passed!")

