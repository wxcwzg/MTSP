"""
Cluster-based constraint loss for PHQ-8 and HAMD-13 subscale prediction.
This loss enforces correlations between related subscales based on empirical data analysis.
"""
import torch
import torch.nn as nn


class ClusterConstraintLoss(nn.Module):
    """
    Prior knowledge constraint loss based on PHQ-8 or HAMD-13 subscale clustering.
    
    Enforces that predictions of highly correlated subscales should be similar,
    based on empirical correlation analysis or clinical knowledge.
    
    For PHQ-8:
    - Cluster 1 (Core Depression): Subscales 1, 2, 6 (Anhedonia, Depressed mood, Worthlessness)
    - Cluster 2 (Cognitive Function): Subscales 7, 8 (Concentration, Psychomotor)
    - Cluster 3 (Somatic Symptoms): Subscales 3, 4, 5 (Sleep, Fatigue, Appetite)
    
    For HAMD-13:
    - Cluster 1 (Core Depression): Guilt (0), Suicide (1), Work_Interests (5)
    - Cluster 2 (Sleep Disorders): Insomnia_Initial (2), Insomnia_Middle (3), Insomnia_Late (4)
    - Cluster 3 (Anxiety): Psychic_Anxiety (6), Hypochondriasis (10)
    - Cluster 4 (Somatic Symptoms): GI_Symptoms (7), Somatic_Symptoms (8), Genital_Symptoms (9), Weight_Loss (11)
    - Cluster 5 (Insight): Insight (12) - standalone
    """
    
    def __init__(self, 
                 scale_type="PHQ-8",
                 cluster1_weight=1.0,
                 cluster2_weight=0.5, 
                 cluster3_weight=0.8,
                 cluster4_weight=0.6,  # For HAMD-13
                 cluster5_weight=0.3,  # For HAMD-13
                 reduction='mean'):
        """
        Args:
            scale_type: "PHQ-8" or "HAMD-13"
            cluster1_weight: Weight for core depression cluster (default: 1.0)
            cluster2_weight: Weight for cognitive/sleep cluster (default: 0.5)
            cluster3_weight: Weight for somatic/anxiety cluster (default: 0.8)
            cluster4_weight: Weight for somatic cluster (HAMD-13 only, default: 0.6)
            cluster5_weight: Weight for insight cluster (HAMD-13 only, default: 0.3)
            reduction: 'mean' or 'sum' for final loss aggregation
        """
        super().__init__()
        self.scale_type = scale_type
        self.cluster1_weight = cluster1_weight
        self.cluster2_weight = cluster2_weight
        self.cluster3_weight = cluster3_weight
        self.cluster4_weight = cluster4_weight
        self.cluster5_weight = cluster5_weight
        self.reduction = reduction
        
        if scale_type == "PHQ-8":
            # Empirical correlation weights from EDAIC dataset
            # Cluster 1: Core Depression (Subscales 1, 2, 6)
            self.cluster1_pairs = [
                (1, 5, 0.751),  # Sub2 ↔ Sub6 (indices 1, 5)
                (0, 1, 0.738),  # Sub1 ↔ Sub2 (indices 0, 1)
                (0, 5, 0.712),  # Sub1 ↔ Sub6 (indices 0, 5)
            ]
            
            # Cluster 2: Cognitive Function (Subscales 7, 8)
            self.cluster2_pairs = [
                (6, 7, 0.579),  # Sub7 ↔ Sub8 (indices 6, 7)
            ]
            
            # Cluster 3: Somatic Symptoms (Subscales 3, 4, 5)
            self.cluster3_pairs = [
                (2, 3, 0.669),  # Sub3 ↔ Sub4 (indices 2, 3)
                (3, 4, 0.621),  # Sub4 ↔ Sub5 (indices 3, 4)
                (2, 4, 0.564),  # Sub3 ↔ Sub5 (indices 2, 4)
            ]
            
            # PHQ-8 only has 3 clusters
            self.cluster4_pairs = []
            self.cluster5_pairs = []
            
        elif scale_type == "HAMD-13":
            # HAMD-13 clustering based on clinical knowledge
            # Cluster 1: Core Depression (Guilt, Suicide, Work_Interests)
            # Using moderate correlation weights (0.6-0.7) as default
            self.cluster1_pairs = [
                (0, 1, 0.70),   # Guilt ↔ Suicide (indices 0, 1)
                (0, 5, 0.65),   # Guilt ↔ Work_Interests (indices 0, 5)
                (1, 5, 0.68),   # Suicide ↔ Work_Interests (indices 1, 5)
            ]
            
            # Cluster 2: Sleep Disorders (Insomnia_Initial, Insomnia_Middle, Insomnia_Late)
            self.cluster2_pairs = [
                (2, 3, 0.75),   # Insomnia_Initial ↔ Insomnia_Middle (indices 2, 3)
                (3, 4, 0.75),   # Insomnia_Middle ↔ Insomnia_Late (indices 3, 4)
                (2, 4, 0.70),   # Insomnia_Initial ↔ Insomnia_Late (indices 2, 4)
            ]
            
            # Cluster 3: Anxiety (Psychic_Anxiety, Hypochondriasis)
            self.cluster3_pairs = [
                (6, 10, 0.65),  # Psychic_Anxiety ↔ Hypochondriasis (indices 6, 10)
            ]
            
            # Cluster 4: Somatic Symptoms (GI_Symptoms, Somatic_Symptoms, Genital_Symptoms, Weight_Loss)
            self.cluster4_pairs = [
                (7, 8, 0.70),   # GI_Symptoms ↔ Somatic_Symptoms (indices 7, 8)
                (8, 9, 0.65),   # Somatic_Symptoms ↔ Genital_Symptoms (indices 8, 9)
                (7, 11, 0.60), # GI_Symptoms ↔ Weight_Loss (indices 7, 11)
                (8, 11, 0.60), # Somatic_Symptoms ↔ Weight_Loss (indices 8, 11)
            ]
            
            # Cluster 5: Insight (standalone, but can be linked to core depression)
            # Insight is typically less correlated, so we use lower weight
            self.cluster5_pairs = [
                (0, 12, 0.50),  # Guilt ↔ Insight (indices 0, 12) - weak link
            ]
        else:
            raise ValueError(f"Unsupported scale_type: {scale_type}. Must be 'PHQ-8' or 'HAMD-13'")
    
    def forward(self, predictions):
        """
        Compute cluster constraint loss.
        
        Args:
            predictions: [batch_size, num_subscales] - Predictions for subscales
                        8 for PHQ-8, 13 for HAMD-13
        
        Returns:
            loss: Scalar constraint loss
        """
        batch_size = predictions.shape[0]
        device = predictions.device
        
        # Cluster 1: Core Depression
        loss_cluster1 = torch.tensor(0.0, device=device)
        for i, j, corr_weight in self.cluster1_pairs:
            # L2 distance weighted by correlation
            pair_loss = corr_weight * torch.mean((predictions[:, i] - predictions[:, j]) ** 2)
            loss_cluster1 += pair_loss
        
        # Cluster 2: Cognitive Function (PHQ-8) or Sleep Disorders (HAMD-13)
        loss_cluster2 = torch.tensor(0.0, device=device)
        for i, j, corr_weight in self.cluster2_pairs:
            pair_loss = corr_weight * torch.mean((predictions[:, i] - predictions[:, j]) ** 2)
            loss_cluster2 += pair_loss
        
        # Cluster 3: Somatic Symptoms (PHQ-8) or Anxiety (HAMD-13)
        loss_cluster3 = torch.tensor(0.0, device=device)
        for i, j, corr_weight in self.cluster3_pairs:
            pair_loss = corr_weight * torch.mean((predictions[:, i] - predictions[:, j]) ** 2)
            loss_cluster3 += pair_loss
        
        # Combine cluster losses with weights
        total_loss = (
            self.cluster1_weight * loss_cluster1 +
            self.cluster2_weight * loss_cluster2 +
            self.cluster3_weight * loss_cluster3
        )
        
        # Add Cluster 4 and 5 for HAMD-13
        if self.scale_type == "HAMD-13":
            # Cluster 4: Somatic Symptoms
            loss_cluster4 = torch.tensor(0.0, device=device)
            for i, j, corr_weight in self.cluster4_pairs:
                pair_loss = corr_weight * torch.mean((predictions[:, i] - predictions[:, j]) ** 2)
                loss_cluster4 += pair_loss
            
            # Cluster 5: Insight
            loss_cluster5 = torch.tensor(0.0, device=device)
            for i, j, corr_weight in self.cluster5_pairs:
                pair_loss = corr_weight * torch.mean((predictions[:, i] - predictions[:, j]) ** 2)
                loss_cluster5 += pair_loss
            
            total_loss = total_loss + (
                self.cluster4_weight * loss_cluster4 +
                self.cluster5_weight * loss_cluster5
            )
        
        if self.reduction == 'mean':
            # Normalize by total number of pairs
            num_pairs = (len(self.cluster1_pairs) + len(self.cluster2_pairs) + 
                        len(self.cluster3_pairs) + len(self.cluster4_pairs) + 
                        len(self.cluster5_pairs))
            if num_pairs > 0:
                total_loss = total_loss / num_pairs
        
        return total_loss
    
    def get_cluster_losses(self, predictions):
        """
        Get individual cluster losses for monitoring.
        
        Args:
            predictions: [batch_size, num_subscales] - Predictions for subscales
        
        Returns:
            dict: Individual cluster losses
        """
        device = predictions.device
        
        # Cluster 1
        loss_cluster1 = torch.tensor(0.0, device=device)
        for i, j, corr_weight in self.cluster1_pairs:
            pair_loss = corr_weight * torch.mean((predictions[:, i] - predictions[:, j]) ** 2)
            loss_cluster1 += pair_loss
        if len(self.cluster1_pairs) > 0:
            loss_cluster1 = loss_cluster1 / len(self.cluster1_pairs)
        
        # Cluster 2
        loss_cluster2 = torch.tensor(0.0, device=device)
        for i, j, corr_weight in self.cluster2_pairs:
            pair_loss = corr_weight * torch.mean((predictions[:, i] - predictions[:, j]) ** 2)
            loss_cluster2 += pair_loss
        if len(self.cluster2_pairs) > 0:
            loss_cluster2 = loss_cluster2 / len(self.cluster2_pairs)
        
        # Cluster 3
        loss_cluster3 = torch.tensor(0.0, device=device)
        for i, j, corr_weight in self.cluster3_pairs:
            pair_loss = corr_weight * torch.mean((predictions[:, i] - predictions[:, j]) ** 2)
            loss_cluster3 += pair_loss
        if len(self.cluster3_pairs) > 0:
            loss_cluster3 = loss_cluster3 / len(self.cluster3_pairs)
        
        result = {
            'cluster1_loss': loss_cluster1.item(),
            'cluster2_loss': loss_cluster2.item(),
            'cluster3_loss': loss_cluster3.item(),
        }
        
        # Add Cluster 4 and 5 for HAMD-13
        if self.scale_type == "HAMD-13":
            # Cluster 4
            loss_cluster4 = torch.tensor(0.0, device=device)
            for i, j, corr_weight in self.cluster4_pairs:
                pair_loss = corr_weight * torch.mean((predictions[:, i] - predictions[:, j]) ** 2)
                loss_cluster4 += pair_loss
            if len(self.cluster4_pairs) > 0:
                loss_cluster4 = loss_cluster4 / len(self.cluster4_pairs)
            
            # Cluster 5
            loss_cluster5 = torch.tensor(0.0, device=device)
            for i, j, corr_weight in self.cluster5_pairs:
                pair_loss = corr_weight * torch.mean((predictions[:, i] - predictions[:, j]) ** 2)
                loss_cluster5 += pair_loss
            if len(self.cluster5_pairs) > 0:
                loss_cluster5 = loss_cluster5 / len(self.cluster5_pairs)
            
            result['cluster4_loss'] = loss_cluster4.item()
            result['cluster5_loss'] = loss_cluster5.item()
            result['total_constraint_loss'] = (
                self.cluster1_weight * loss_cluster1 +
                self.cluster2_weight * loss_cluster2 +
                self.cluster3_weight * loss_cluster3 +
                self.cluster4_weight * loss_cluster4 +
                self.cluster5_weight * loss_cluster5
            ).item()
        else:
            result['total_constraint_loss'] = (
                self.cluster1_weight * loss_cluster1 +
                self.cluster2_weight * loss_cluster2 +
                self.cluster3_weight * loss_cluster3
            ).item()
        
        return result


def test_cluster_constraint_loss():
    """Test the cluster constraint loss."""
    print("Testing ClusterConstraintLoss...")
    
    # Create random predictions
    batch_size = 4
    predictions = torch.randn(batch_size, 8)
    
    # Test different configurations
    configs = [
        {"cluster1_weight": 1.0, "cluster2_weight": 0.5, "cluster3_weight": 0.8},
        {"cluster1_weight": 0.5, "cluster2_weight": 0.5, "cluster3_weight": 0.5},
        {"cluster1_weight": 2.0, "cluster2_weight": 1.0, "cluster3_weight": 1.5},
    ]
    
    for i, config in enumerate(configs):
        print(f"\nConfiguration {i+1}: {config}")
        criterion = ClusterConstraintLoss(**config)
        
        loss = criterion(predictions)
        print(f"  Total loss: {loss.item():.4f}")
        
        cluster_losses = criterion.get_cluster_losses(predictions)
        for key, value in cluster_losses.items():
            print(f"  {key}: {value:.4f}")
    
    # Test with similar predictions (should have low loss)
    print("\n" + "="*60)
    print("Test with similar predictions within clusters:")
    predictions_similar = torch.randn(batch_size, 8)
    # Make Cluster 1 subscales similar
    predictions_similar[:, 0] = predictions_similar[:, 1] = predictions_similar[:, 5] = torch.randn(batch_size)
    
    criterion = ClusterConstraintLoss()
    loss = criterion(predictions_similar)
    print(f"Loss with similar predictions: {loss.item():.4f}")
    
    cluster_losses = criterion.get_cluster_losses(predictions_similar)
    print("Individual cluster losses:")
    for key, value in cluster_losses.items():
        print(f"  {key}: {value:.4f}")
    
    print("\nTest passed!")


if __name__ == "__main__":
    test_cluster_constraint_loss()



