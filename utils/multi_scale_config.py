"""
Unified configuration for multi-scale depression assessment.
Supports both PHQ-8 and HAMD-13 scales.
"""
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ScaleConfig:
    """Configuration for a specific depression scale."""
    name: str  # "PHQ-8" or "HAMD-13"
    num_subscales: int  # 8 for PHQ-8, 13 for HAMD-13
    subscale_ranges: List[int]  # Max value for each subscale
    total_range: int  # Max total score
    subscale_names: List[str]  # Names of subscales
    
    @staticmethod
    def get_phq8_config():
        """Get PHQ-8 scale configuration."""
        return ScaleConfig(
            name="PHQ-8",
            num_subscales=8,
            subscale_ranges=[3, 3, 3, 3, 3, 3, 3, 3],  # All subscales: 0-3
            total_range=24,  # 8 * 3
            subscale_names=[
                "Anhedonia",
                "Depressed_Mood",
                "Sleep_Problems",
                "Fatigue",
                "Appetite_Changes",
                "Worthlessness",
                "Concentration",
                "Psychomotor"
            ]
        )
    
    @staticmethod
    def get_hamd13_config():
        """Get HAMD-13 scale configuration."""
        return ScaleConfig(
            name="HAMD-13",
            num_subscales=13,
            subscale_ranges=[4, 4, 2, 2, 2, 4, 4, 2, 2, 2, 4, 2, 2],
            total_range=36,  # sum of all max values
            subscale_names=[
                "Guilt",
                "Suicide",
                "Insomnia_Initial",
                "Insomnia_Middle",
                "Insomnia_Late",
                "Work_Interests",
                "Psychic_Anxiety",
                "GI_Symptoms",
                "Somatic_Symptoms",
                "Genital_Symptoms",
                "Hypochondriasis",
                "Weight_Loss",
                "Insight"
            ]
        )
    
    def get_normalization_ranges(self):
        """
        Get normalization min/max for each subscale.
        Returns: (min_list, max_list)
        """
        min_list = [0.0] * self.num_subscales
        max_list = [float(r) for r in self.subscale_ranges]
        return min_list, max_list


@dataclass
class MultiScaleTrainingConfig:
    """
    Unified training configuration for multiple depression scales.
    """
    
    def to_dict(self):
        return {key: getattr(self, key) for key in self.__annotations__}
    
    # ============================================================================
    # Scale Selection
    # ============================================================================
    scale_type: str = "PHQ-8"  # "PHQ-8" or "HAMD-13"
    dataset_name: str = "edaic"  # "edaic" (PHQ-8), "cidh" (HAMD-13), "pdch" (HAMD-13)
    
    # ============================================================================
    # Random Seed
    # ============================================================================
    seed: int = 1021# edaic seed 1021，pdch seed 1060
    
    # ============================================================================
    # Basic Training Parameters
    # ============================================================================
    batch_size: int = 50
    num_epochs: int = 80  # 15  
    early_stopping_patience: int = 15
    learning_rate: float = 2e-4
    shuffle: bool = True

    # Data parameters
    dataset_dir: str = "../data/EDAIC"
    label_file: str = "../data/EDAIC/labels/detailed_labels.csv"
    label_dim: int = 8  # 8 PHQ-8 subscales (for multi-task subscale supervision)
    sum_labels: bool = False  # False: use 8 subscales; True: use total score (0-24)
    label_per_utterance: bool = False  # False: one label per sequence

    # ============================================================================
    # Model Parameters
    # ============================================================================
    # BERT model selection:
    # - English Mental: "mental/mental-bert-base-uncased" (for PHQ-8)
    # - Chinese Medical: "medbert-base-wwm-chinese" (for HAMD-13)
    model_name: str = "mental/mental-bert-base-uncased"
    model_type: str = "transformer"
    hidden_dim: int = 200
    num_layers: int = 2
    bidirectional: bool = True
    dropout: float = 0.3
    transform_targets: bool = True

    # DataLoader parameters
    pin_memory: bool = False
    prefetch_factor: int = 2
    persistent_workers: bool = True

    # Learning rate scheduling
    lr_scheduler: str = "cosine"  # Options: "cosine", "plateau", "one_cycle", "step", "none"
    min_lr: float = 1e-4          # Minimum learning rate for schedulers
    lr_patience: int = 5          # Patience for ReduceLROnPlateau
    lr_factor: float = 0.5        # Factor by which to reduce LR on plateau
    lr_step_size: int = 15        # Epochs per step for StepLR
    lr_gamma: float = 0.1         # Multiplier for StepLR
    lr_warmup_epochs: int = 5     # Warmup epochs for OneCycleLR
    lr_pct_start: float = 0.3     # Percentage of cycle for LR increase in OneCycleLR

    # Loss function parameters
    loss_type: str = "mse"  # Options: "mse" (Mean Squared Error), "mae" (Mean Absolute Error)
    
    # Output head type for regression
    output_head_type: str = "linear"  # Options: "linear", "mlp", "mlp_bn", "mlp_deep"
    # - "linear": Simple linear layer (default)
    # - "mlp": 2-layer MLP (Linear -> ReLU -> Dropout -> Linear)
    # - "mlp_bn": 2-layer MLP with BatchNorm (Linear -> BN -> ReLU -> Dropout -> Linear)
    # - "mlp_deep": 3-layer MLP (Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> Linear)

    # ============================================================================
    # Label Configuration
    # ============================================================================
    label_dim: int = 13  # 8 for PHQ-8, 13 for HAMD-13
    #label_dim: int = 8  # 8 for PHQ-8, 13 for HAMD-13
    sum_labels: bool = False  # False: use subscales; True: use total score
    label_per_utterance: bool = False  # Not used for HAMD-13 (no utterances)
    
    # ============================================================================
    # Prediction Mode
    # ============================================================================
    prediction_mode: str = "regression"  # "regression" or "ordinal"
    num_ordinal_classes: int = 5  # For ordinal classification (PHQ-8: 4, HAMD-13: 5)
    #num_ordinal_classes: int = 4  # For ordinal classification (PHQ-8: 4, HAMD-13: 5)
    
    # ============================================================================
    # Normalization Configuration
    # ============================================================================
    use_normalization: bool = True  # Enable label normalization
    # For PHQ-8: uniform subscales (all 0-3)
    # For HAMD-13: per-subscale normalization ranges
    per_subscale_normalization: bool = True  # Enable for HAMD-13
    subscale_min_list: Optional[List[float]] = None
    subscale_max_list: Optional[List[float]] = None
    transform_targets: bool = True
    # ============================================================================
    # Cluster Constraint Loss (for PHQ-8 and HAMD-13)
    # ============================================================================
    # Based on hierarchical clustering of subscales correlation
    use_cluster_constraint: bool = False  # Enable cluster-based constraint loss
    cluster_constraint_weight: float = 0.01  # Weight for constraint loss (α in total loss)
    cluster1_weight: float = 1.0  # Weight for Cluster 1 (Core Depression)
    cluster2_weight: float = 0.5  # Weight for Cluster 2 (Cognitive/Sleep)
    cluster3_weight: float = 0.8  # Weight for Cluster 3 (Somatic/Anxiety)
    cluster4_weight: float = 0.6  # Weight for Cluster 4 (HAMD-13: Somatic Symptoms)
    cluster5_weight: float = 0.3  # Weight for Cluster 5 (HAMD-13: Insight)
    
    # ============================================================================
    # Task-Level Self-Paced Learning (SPL) parameters
    # ============================================================================
    # Recommended values based on training logs analysis:
    # - Initial subscale MAE: ~0.94, Final: ~0.56
    # - Initial total MAE: ~6.13, Final: ~2.33
    # STRICT CONFIG: Only activate easiest tasks initially (lambda=0.2 ~20% of initial MAE)
    use_task_spl: bool = False  # Enable Task-Level Self-Paced Learning
    spl_pace_function: str = "linear"  # "linear", "log", "self_paced", "mixture" - log for smoother growth
    # spl_initial_lambda: float = 0.5  # Initial threshold (lambda) - ~20% of initial subscale MAE (STRICT: only easiest tasks)
    # spl_final_lambda: float = 6.0  # Final threshold - ~1.5x final subscale MAE to include all tasks
    spl_initial_lambda: float = 0.2  # Initial threshold (lambda) - ~20% of initial subscale MAE (STRICT: only easiest tasks)
    spl_final_lambda: float = 2.0  # Final threshold - ~1.5x final subscale MAE to include all tasks
    spl_lambda_growth: str = "linear"  # "linear", "exponential", "step" - exponential for slow start, fast end
    spl_easy_ratio: float = 0.7  # Ratio of easy tasks to use initially (for mixture)
    spl_use_regularization: bool = False  # Use regularization term
    spl_reg_weight: float = 0.1  # Regularization weight
    spl_min_task_ratio: float = 0.25  # Minimum ratio of tasks to use (ensures at least 2 tasks)
    spl_difficulty_type: str = "standard"  # "standard", "gentle", "aggressive", "step"
    
    # ============================================================================
    # Task Graph GAT (Graph Attention Network) Configuration
    # ============================================================================
    # Task Graph models inter-task relationships based on clinical symptom clusters.
    # For PHQ-8: Core Depression, Cognitive Function, Somatic Symptoms
    # For HAMD-13: Cognitive, Sleep, Retardation, Anxiety/Somatization, Weight
    use_task_graph: bool = False  # Enable Task Graph GAT for task embedding learning
    task_graph_embed_dim: int = 64  # Dimension of learnable task embeddings
    task_graph_hidden_dim: int = 128  # Hidden dimension for GAT layers (will match model hidden_dim if None)
    task_graph_num_layers: int = 2  # Number of GAT layers
    task_graph_num_heads: int = 4  # Number of attention heads in GAT
    task_graph_dropout: float = 0.1  # Dropout rate for GAT
    
    # Learnable edge weight initialization (based on clinical priors)
    # - omega_intra: Weight for intra-cluster edges (strong within-cluster correlation)
    # - omega_cross: Weight for cross-cluster edges (weaker between-cluster correlation)
    task_graph_omega_intra: float = 1.0  # Initial weight for intra-cluster edges
    task_graph_omega_cross: float = 0.6  # Initial weight for cross-cluster edges (0.5-0.7)
    task_graph_learnable_weights: bool = True  # Whether edge weights are learnable
    
    # Fusion method for combining shared embedding with task embeddings
    # - "concat": Concatenate and project
    # - "add": Add task embedding to shared embedding
    # - "gate": Gated fusion with learnable gate
    task_graph_fusion_type: str = "gate"  # Fusion type: "concat", "add", "gate"
    
    # ============================================================================
    # Auto-configuration
    # ============================================================================
    dataset_dir: str = ""
    label_file: Optional[str] = None
    
    def __post_init__(self):
        """Auto-configure based on dataset_name."""
        self.auto_configure()
    
    def auto_configure(self):
        """Auto-configure based on dataset_name."""
        if self.dataset_name.lower() == "edaic":
            self.dataset_dir = "../data/EDAIC"
            self.label_file = "../data/EDAIC/labels/detailed_labels.csv"
            self.scale_type = "PHQ-8"
            self.model_name = "mental/mental-bert-base-uncased"
        elif self.dataset_name.lower() == "cidh":
            # Original dialogue format (CIDH_hamd13, 328 samples)
            self.dataset_dir = "../data"
            self.label_file = None  # JSON format, no separate label file
            self.scale_type = "HAMD-13"
            # CIDH is Chinese, use Chinese Medical BERT
            self.model_name = "medbert-base-wwm-chinese"
        elif self.dataset_name.lower() == "cidh":
            # Summarized format (CIDH_hamd_1690, 1689 samples)
            self.dataset_dir = "../data"
            self.label_file = None
            self.scale_type = "HAMD-13"
            self.model_name = "medbert-base-wwm-chinese"
        elif self.dataset_name.lower() == "pdch":
            self.dataset_dir = "../data"
            self.label_file = None
            self.scale_type = "HAMD-13"
            # PDCH is Chinese, use Chinese Medical BERT
            self.model_name = "medbert-base-wwm-chinese"
            # PDCH dataset is small (69 training samples), requires special small dataset configuration
            self.hidden_dim = 100  # Reduced from 200 to 100 to decrease model complexity
            self.num_layers = 1    # Reduced from 2 to 1 to decrease number of parameters
            self.dropout = 0.5     # Increased from 0.3 to 0.5 to increase regularization
            self.learning_rate = 5e-5  # Reduced from 2e-4 to 5e-5 to lower learning rate
            self.batch_size = 8     # Reduced from 10 to 8 for smaller batch size
            self.early_stopping_patience = 15  # Keep at 15 for early stopping to prevent overfitting
        else:
            raise ValueError(f"Unknown dataset_name: {self.dataset_name}. Supported: edaic, cidh, pdch")
        
        # Get scale configuration after scale_type is set
        if self.scale_type == "PHQ-8":
            scale_config = ScaleConfig.get_phq8_config()
        elif self.scale_type == "HAMD-13":
            scale_config = ScaleConfig.get_hamd13_config()
        else:
            raise ValueError(f"Unknown scale_type: {self.scale_type}")
        
        # Update label_dim based on scale
        self.label_dim = scale_config.num_subscales
        
        # Configure normalization based on scale
        if self.scale_type == "PHQ-8":
            # PHQ-8: uniform subscales (all 0-3)
            self.per_subscale_normalization = True
            self.normalization_min = 0.0
            self.normalization_max = 3.0
            self.subscale_min_list = None
            self.subscale_max_list = None
        elif self.scale_type == "HAMD-13":
            # HAMD-13: non-uniform subscales (mixed 0-2 and 0-4)
            self.per_subscale_normalization = True
            min_list, max_list = scale_config.get_normalization_ranges()
            self.subscale_min_list = min_list
            self.subscale_max_list = max_list
            # Set global min/max to common range for backward compatibility
            self.normalization_min = 0.0
            self.normalization_max = max(max_list)  # 4.0 for HAMD-13
        
        # Configure ordinal classes based on scale
        if self.prediction_mode == "ordinal":
            if self.scale_type == "PHQ-8":
                self.num_ordinal_classes = 4  # 0, 1, 2, 3
            elif self.scale_type == "HAMD-13":
                # For HAMD-13, use the maximum subscale range
                self.num_ordinal_classes = max(scale_config.subscale_ranges) + 1  # 5 (0-4)
        
        # Cluster constraint is now available for both PHQ-8 and HAMD-13
        # Keep the default value (True) for both scales
    
    def get_scale_config(self) -> ScaleConfig:
        """Get the scale configuration object."""
        if self.scale_type == "PHQ-8":
            return ScaleConfig.get_phq8_config()
        elif self.scale_type == "HAMD-13":
            return ScaleConfig.get_hamd13_config()
        else:
            raise ValueError(f"Unknown scale_type: {self.scale_type}")
    
    def print_config(self):
        """Print configuration summary."""
        print("="*80)
        print(f"Multi-Scale Training Configuration: {self.scale_type}")
        print("="*80)
        print(f"Dataset: {self.dataset_name}")
        print(f"Model: {self.model_name}")
        print(f"Prediction mode: {self.prediction_mode}")
        if self.prediction_mode == "regression":
            print(f"Output head type: {self.output_head_type}")
            print(f"Loss type: {self.loss_type}")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Hidden dim: {self.hidden_dim}")
        print(f"Num layers: {self.num_layers}")
        print(f"Dropout: {self.dropout}")
        scale_config = self.get_scale_config()
        print(f"Number of subscales: {scale_config.num_subscales}")
        if self.per_subscale_normalization:
            print(f"Per-subscale normalization: {self.subscale_min_list} to {self.subscale_max_list}")
            print(f"Subscale ranges: {scale_config.subscale_ranges}")
        else:
            print(f"Global normalization: [{self.normalization_min}, {self.normalization_max}]")
        
        print(f"Cluster constraint: {self.use_cluster_constraint} ({self.scale_type})")
        print(f"Task-level SPL: {self.use_task_spl}")
        print(f"Task-level SPL lambda: {self.spl_initial_lambda} to {self.spl_final_lambda}")
        
        # Task Graph GAT configuration
        print(f"Task Graph GAT: {self.use_task_graph}")
        if self.use_task_graph:
            print(f"  - Task embed dim: {self.task_graph_embed_dim}")
            print(f"  - GAT hidden dim: {self.task_graph_hidden_dim}")
            print(f"  - GAT layers: {self.task_graph_num_layers}")
            print(f"  - GAT heads: {self.task_graph_num_heads}")
            print(f"  - Omega intra: {self.task_graph_omega_intra}")
            print(f"  - Omega cross: {self.task_graph_omega_cross}")
            print(f"  - Learnable weights: {self.task_graph_learnable_weights}")
            print(f"  - Fusion type: {self.task_graph_fusion_type}")
        print("="*80)


def create_config_for_dataset(dataset_name: str, **kwargs) -> MultiScaleTrainingConfig:
    """Create configuration for a specific dataset."""
    config = MultiScaleTrainingConfig(dataset_name=dataset_name, **kwargs)
    return config


def get_hamd13_cidh_config(**kwargs) -> MultiScaleTrainingConfig:
    """Get HAMD-13 configuration for CIDH dataset."""
    return create_config_for_dataset("cidh", **kwargs)


def get_hamd13_pdch_config(**kwargs) -> MultiScaleTrainingConfig:
    """Get HAMD-13 configuration for PDCH dataset."""
    return create_config_for_dataset("pdch", **kwargs)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("Multi-Scale Training Configuration Test")
    print("="*80)
    
    # Test PHQ-8 (EDAIC)
    print("\n1. PHQ-8 Configuration (EDAIC)")
    print("-"*80)
    cfg_phq8 = create_config_for_dataset("edaic")
    cfg_phq8.print_config()
    
    # Test HAMD-13 (CIDH)
    print("\n2. HAMD-13 Configuration (CIDH)")
    print("-"*80)
    cfg_hamd13_eval = get_hamd13_cidh_config()
    cfg_hamd13_eval.print_config()
    
    # Test HAMD-13 (PDCH)
    print("\n3. HAMD-13 Configuration (PDCH)")
    print("-"*80)
    cfg_hamd13_pdch = get_hamd13_pdch_config()
    cfg_hamd13_pdch.print_config()
    
    # Test scale configs
    print("\n4. Scale Configurations")
    print("-"*80)
    
    phq8_scale = ScaleConfig.get_phq8_config()
    print(f"\nPHQ-8 Scale:")
    print(f"  Subscales: {phq8_scale.num_subscales}")
    print(f"  Ranges: {phq8_scale.subscale_ranges}")
    print(f"  Total range: 0-{phq8_scale.total_range}")
    print(f"  Names: {phq8_scale.subscale_names}")
    
    hamd13_scale = ScaleConfig.get_hamd13_config()
    print(f"\nHAMD-13 Scale:")
    print(f"  Subscales: {hamd13_scale.num_subscales}")
    print(f"  Ranges: {hamd13_scale.subscale_ranges}")
    print(f"  Total range: 0-{hamd13_scale.total_range}")
    print(f"  Names: {hamd13_scale.subscale_names}")
    
    print("\n" + "="*80)
    print("All tests completed!")
    print("="*80)

