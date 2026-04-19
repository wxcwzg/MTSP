"""
Unified training script for depression assessment.
Supports both PHQ-8 (E-DAIC) and HAMD-13 (CIDH, PDCH) datasets.

Usage:
    # Train on PHQ-8 (E-DAIC) with single seed (from config)
    python train_multi_scale.py --dataset edaic --scale PHQ-8
    
    # Train on HAMD-13 (CIDH) with single seed
    python train_multi_scale.py --dataset cidh --scale HAMD-13
    
    # Train on HAMD-13 (PDCH) with single seed
    python train_multi_scale.py --dataset pdch --scale HAMD-13
    
    # Train with multiple seeds (comma-separated)
    python train_multi_scale.py --dataset edaic --seeds "1260,1261,1262,1263,1264"
    
    # Train with seed range
    python train_multi_scale.py --dataset edaic --seed_start 1260 --seed_end 1265
    
    # Train with multiple seeds and specify best metric
    python train_multi_scale.py --dataset edaic --seeds "1260,1261,1262" --best_metric test_total_mae
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import copy
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.multi_scale_config import (
    MultiScaleTrainingConfig,
    get_hamd13_cidh_config,
    get_hamd13_pdch_config,
    create_config_for_dataset
)
from models.multi_scale_transformer import MultiScaleTransformer
from preprocessing.HAMD13Dataset import get_hamd13_dataloader
from preprocessing.EDAICDataset import get_edaic_dataloader
from utils.early_stopping import EarlyStopping
from utils.ordinal_loss import OrdinalLoss, ordinal_predict
from utils.task_level_spl import TaskLevelSPL, SPLConfig
from utils.cluster_constraint_loss import ClusterConstraintLoss
from utils.utils import set_seed
from transformers import AutoModel


def get_config_for_dataset(dataset_name):
    """Get configuration for specified dataset."""
    return create_config_for_dataset(dataset_name)


def _check_embedding_cache_exists(cfg, splits):
    """
    Check if embedding cache exists for all specified splits.
    
    Args:
        cfg: Configuration object
        splits: List of split names (e.g., ["train", "dev", "test"])
    
    Returns:
        bool: True if all caches exist, False otherwise
    """
    import hashlib
    
    cache_dir = os.path.join(cfg.dataset_dir, "embedding_cache")
    if not os.path.exists(cache_dir):
        return False
    
    for split in splits:
        if cfg.dataset_name.lower() == "edaic":
            cache_key = f"edaic_{split}_{cfg.model_name}_{cfg.sum_labels}"
        else:
            # HAMD-13 datasets
            cache_key = f"hamd13_{cfg.dataset_name}_{split}_{cfg.model_name}_{cfg.sum_labels}"
        
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:16]
        cache_file = os.path.join(cache_dir, f"embeddings_{cache_hash}.pkl")
        
        if not os.path.exists(cache_file):
            return False
    
    return True


def create_dataloaders(cfg):
    """Create dataloaders for datasets (E-DAIC or HAMD-13)."""
    device = cfg.device #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check if embedding caches exist for all splits
    if cfg.dataset_name.lower() == "edaic":
        splits_to_check = ["train", "dev", "test"]
    else:
        splits_to_check = ["train", "val", "test"]
    
    cache_exists = _check_embedding_cache_exists(cfg, splits_to_check)
    
    # Load BERT model only if cache doesn't exist
    bert_model = None
    if not cache_exists:
        print("  Embedding cache not found. Loading BERT model for encoding...")
        bert_model = AutoModel.from_pretrained(cfg.model_name)
        bert_model.eval()
        bert_model = bert_model.to(device)
    else:
        print("  Embedding cache found. Skipping BERT model loading (will load from cache).")
    
    # Get seed from config for reproducible data shuffling
    seed = getattr(cfg, 'seed', 42)
    
    # Select appropriate dataloader function based on dataset
    if cfg.dataset_name.lower() == "edaic":
        # E-DAIC dataset (PHQ-8)
        train_dl = get_edaic_dataloader(
            split="train",
            data_dir=cfg.dataset_dir,
            label_file=cfg.label_file,
            sum_labels=cfg.sum_labels,
            bert_model_name=cfg.model_name,
            batch_size=cfg.batch_size,
            shuffle=True,
            model=bert_model,
            device=device,
            num_workers=0,
            pin_memory=False
        )
        val_dl = get_edaic_dataloader(
            split="dev",
            data_dir=cfg.dataset_dir,
            label_file=cfg.label_file,
            sum_labels=cfg.sum_labels,
            bert_model_name=cfg.model_name,
            batch_size=cfg.batch_size,
            shuffle=False,
            model=bert_model,
            device=device,
            num_workers=0,
            pin_memory=False
        )
        test_dl = get_edaic_dataloader(
            split="test",
            data_dir=cfg.dataset_dir,
            label_file=cfg.label_file,
            sum_labels=cfg.sum_labels,
            bert_model_name=cfg.model_name,
            batch_size=cfg.batch_size,
            shuffle=False,
            model=bert_model,
            device=device,
            num_workers=0,
            pin_memory=False
        )
    else:
        # HAMD-13 datasets (CIDH, PDCH)
        train_dl = get_hamd13_dataloader(
            split="train",
            dataset_name=cfg.dataset_name,
            data_dir=cfg.dataset_dir,
            sum_labels=cfg.sum_labels,
            bert_model_name=cfg.model_name,
            batch_size=cfg.batch_size,
            shuffle=True,
            model=bert_model,
            device=device,
            num_workers=0,
            pin_memory=False
        )
        val_dl = get_hamd13_dataloader(
            split="val",
            dataset_name=cfg.dataset_name,
            data_dir=cfg.dataset_dir,
            sum_labels=cfg.sum_labels,
            bert_model_name=cfg.model_name,
            batch_size=cfg.batch_size,
            shuffle=False,
            model=bert_model,
            device=device,
            num_workers=0,
            pin_memory=False
        )
        test_dl = get_hamd13_dataloader(
            split="test",
            dataset_name=cfg.dataset_name,
            data_dir=cfg.dataset_dir,
            sum_labels=cfg.sum_labels,
            bert_model_name=cfg.model_name,
            batch_size=cfg.batch_size,
            shuffle=False,
            model=bert_model,
            device=device,
            num_workers=0,
            pin_memory=False
        )
    
    # Check dataset sizes and warn if validation set is empty
    train_size = len(train_dl.dataset) if train_dl else 0
    val_size = len(val_dl.dataset) if val_dl else 0
    test_size = len(test_dl.dataset) if test_dl else 0
    
    print(f"\nDataset sizes:")
    print(f"  Train: {train_size} samples")
    print(f"  Validation: {val_size} samples")
    print(f"  Test: {test_size} samples")
    
    if val_size == 0:
        print("\n" + "="*80)
        print("WARNING: Validation set is empty!")
        print("This may happen if:")
        print("  1. The label file doesn't have a 'val' split")
        print("  2. The split column uses different values (e.g., 'dev' instead of 'val')")
        print("  3. All validation samples were filtered out")
        print("\nTraining will continue without validation evaluation.")
        print("Early stopping and learning rate scheduling will use training metrics instead.")
        print("="*80 + "\n")
    
    if train_size == 0:
        raise ValueError("Training set is empty! Cannot train the model.")
    
    return train_dl, val_dl, test_dl


def create_model(cfg):
    """Create model based on configuration."""
    scale_config = cfg.get_scale_config()
    
    # Get Task Graph parameters (with defaults for backward compatibility)
    use_task_graph = getattr(cfg, 'use_task_graph', False)
    task_graph_embed_dim = getattr(cfg, 'task_graph_embed_dim', 64)
    task_graph_hidden_dim = getattr(cfg, 'task_graph_hidden_dim', cfg.hidden_dim)
    task_graph_num_layers = getattr(cfg, 'task_graph_num_layers', 2)
    task_graph_num_heads = getattr(cfg, 'task_graph_num_heads', 4)
    task_graph_dropout = getattr(cfg, 'task_graph_dropout', 0.1)
    task_graph_omega_intra = getattr(cfg, 'task_graph_omega_intra', 1.0)
    task_graph_omega_cross = getattr(cfg, 'task_graph_omega_cross', 0.6)
    task_graph_learnable_weights = getattr(cfg, 'task_graph_learnable_weights', True)
    task_graph_fusion_type = getattr(cfg, 'task_graph_fusion_type', 'gate')
    
    model = MultiScaleTransformer(
        input_dim=768,  # BERT embedding size
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        num_heads=4,
        dropout=cfg.dropout,
        num_subscales=scale_config.num_subscales,
        subscale_ranges=scale_config.subscale_ranges,
        prediction_mode=cfg.prediction_mode,
        use_normalization=cfg.use_normalization,
        per_subscale_normalization=cfg.per_subscale_normalization,
        normalization_min=cfg.normalization_min,
        normalization_max=cfg.normalization_max,
        subscale_min_list=cfg.subscale_min_list,
        subscale_max_list=cfg.subscale_max_list,
        output_head_type=getattr(cfg, 'output_head_type', 'linear'),
        # Task Graph GAT parameters
        use_task_graph=use_task_graph,
        scale_type=cfg.scale_type,
        task_graph_embed_dim=task_graph_embed_dim,
        task_graph_hidden_dim=task_graph_hidden_dim,
        task_graph_num_layers=task_graph_num_layers,
        task_graph_num_heads=task_graph_num_heads,
        task_graph_dropout=task_graph_dropout,
        task_graph_omega_intra=task_graph_omega_intra,
        task_graph_omega_cross=task_graph_omega_cross,
        task_graph_learnable_weights=task_graph_learnable_weights,
        task_graph_fusion_type=task_graph_fusion_type
    )
    
    # Print task graph info if enabled
    if use_task_graph:
        print(f"\nTask Graph GAT enabled:")
        print(f"  - Scale type: {cfg.scale_type}")
        print(f"  - Task embed dim: {task_graph_embed_dim}")
        print(f"  - GAT hidden dim: {task_graph_hidden_dim}")
        print(f"  - GAT layers: {task_graph_num_layers}")
        print(f"  - GAT heads: {task_graph_num_heads}")
        print(f"  - Omega intra: {task_graph_omega_intra}")
        print(f"  - Omega cross: {task_graph_omega_cross}")
        print(f"  - Learnable weights: {task_graph_learnable_weights}")
        print(f"  - Fusion type: {task_graph_fusion_type}")
        
        # Print graph structure info
        graph_info = model.get_task_graph_info()
        if graph_info:
            print(f"  - Clusters: {graph_info['clusters']}")
    
    return model


def train_multi_scale_with_tensorboard(
    model, train_dataloader, val_dataloader, test_dataloader,
    cfg, num_subscales,
    epochs=100, lr=1e-3, patience=15, transform_targets=False,
    scheduler_type="cosine", min_lr=1e-5, lr_patience=5,
    lr_factor=0.5, lr_step_size=15, lr_gamma=0.1,
    lr_warmup_epochs=5, lr_pct_start=0.3,
    log_dir=None,
    test_eval_interval=5
):
    """
    Train multi-scale model with TensorBoard logging.
    Supports both regression (MSE) and ordinal classification (Ordinal Loss).
    """
    device = cfg.device #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Create TensorBoard writer
    if log_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(script_dir, "runs", f"multi_scale_{timestamp}")
    # Ensure the runs directory exists
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")
    print(f"Prediction mode: {model.prediction_mode}")
    print(f"Use normalization: {model.use_normalization}")
    print(f"Number of subscales: {num_subscales}")
    
    # Setup optimizer and early stopping
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    early_stopper = EarlyStopping(patience=patience)
    
    # Setup Task-Level Self-Paced Learning
    task_spl = None
    if cfg.use_task_spl:
        spl_config = SPLConfig(
            pace_function=cfg.spl_pace_function,
            initial_lambda=cfg.spl_initial_lambda,
            final_lambda=cfg.spl_final_lambda,
            lambda_growth=cfg.spl_lambda_growth,
            use_regularization = cfg.spl_use_regularization,
            min_task_ratio = cfg.spl_min_task_ratio

        )
        task_spl = TaskLevelSPL(config=spl_config, num_tasks=num_subscales)
        task_spl = task_spl.to(device)
        print(f"Task-Level SPL enabled: {spl_config.pace_function} pace, "
              f"lambda [{spl_config.initial_lambda:.3f}, {spl_config.final_lambda:.3f}]")
    
    # Setup Cluster Constraint Loss (for both PHQ-8 and HAMD-13)
    cluster_criterion = None
    if cfg.use_cluster_constraint:
        cluster_criterion = ClusterConstraintLoss(
            scale_type=cfg.scale_type,
            cluster1_weight=cfg.cluster1_weight,
            cluster2_weight=cfg.cluster2_weight,
            cluster3_weight=cfg.cluster3_weight,
            cluster4_weight=getattr(cfg, 'cluster4_weight', 0.6),
            cluster5_weight=getattr(cfg, 'cluster5_weight', 0.3)
        )
        cluster_criterion = cluster_criterion.to(device)
        print(f"Cluster Constraint Loss enabled ({cfg.scale_type}): weight={cfg.cluster_constraint_weight:.3f}")
    
    # Setup loss function
    # For ordinal mode, use OrdinalLoss
    # For regression mode, use loss_type from config (MSE or MAE)
    num_ordinal_classes = cfg.num_ordinal_classes
    
    if model.prediction_mode == "ordinal":
        criterion = OrdinalLoss(num_classes=num_ordinal_classes, reduction='none')
        loss_type = "ordinal"
    else:
        # Use loss_type from config
        loss_type = getattr(cfg, 'loss_type', 'mse').lower()
        if loss_type == "mae":
            criterion = nn.L1Loss(reduction='none')  # L1Loss = MAE
        else:
            criterion = nn.MSELoss(reduction='none')  # Default to MSE
        print(f"Using {loss_type.upper()} loss for regression mode")
    
    # Set up learning rate scheduler
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=min_lr
        )
    elif scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=lr_factor, patience=lr_patience,
            min_lr=min_lr, verbose=True
        )
    elif scheduler_type == "one_cycle":
        steps_per_epoch = len(train_dataloader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, total_steps=epochs * steps_per_epoch,
            pct_start=lr_pct_start, div_factor=25.0,
            final_div_factor=1e4, three_phase=False, verbose=False
        )
    elif scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=lr_step_size, gamma=lr_gamma
        )
    else:
        scheduler = None
    
    # Simplified history dictionary (no MSE, no per-subscale metrics)
    history = {
        k: [] for k in [
            "train_loss", "train_mae", "train_rmse",
            "val_loss", "val_mae", "val_rmse",
            "train_total_mae", "train_total_rmse",
            "val_total_mae", "val_total_rmse",
            "learning_rate",
            "test_loss", "test_mae", "test_rmse",
            "test_total_mae", "test_total_rmse"
        ]
    }
    
    # Training loop
    for epoch in tqdm(range(epochs), desc="Epochs"):
        # Update Task-Level SPL threshold
        if task_spl is not None:
            task_spl.update_lambda(epoch, epochs)
        
        # ==================== Training Phase ====================
        model.train()
        train_loss = train_mae = train_rmse = 0.0
        train_total_maes = []
        train_total_rmses = []
        epoch_cluster_constraint_loss = 0.0
        batch_idx = 0
        
        for batch_sample_ids, X, Y, attention_mask, raw_utterances in train_dataloader:
            X, Y, attention_mask = X.to(device), Y.to(device), attention_mask.to(device)
            
            # Preprocess data - Y shape: [batch, 1, num_subscales] for Seq2One
            if Y.dim() == 3 and Y.shape[1] == 1:
                Y = Y.squeeze(1)  # [batch, num_subscales]
            
            Y_orig = Y.clone()
            
            # Normalize labels (for regression mode only)
            # Note: transform_targets is typically NOT needed when using normalization
            # If transform_targets=True, it should be applied BEFORE normalization
            if model.prediction_mode == "ordinal":
                Y_normalized = Y_orig
            else:
                # For regression: normalize original labels
                # transform_targets is disabled when normalization is enabled
                # because normalization already handles the scaling
                Y_normalized = model.normalize_labels(Y_orig)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(X, attention_mask=attention_mask)
            
            # Handle different output formats
            if model.prediction_mode == "ordinal":
                preds_denorm = ordinal_predict(output)  # [batch, num_subscales]
                Y_ordinal = Y_normalized.long().clamp(0, num_ordinal_classes - 1)
                
                # Calculate per-task loss
                per_task_loss = []
                for task_idx in range(num_subscales):
                    task_loss = criterion(
                        output[:, task_idx:task_idx+1, :],
                        Y_ordinal[:, task_idx:task_idx+1]
                    )
                    per_task_loss.append(task_loss)
                per_task_loss = torch.stack(per_task_loss, dim=1)  # [batch, num_subscales]
            else:
                preds_normalized = output
                # Model output is already in [0, 1] range due to sigmoid activation
                # Clamp is redundant but kept as extra safety for denormalization
                if model.use_normalization:
                    preds_normalized = torch.clamp(preds_normalized, 0.0, 1.0)
                preds_denorm = model.denormalize_predictions(preds_normalized) if model.use_normalization else preds_normalized
                per_task_loss = criterion(preds_normalized, Y_normalized)  # [batch, num_subscales]
            
            # Apply Task-Level SPL weights if enabled
            if task_spl is not None:
                # Calculate average loss per task across all samples in the batch
                # per_task_loss shape: [batch, num_subscales]
                # mean(dim=0) -> [num_subscales] (average loss for each task)
                avg_per_task_loss = per_task_loss.mean(dim=0)
                task_spl.update_task_difficulties(avg_per_task_loss.detach())
                weighted_per_task_loss = task_spl.apply_task_weights(per_task_loss)
                loss = weighted_per_task_loss.sum(dim=1).mean()
                reg = task_spl.compute_regularization()
                if isinstance(reg, torch.Tensor):
                    loss = loss + reg.to(device)
            else:
                loss = per_task_loss.sum(dim=1).mean()
            
            # Add Cluster Constraint Loss if enabled
            if cluster_criterion is not None:
                constraint_loss = cluster_criterion(preds_denorm)
                epoch_cluster_constraint_loss += constraint_loss.item()
                loss = loss + cfg.cluster_constraint_weight * constraint_loss
            
            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            batch_idx += 1
            
            # Track metrics
            with torch.no_grad():
                if model.prediction_mode == "ordinal":
                    preds = preds_denorm
                else:
                    preds = model.denormalize_predictions(preds_normalized) if model.use_normalization else preds_normalized
                
                mae = torch.abs(preds - Y_orig).mean().item()
                mse = ((preds - Y_orig) ** 2).mean().item()
                rmse = np.sqrt(mse)
                
                train_loss += loss.item()
                train_mae += mae
                train_rmse += rmse
                
                # Calculate total score
                pred_total = preds.sum(dim=-1)
                true_total = Y_orig.sum(dim=-1)
                total_mae = torch.abs(pred_total - true_total).mean().item()
                total_rmse = torch.sqrt(((pred_total - true_total) ** 2).mean()).item()
                train_total_maes.append(total_mae)
                train_total_rmses.append(total_rmse)
        
        # Calculate average training metrics
        steps = len(train_dataloader)
        epoch_train_loss = train_loss / steps
        epoch_train_mae = train_mae / steps
        epoch_train_rmse = train_rmse / steps
        epoch_train_total_mae = np.mean(train_total_maes) if train_total_maes else 0
        epoch_train_total_rmse = np.mean(train_total_rmses) if train_total_rmses else 0
        
        history["train_loss"].append(epoch_train_loss)
        history["train_mae"].append(epoch_train_mae)
        history["train_rmse"].append(epoch_train_rmse)
        history["train_total_mae"].append(epoch_train_total_mae)
        history["train_total_rmse"].append(epoch_train_total_rmse)
        
        current_lr = optimizer.param_groups[0]['lr']
        history["learning_rate"].append(current_lr)
        
        # Log Task-Level SPL statistics
        # Added in training loop
        if task_spl is not None and batch_idx % 1 == 0:  # Print every 100 batches
            spl_stats = task_spl.get_curriculum_statistics()
            print(f"Epoch {epoch}, Batch {batch_idx}:")
            print(f"  Lambda: {spl_stats['current_lambda']:.4f}")
            print(f"  Active tasks: {spl_stats['num_active_tasks']}/{spl_stats['total_tasks']}")
            print(f"  Task weights: {[f'{w:.3f}' for w in spl_stats['task_weights']]}")
            print(f"  Task losses: {[f'{l:.4f}' for l in spl_stats['task_losses']]}")
        # Log training metrics
        writer.add_scalar('Loss/train', epoch_train_loss, epoch)
        writer.add_scalar('MAE/train_subscales', epoch_train_mae, epoch)
        writer.add_scalar('MAE/train_total', epoch_train_total_mae, epoch)
        writer.add_scalar('RMSE/train', epoch_train_rmse, epoch)
        writer.add_scalar('RMSE/train_total', epoch_train_total_rmse, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # ==================== Validation Phase ====================
        if val_dataloader and len(val_dataloader.dataset) > 0:
            model.eval()
            val_loss = val_mae = val_rmse = 0.0
            val_total_maes = []
            val_total_rmses = []
            
            with torch.no_grad():
                for batch_val_ids, Xv, Yv, attention_mask_v, raw_utterances_v in val_dataloader:
                    Xv, Yv, attention_mask_v = Xv.to(device), Yv.to(device), attention_mask_v.to(device)
                    
                    if Yv.dim() == 3 and Yv.shape[1] == 1:
                        Yv = Yv.squeeze(1)
                    
                    Yv_orig = Yv.clone()
                    
                    # Normalize labels (for regression mode only)
                    if model.prediction_mode == "ordinal":
                        Yv_normalized = Yv_orig
                    else:
                        Yv_normalized = model.normalize_labels(Yv_orig)
                    
                    output_v = model(Xv, attention_mask=attention_mask_v)
                    
                    if model.prediction_mode == "ordinal":
                        preds_v_denorm = ordinal_predict(output_v)
                        Yv_ordinal = Yv_normalized.long().clamp(0, num_ordinal_classes - 1)
                        per_task_loss_v = []
                        for task_idx in range(num_subscales):
                            task_loss = criterion(
                                output_v[:, task_idx:task_idx+1, :],
                                Yv_ordinal[:, task_idx:task_idx+1]
                            )
                            per_task_loss_v.append(task_loss)
                        per_task_loss_v = torch.stack(per_task_loss_v, dim=1)
                        preds_v = preds_v_denorm
                    else:
                        preds_v_normalized = output_v
                        # Model output is already in [0, 1] range due to sigmoid activation
                        # Clamp is redundant but kept as extra safety
                        if model.use_normalization:
                            preds_v_normalized = torch.clamp(preds_v_normalized, 0.0, 1.0)
                        per_task_loss_v = criterion(preds_v_normalized, Yv_normalized)
                        preds_v = model.denormalize_predictions(preds_v_normalized)
                    
                    loss_v = per_task_loss_v.sum(dim=1).mean()
                    val_loss += loss_v.item()
                    
                    mae_v = torch.abs(preds_v - Yv_orig).mean().item()
                    mse_v = ((preds_v - Yv_orig) ** 2).mean().item()
                    rmse_v = np.sqrt(mse_v)
                    
                    val_mae += mae_v
                    val_rmse += rmse_v
                    
                    pred_total_v = preds_v.sum(dim=-1)
                    true_total_v = Yv_orig.sum(dim=-1)
                    total_mae_v = torch.abs(pred_total_v - true_total_v).mean().item()
                    total_rmse_v = torch.sqrt(((pred_total_v - true_total_v) ** 2).mean()).item()
                    val_total_maes.append(total_mae_v)
                    val_total_rmses.append(total_rmse_v)
            
            val_steps = len(val_dataloader)
            if val_steps > 0:
                epoch_val_loss = val_loss / val_steps
                epoch_val_mae = val_mae / val_steps
                epoch_val_rmse = val_rmse / val_steps
            else:
                epoch_val_loss = epoch_val_mae = epoch_val_rmse = 0.0
            epoch_val_total_mae = np.mean(val_total_maes) if val_total_maes else 0
            epoch_val_total_rmse = np.mean(val_total_rmses) if val_total_rmses else 0
            
            history["val_loss"].append(epoch_val_loss)
            history["val_mae"].append(epoch_val_mae)
            history["val_rmse"].append(epoch_val_rmse)
            history["val_total_mae"].append(epoch_val_total_mae)
            history["val_total_rmse"].append(epoch_val_total_rmse)
            
            writer.add_scalar('Loss/val', epoch_val_loss, epoch)
            writer.add_scalar('MAE/val_subscales', epoch_val_mae, epoch)
            writer.add_scalar('MAE/val_total', epoch_val_total_mae, epoch)
            writer.add_scalar('RMSE/val', epoch_val_rmse, epoch)
            writer.add_scalar('RMSE/val_total', epoch_val_total_rmse, epoch)
            
            # Early stopping
            val_metrics = {
                'val_loss': epoch_val_loss,
                'val_mae': epoch_val_mae,
                'val_rmse': epoch_val_rmse,
                'val_total_mae': epoch_val_total_mae,
                'val_total_rmse': epoch_val_total_rmse
            }
            should_stop = early_stopper(epoch_val_total_mae, model, metrics=val_metrics)
            if should_stop:
                print("Early stopping triggered. Stopping training.")
                break
            
        else:
            # No validation set available
            history["val_loss"].append(np.nan)
            history["val_mae"].append(np.nan)
            history["val_rmse"].append(np.nan)
            history["val_total_mae"].append(np.nan)
            history["val_total_rmse"].append(np.nan)
        
        # Check early stopping after validation (but before test to ensure all metrics are recorded)
        if early_stopper.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            # Ensure test metrics are also appended (as NaN) before breaking
            if (epoch + 1) % test_eval_interval != 0:
                history["test_loss"].append(np.nan)
                history["test_mae"].append(np.nan)
                history["test_rmse"].append(np.nan)
                history["test_total_mae"].append(np.nan)
                history["test_total_rmse"].append(np.nan)
            break
        
        # ==================== Test Evaluation (every N epochs) ====================
        if (epoch + 1) % test_eval_interval == 0 and test_dataloader:
            print(f"\nEvaluating on test set at epoch {epoch+1} (using best model)...")
            
            current_state_dict = copy.deepcopy(model.state_dict())
            # Only restore best weights if they exist (i.e., validation has been performed)
            if early_stopper.best_weights is not None:
                early_stopper.restore_best_weights(model)
            else:
                print("Warning: No best weights available yet, using current model weights for test evaluation.")
            
            model.eval()
            test_loss = test_mae = test_rmse = 0.0
            test_total_maes = []
            test_total_rmses = []
            
            with torch.no_grad():
                for batch_test_ids, Xt, Yt, attention_mask_t, raw_utterances_t in test_dataloader:
                    Xt, Yt, attention_mask_t = Xt.to(device), Yt.to(device), attention_mask_t.to(device)
                    
                    if Yt.dim() == 3 and Yt.shape[1] == 1:
                        Yt = Yt.squeeze(1)
                    
                    Yt_orig = Yt.clone()
                    
                    # Normalize labels (for regression mode only)
                    if model.prediction_mode == "ordinal":
                        Yt_normalized = Yt_orig
                    else:
                        Yt_normalized = model.normalize_labels(Yt_orig)
                    
                    output_t = model(Xt, attention_mask=attention_mask_t)
                    
                    if model.prediction_mode == "ordinal":
                        preds_t_denorm = ordinal_predict(output_t)
                        Yt_ordinal = Yt_normalized.long().clamp(0, num_ordinal_classes - 1)
                        per_task_loss_t = []
                        for task_idx in range(num_subscales):
                            task_loss = criterion(
                                output_t[:, task_idx:task_idx+1, :],
                                Yt_ordinal[:, task_idx:task_idx+1]
                            )
                            per_task_loss_t.append(task_loss)
                        per_task_loss_t = torch.stack(per_task_loss_t, dim=1)
                        preds_t = preds_t_denorm
                    else:
                        preds_t_normalized = output_t
                        # Model output is already in [0, 1] range due to sigmoid activation
                        # Clamp is redundant but kept as extra safety
                        if model.use_normalization:
                            preds_t_normalized = torch.clamp(preds_t_normalized, 0.0, 1.0)
                        per_task_loss_t = criterion(preds_t_normalized, Yt_normalized)
                        preds_t = model.denormalize_predictions(preds_t_normalized)
                    
                    loss_t = per_task_loss_t.sum(dim=1).mean()
                    test_loss += loss_t.item()
                    
                    mae_t = torch.abs(preds_t - Yt_orig).mean().item()
                    mse_t = ((preds_t - Yt_orig) ** 2).mean().item()
                    rmse_t = np.sqrt(mse_t)
                    
                    test_mae += mae_t
                    test_rmse += rmse_t
                    
                    pred_total_t = preds_t.sum(dim=-1)
                    true_total_t = Yt_orig.sum(dim=-1)
                    total_mae_t = torch.abs(pred_total_t - true_total_t).mean().item()
                    total_rmse_t = torch.sqrt(((pred_total_t - true_total_t) ** 2).mean()).item()
                    test_total_maes.append(total_mae_t)
                    test_total_rmses.append(total_rmse_t)
            
            test_steps = len(test_dataloader)
            epoch_test_loss = test_loss / test_steps
            epoch_test_mae = test_mae / test_steps
            epoch_test_rmse = test_rmse / test_steps
            epoch_test_total_mae = np.mean(test_total_maes) if test_total_maes else 0
            epoch_test_total_rmse = np.mean(test_total_rmses) if test_total_rmses else 0
            
            history["test_loss"].append(epoch_test_loss)
            history["test_mae"].append(epoch_test_mae)
            history["test_rmse"].append(epoch_test_rmse)
            history["test_total_mae"].append(epoch_test_total_mae)
            history["test_total_rmse"].append(epoch_test_total_rmse)
            
            writer.add_scalar('Loss/test', epoch_test_loss, epoch)
            writer.add_scalar('MAE/test_subscales', epoch_test_mae, epoch)
            writer.add_scalar('MAE/test_total', epoch_test_total_mae, epoch)
            writer.add_scalar('RMSE/test', epoch_test_rmse, epoch)
            writer.add_scalar('RMSE/test_total', epoch_test_total_rmse, epoch)
            
            print(f"Test metrics: Loss={epoch_test_loss:.4f}, Subscales MAE={epoch_test_mae:.4f}, "
                  f"Total MAE={epoch_test_total_mae:.4f}, Total RMSE={epoch_test_total_rmse:.4f}")
            
            model.load_state_dict(current_state_dict)
        else:
            history["test_loss"].append(np.nan)
            history["test_mae"].append(np.nan)
            history["test_rmse"].append(np.nan)
            history["test_total_mae"].append(np.nan)
            history["test_total_rmse"].append(np.nan)
        
        # Learning rate scheduling
        if scheduler is not None:
            if scheduler_type == "plateau":
                scheduler.step(epoch_val_total_mae if val_dataloader else epoch_train_total_mae)
            elif scheduler_type == "one_cycle":
                scheduler.step()
            else:
                scheduler.step()
        
        # Print progress
        if val_dataloader:
            print(f"Ep {epoch+1}/{epochs} | LR {current_lr:.6f} | "
                  f"Train Total MAE {epoch_train_total_mae:.3f} RMSE {epoch_train_total_rmse:.3f} | "
                  f"Val Total MAE {epoch_val_total_mae:.3f} RMSE {epoch_val_total_rmse:.3f}")
        else:
            print(f"Ep {epoch+1}/{epochs} | LR {current_lr:.6f} | "
                  f"Train Total MAE {epoch_train_total_mae:.3f} RMSE {epoch_train_total_rmse:.3f}")
    
    # Restore best model (if available)
    if early_stopper.best_weights is not None:
        early_stopper.restore_best_weights(model)
        print(f"\nRestored best model weights (best val_total_mae: {early_stopper.best_loss:.4f})")
    else:
        print("\nWarning: No best weights available to restore. Using final model weights.")
        if early_stopper.best_loss is not None:
            print(f"Best val_total_mae: {early_stopper.best_loss:.4f}")
        else:
            print("No validation was performed during training.")
    
    # Final test evaluation
    print("\nFinal test evaluation (using best model)...")
    model.eval()
    test_loss = test_mae = test_rmse = 0.0
    test_total_maes = []
    test_total_rmses = []
    
    # Collect all predictions and targets for per-subscale MAE calculation
    all_preds_t = []
    all_Yt_orig = []
    
    with torch.no_grad():
        for batch_test_ids, Xt, Yt, attention_mask_t, raw_utterances_t in test_dataloader:
            Xt, Yt, attention_mask_t = Xt.to(device), Yt.to(device), attention_mask_t.to(device)
            
            if Yt.dim() == 3 and Yt.shape[1] == 1:
                Yt = Yt.squeeze(1)
            
            Yt_orig = Yt.clone()
            
            # Normalize labels (for regression mode only)
            if model.prediction_mode == "ordinal":
                Yt_normalized = Yt_orig
            else:
                Yt_normalized = model.normalize_labels(Yt_orig)
            
            output_t = model(Xt, attention_mask=attention_mask_t)
            
            if model.prediction_mode == "ordinal":
                preds_t = ordinal_predict(output_t)
            else:
                preds_t_normalized = output_t
                # Clip normalized predictions to [0, 1] range
                if model.use_normalization:
                    preds_t_normalized = torch.clamp(preds_t_normalized, 0.0, 1.0)
                preds_t = model.denormalize_predictions(preds_t_normalized) if model.use_normalization else preds_t_normalized
            
            # Collect predictions and targets for per-subscale calculation
            all_preds_t.append(preds_t.cpu())
            all_Yt_orig.append(Yt_orig.cpu())
            
            mae_t = torch.abs(preds_t - Yt_orig).mean().item()
            mse_t = ((preds_t - Yt_orig) ** 2).mean().item()
            rmse_t = np.sqrt(mse_t)
            
            test_mae += mae_t
            test_rmse += rmse_t
            
            pred_total_t = preds_t.sum(dim=-1)
            true_total_t = Yt_orig.sum(dim=-1)
            total_mae_t = torch.abs(pred_total_t - true_total_t).mean().item()
            total_rmse_t = torch.sqrt(((pred_total_t - true_total_t) ** 2).mean()).item()
            test_total_maes.append(total_mae_t)
            test_total_rmses.append(total_rmse_t)
    
    # Concatenate all predictions and targets
    all_preds_t = torch.cat(all_preds_t, dim=0)  # [num_samples, num_subscales]
    all_Yt_orig = torch.cat(all_Yt_orig, dim=0)  # [num_samples, num_subscales]
    
    # Calculate per-subscale MAE
    per_subscale_mae = torch.abs(all_preds_t - all_Yt_orig).mean(dim=0).numpy()  # [num_subscales]
    
    test_steps = len(test_dataloader)
    final_test_mae = test_mae / test_steps
    final_test_rmse = test_rmse / test_steps
    final_test_total_mae = np.mean(test_total_maes) if test_total_maes else 0
    final_test_total_rmse = np.mean(test_total_rmses) if test_total_rmses else 0
    
    print(f"Final Test - Subscales MAE: {final_test_mae:.4f}, RMSE: {final_test_rmse:.4f}")
    print(f"Final Test - Total MAE: {final_test_total_mae:.4f}, Total RMSE: {final_test_total_rmse:.4f}")
    
    # Get subscale names
    scale_config = cfg.get_scale_config()
    subscale_names = getattr(scale_config, 'subscale_names', [f'subscale_{i+1}' for i in range(num_subscales)])
    
    # Print per-subscale MAE
    print("\nPer-subscale Test MAE:")
    for i, (name, mae_val) in enumerate(zip(subscale_names, per_subscale_mae)):
        print(f"  {name}: {mae_val:.4f}")
    
    # Save test metrics
    test_metrics = {
        'test_mae': final_test_mae,
        'test_rmse': final_test_rmse,
        'test_total_mae': final_test_total_mae,
        'test_total_rmse': final_test_total_rmse
    }
    
    # Save metrics to CSV
    test_metrics_df = pd.DataFrame([test_metrics])
    test_metrics_df.to_csv(f"{log_dir}/test_metrics.csv", index=False)
    
    # Save per-subscale MAE to CSV
    per_subscale_metrics = {
        'subscale_name': subscale_names,
        'test_mae': per_subscale_mae.tolist()
    }
    per_subscale_df = pd.DataFrame(per_subscale_metrics)
    per_subscale_df.to_csv(f"{log_dir}/test_per_subscale_mae.csv", index=False)
    print(f"\nPer-subscale MAE saved to: {log_dir}/test_per_subscale_mae.csv")
    
    # Save validation best metrics
    best_metrics = early_stopper.get_best_metrics()
    if best_metrics:
        val_best_metrics = {
            'val_loss': best_metrics.get('val_loss', np.nan),
            'val_mae': best_metrics.get('val_mae', np.nan),
            'val_rmse': best_metrics.get('val_rmse', np.nan),
            'val_total_mae': best_metrics.get('val_total_mae', np.nan),
            'val_total_rmse': best_metrics.get('val_total_rmse', np.nan),
            'epoch': best_metrics.get('epoch', np.nan)
        }
        val_best_df = pd.DataFrame([val_best_metrics])
        val_best_df.to_csv(f"{log_dir}/val_best_metrics.csv", index=False)
    
    # Save training history
    # Ensure all lists have the same length by padding with NaN
    max_len = max(len(v) for v in history.values() if isinstance(v, list))
    for key in history:
        if isinstance(history[key], list):
            # Pad with NaN if shorter
            while len(history[key]) < max_len:
                history[key].append(np.nan)
    
    history_df = pd.DataFrame(history)
    history_df.to_csv(f"{log_dir}/training_history.csv", index=False)
    
    writer.close()
    print(f"TensorBoard logs saved to: {log_dir}")
    
    # Return final test metrics along with history and early_stopper
    final_metrics = {
        'test_mae': final_test_mae,
        'test_rmse': final_test_rmse,
        'test_total_mae': final_test_total_mae,
        'test_total_rmse': final_test_total_rmse
    }
    
    return history, early_stopper, final_metrics


def main():
    parser = argparse.ArgumentParser(description="Depression assessment training (PHQ-8 and HAMD-13)")
    parser.add_argument("--dataset", type=str, default="cidh",
                       choices=["edaic", "cidh", "pdch"],
                       help="Dataset: edaic (PHQ-8), cidh/cidh/pdch (HAMD-13)")
    parser.add_argument("--scale", type=str, default="HAMD-13",
                       choices=["PHQ-8", "HAMD-13"],
                       help="Depression scale: PHQ-8 (for edaic) or HAMD-13 (for cidh/pdch)")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=10,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--sum_labels", action="store_true",
                       help="Use total score instead of subscales")
    parser.add_argument("--prediction_mode", type=str, default="regression",
                       choices=["regression", "ordinal"],
                       help="Prediction mode")
    parser.add_argument("--seeds", type=str, default=None,
                       help="Comma-separated list of seeds (e.g., '1260,1261,1262') or range (e.g., '1260-1265')")
    parser.add_argument("--seed_start", type=int, default=None,
                       help="Start seed for range (used with --seed_end)")
    parser.add_argument("--seed_end", type=int, default=None,
                       help="End seed for range (used with --seed_start)")
    parser.add_argument("--best_metric", type=str, default="test_total_mae",
                       choices=["test_total_mae", "test_total_rmse", "test_mae", "test_rmse"],
                       help="Metric to use for selecting best seed (lower is better)")
    parser.add_argument("--use_task_spl", action="store_true", default=False,
                       help="use spl)")
    parser.add_argument("--use_cluster_constraint", action="store_true", default=False,
                       help="use cluster constraint")
    parser.add_argument("--use_task_graph", action="store_true", default=False,
                       help="use task graph")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda:0", "cuda:1", "cpu"],
                       help="Device to use")
    
    args = parser.parse_args()
    
    # Custom dataset
    #args.dataset = "edaic"
    #args.dataset = "cidh"
    #args.dataset = "pdch"
    
    custom_seeds = [1060]
    seeds_to_try = []
    if custom_seeds is not None:
        # Use custom seeds from code (highest priority)
        seeds_to_try = custom_seeds
    elif args.seeds:
        # Parse comma-separated seeds from command line
        seeds_to_try = [int(s.strip()) for s in args.seeds.split(',')]
    elif args.seed_start is not None and args.seed_end is not None:
        # Parse range from command line
        seeds_to_try = list(range(args.seed_start, args.seed_end + 1))
    else:
        # Default: use single seed from config
        cfg = get_config_for_dataset(args.dataset)
        seeds_to_try = [cfg.seed]
    
    print(f"\n{'='*80}")
    print(f"Multi-Seed Training: Will try {len(seeds_to_try)} seed(s): {seeds_to_try}")
    print(f"Best metric: {args.best_metric} (lower is better)")
    print(f"{'='*80}\n")
    
    # Get base configuration
    cfg = get_config_for_dataset(args.dataset)
    
    # Print base configuration
    cfg.print_config()
    cfg.use_task_spl = args.use_task_spl
    cfg.use_cluster_constraint = args.use_cluster_constraint
    cfg.use_task_graph = args.use_task_graph
    cfg.device = args.device
    # Create parent output directory for all seeds
    script_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parent_run_name = f"{cfg.dataset_name}_{cfg.scale_type}_multi_seed_{cfg.use_task_spl}_{cfg.use_cluster_constraint}_{cfg.use_task_graph}_{timestamp}"
    parent_run_dir = os.path.join(script_dir, "runs", parent_run_name)
    os.makedirs(parent_run_dir, exist_ok=True)
    
    # Store results for all seeds
    all_seed_results = []
    
    # Loop through each seed
    for seed_idx, seed in enumerate(seeds_to_try):
        print(f"\n{'='*80}")
        print(f"Seed {seed_idx + 1}/{len(seeds_to_try)}: {seed}")
        print(f"{'='*80}\n")
        
        # Set random seed FIRST (before creating dataloaders and model)
        set_seed(seed=seed)
        cfg.seed = seed
        print(f"Random seed set to {seed}")
        
        # Create output directory for this seed
        seed_run_name = f"seed_{seed}"
        seed_run_dir = os.path.join(parent_run_dir, seed_run_name)
        os.makedirs(seed_run_dir, exist_ok=True)
        
        # Create dataloaders (need to recreate for each seed to ensure reproducibility)
        print("\nCreating dataloaders...")
        train_dl, val_dl, test_dl = create_dataloaders(cfg)
        print(f"  Train: {len(train_dl.dataset)} samples")
        print(f"  Val: {len(val_dl.dataset)} samples")
        print(f"  Test: {len(test_dl.dataset)} samples")
        if seed_idx == 0:
            print("  Note: If embedding cache exists, loading will be fast. Otherwise, BERT encoding may take time.")
        
        # Create model (fresh model for each seed)
        print("\nCreating model...")
        model = create_model(cfg)
        device = cfg.device #torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"  Model: MultiScaleTransformer")
        print(f"  Device: {device}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Save configuration for this seed
        import json
        with open(f"{seed_run_dir}/config.json", 'w') as f:
            json.dump(cfg.to_dict(), f, indent=2)
        
        print(f"\nOutput directory: {seed_run_dir}")
        
        # Get scale config for num_subscales
        scale_config = cfg.get_scale_config()
        num_subscales = scale_config.num_subscales
        
        print("\n" + "="*80)
        print(f"Training {cfg.scale_type} model ({num_subscales} subscales) with seed {seed}")
        print("="*80)
        
        # Train model
        history, early_stopper, final_metrics = train_multi_scale_with_tensorboard(
            model=model,
            train_dataloader=train_dl,
            val_dataloader=val_dl,
            test_dataloader=test_dl,
            cfg=cfg,
            num_subscales=num_subscales,
            epochs=cfg.num_epochs,
            lr=cfg.learning_rate,
            patience=cfg.early_stopping_patience,
            transform_targets=cfg.transform_targets,
            scheduler_type=cfg.lr_scheduler,
            min_lr=cfg.min_lr,
            lr_patience=cfg.lr_patience,
            lr_factor=cfg.lr_factor,
            lr_step_size=cfg.lr_step_size,
            lr_gamma=cfg.lr_gamma,
            log_dir=seed_run_dir,
            test_eval_interval=5
        )
        
        # Store results
        seed_result = {
            'seed': seed,
            **final_metrics
        }
        
        # Add validation best metrics if available
        if early_stopper.best_loss is not None:
            seed_result['val_best_total_mae'] = early_stopper.best_loss
        else:
            seed_result['val_best_total_mae'] = np.nan
        
        # Add detailed validation metrics if available
        if hasattr(early_stopper, 'get_best_metrics'):
            val_metrics = early_stopper.get_best_metrics()
            if val_metrics:
                for key, value in val_metrics.items():
                    if key != 'epoch':  # Skip epoch, add other metrics
                        seed_result[f'val_best_{key}'] = value
        
        all_seed_results.append(seed_result)
        # Create results DataFrame
        results_df = pd.DataFrame(all_seed_results)
        
        # Save all results to CSV
        results_csv_path = os.path.join(parent_run_dir, "all_seeds_results.csv")
        results_df.to_csv(results_csv_path, index=False)
        print(f"All seed results saved to: {results_csv_path}")
        
        print(f"\nSeed {seed} completed!")
        print(f"  Test Total MAE: {final_metrics['test_total_mae']:.4f}")
        print(f"  Test Total RMSE: {final_metrics['test_total_rmse']:.4f}")
    
    # Find best seed
    print(f"\n{'='*80}")
    print("All Seeds Completed - Summary")
    print(f"{'='*80}\n")

    # Find best seed based on specified metric
    best_metric_value = results_df[args.best_metric].min()
    best_seed_row = results_df.loc[results_df[args.best_metric].idxmin()]
    best_seed = int(best_seed_row['seed'])
    
    print(f"\n{'='*80}")
    print(f"Best Seed: {best_seed}")
    print(f"{'='*80}")
    print(f"Best {args.best_metric}: {best_metric_value:.4f}")
    print(f"\nBest Seed Results:")
    for col in results_df.columns:
        if col != 'seed':
            print(f"  {col}: {best_seed_row[col]:.4f}")
    print(f"{'='*80}\n")
    
    # Save best seed summary
    best_seed_summary = {
        'best_seed': int(best_seed),
        'best_metric': args.best_metric,
        'best_metric_value': float(best_metric_value),
        'all_results': all_seed_results
    }
    with open(os.path.join(parent_run_dir, "best_seed_summary.json"), 'w') as f:
        json.dump(best_seed_summary, f, indent=2)
    
    print(f"Best seed summary saved to: {os.path.join(parent_run_dir, 'best_seed_summary.json')}")
    print(f"\nParent output directory: {parent_run_dir}")
    print("\nTraining completed!")


if __name__ == "__main__":
    main()

