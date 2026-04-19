"""
Training script for total score prediction only (no subscale supervision).
This script trains models to predict only the total depression score,
without subscale-level supervision.

Usage:
    # Train on PHQ-8 (E-DAIC) - total score only
    python train_transformer_totalscore.py --dataset edaic --scale PHQ-8
    
    # Train on HAMD-13 (CIDH) - total score only
    python train_transformer_totalscore.py --dataset cidh --scale HAMD-13
    
    # Train on HAMD-13 (PDCH) - total score only
    python train_transformer_totalscore.py --dataset pdch --scale HAMD-13
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.multi_scale_config import create_config_for_dataset
from models.multi_scale_transformer import MultiScaleTransformer
from preprocessing.HAMD13Dataset import get_hamd13_dataloader
from preprocessing.EDAICDataset import get_edaic_dataloader
from utils.early_stopping import EarlyStopping
from utils.utils import set_seed
from transformers import AutoModel


def create_dataloaders(cfg):
    """Create dataloaders for datasets (E-DAIC or HAMD-13)."""
    # Force total score prediction (sum_labels=True)
    cfg.sum_labels = True
    
    # Load BERT model for encoding
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    bert_model = AutoModel.from_pretrained(cfg.model_name)
    bert_model.eval()
    bert_model = bert_model.to(device)
    
    # Select appropriate dataloader function based on dataset
    if cfg.dataset_name.lower() == "edaic":
        # E-DAIC dataset (PHQ-8)
        train_dl = get_edaic_dataloader(
            split="train",
            data_dir=cfg.dataset_dir,
            label_file=cfg.label_file,
            sum_labels=True,  # Always use total score
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
            sum_labels=True,
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
            sum_labels=True,
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
            sum_labels=True,  # Always use total score
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
            sum_labels=True,
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
            sum_labels=True,
            bert_model_name=cfg.model_name,
            batch_size=cfg.batch_size,
            shuffle=False,
            model=bert_model,
            device=device,
            num_workers=0,
            pin_memory=False
        )
    
    return train_dl, val_dl, test_dl


def create_model(cfg):
    """Create model for total score prediction."""
    scale_config = cfg.get_scale_config()
    
    # For total score prediction, we still use MultiScaleTransformer
    # but configure it to output a single value (total score)
    # We'll use num_subscales=1 and adjust the output accordingly
    model = MultiScaleTransformer(
        input_dim=768,  # BERT embedding size
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        num_heads=4,
        dropout=cfg.dropout,
        num_subscales=1,  # Only one output: total score
        subscale_ranges=[scale_config.total_range],  # Max total score
        prediction_mode=cfg.prediction_mode,
        use_normalization=cfg.use_normalization,
        per_subscale_normalization=False,  # Single output, no per-subscale normalization
        normalization_min=0.0,
        normalization_max=float(scale_config.total_range),
        subscale_min_list=None,
        subscale_max_list=None
    )
    
    return model


def train_totalscore_with_tensorboard(
    model, train_dataloader, val_dataloader, test_dataloader,
    cfg,
    epochs=100, lr=1e-3, patience=15,
    scheduler_type="cosine", min_lr=1e-5, lr_patience=5,
    lr_factor=0.5, lr_step_size=15, lr_gamma=0.1,
    lr_warmup_epochs=5, lr_pct_start=0.3,
    log_dir=None,
    test_eval_interval=5
):
    """
    Train total score prediction model with TensorBoard logging.
    This is a baseline approach that predicts only the total score.
    """
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Create TensorBoard writer
    if log_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"runs/totalscore_{cfg.dataset_name}_{cfg.scale_type}_{timestamp}"
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")
    print(f"Training mode: Total Score Prediction Only")
    print(f"Prediction mode: {model.prediction_mode}")
    print(f"Use normalization: {model.use_normalization}")
    
    # Setup optimizer and early stopping
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    early_stopper = EarlyStopping(patience=patience)
    
    # Setup loss function (only MSE for regression, no ordinal for total score)
    if model.prediction_mode == "regression":
        criterion = nn.MSELoss()
    else:
        raise ValueError("Total score prediction only supports regression mode")
    
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
    
    # History dictionary
    history = {
        k: [] for k in [
            "train_loss", "train_mae", "train_rmse",
            "val_loss", "val_mae", "val_rmse",
            "learning_rate",
            "test_loss", "test_mae", "test_rmse"
        ]
    }
    
    # Training loop
    for epoch in tqdm(range(epochs), desc="Epochs"):
        # ==================== Training Phase ====================
        model.train()
        train_loss = train_mae = train_rmse = 0.0
        
        for batch_sample_ids, X, Y, attention_mask, raw_utterances in train_dataloader:
            X, Y, attention_mask = X.to(device), Y.to(device), attention_mask.to(device)
            
            # Y shape: [batch, 1] for total score
            if Y.dim() > 1 and Y.shape[1] > 1:
                # If we got subscales, sum them to get total score
                Y = Y.sum(dim=1, keepdim=True)
            elif Y.dim() == 1:
                Y = Y.unsqueeze(1)
            
            Y_orig = Y.clone()
            
            # Normalize labels
            if model.use_normalization:
                Y_normalized = model.normalize_labels(Y_orig)
            else:
                Y_normalized = Y_orig
            
            # Forward pass
            optimizer.zero_grad()
            output = model(X, attention_mask=attention_mask)
            
            # Output shape: [batch, 1] for total score
            if output.dim() > 1 and output.shape[1] > 1:
                # If model outputs multiple values, take the first one
                output = output[:, 0:1]
            
            # Denormalize predictions
            if model.use_normalization:
                preds_normalized = torch.clamp(output, 0.0, 1.0)
                preds_denorm = model.denormalize_predictions(preds_normalized)
            else:
                preds_denorm = output
            
            # Calculate loss
            loss = criterion(output, Y_normalized)
            
            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                train_loss += loss.item()
                mae = torch.abs(preds_denorm - Y_orig).mean().item()
                rmse = torch.sqrt(torch.mean((preds_denorm - Y_orig) ** 2)).item()
                train_mae += mae
                train_rmse += rmse
        
        # Average training metrics
        num_batches = len(train_dataloader)
        train_loss /= num_batches
        train_mae /= num_batches
        train_rmse /= num_batches
        
        # ==================== Validation Phase ====================
        model.eval()
        val_loss = val_mae = val_rmse = 0.0
        
        with torch.no_grad():
            for batch_sample_ids, X, Y, attention_mask, raw_utterances in val_dataloader:
                X, Y, attention_mask = X.to(device), Y.to(device), attention_mask.to(device)
                
                if Y.dim() > 1 and Y.shape[1] > 1:
                    Y = Y.sum(dim=1, keepdim=True)
                elif Y.dim() == 1:
                    Y = Y.unsqueeze(1)
                
                Y_orig = Y.clone()
                
                if model.use_normalization:
                    Y_normalized = model.normalize_labels(Y_orig)
                else:
                    Y_normalized = Y_orig
                
                output = model(X, attention_mask=attention_mask)
                
                if output.dim() > 1 and output.shape[1] > 1:
                    output = output[:, 0:1]
                
                if model.use_normalization:
                    preds_normalized = torch.clamp(output, 0.0, 1.0)
                    preds_denorm = model.denormalize_predictions(preds_normalized)
                else:
                    preds_denorm = output
                
                loss = criterion(output, Y_normalized)
                
                val_loss += loss.item()
                mae = torch.abs(preds_denorm - Y_orig).mean().item()
                rmse = torch.sqrt(torch.mean((preds_denorm - Y_orig) ** 2)).item()
                val_mae += mae
                val_rmse += rmse
        
        num_batches = len(val_dataloader)
        val_loss /= num_batches
        val_mae /= num_batches
        val_rmse /= num_batches
        
        # Update learning rate
        if scheduler is not None:
            if scheduler_type == "plateau":
                scheduler.step(val_loss)
            elif scheduler_type == "one_cycle":
                pass  # Updated per step
            else:
                scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Early stopping
        if early_stopper(val_loss, model):
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('MAE/train', train_mae, epoch)
        writer.add_scalar('MAE/val', val_mae, epoch)
        writer.add_scalar('RMSE/train', train_rmse, epoch)
        writer.add_scalar('RMSE/val', val_rmse, epoch)
        writer.add_scalar('LearningRate', current_lr, epoch)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_mae'].append(train_mae)
        history['train_rmse'].append(train_rmse)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['val_rmse'].append(val_rmse)
        history['learning_rate'].append(current_lr)
        
        # Print progress
        print(f"\nEp {epoch+1}/{epochs} | LR {current_lr:.6f} | "
              f"Train MAE {train_mae:.4f} RMSE {train_rmse:.4f} | "
              f"Val MAE {val_mae:.4f} RMSE {val_rmse:.4f}")
        
        # Test evaluation (periodic)
        if (epoch + 1) % test_eval_interval == 0 or (epoch + 1) == epochs:
            model.eval()
            test_loss = test_mae = test_rmse = 0.0
            
            with torch.no_grad():
                for batch_sample_ids, X, Y, attention_mask, raw_utterances in test_dataloader:
                    X, Y, attention_mask = X.to(device), Y.to(device), attention_mask.to(device)
                    
                    if Y.dim() > 1 and Y.shape[1] > 1:
                        Y = Y.sum(dim=1, keepdim=True)
                    elif Y.dim() == 1:
                        Y = Y.unsqueeze(1)
                    
                    Y_orig = Y.clone()
                    
                    if model.use_normalization:
                        Y_normalized = model.normalize_labels(Y_orig)
                    else:
                        Y_normalized = Y_orig
                    
                    output = model(X, attention_mask=attention_mask)
                    
                    if output.dim() > 1 and output.shape[1] > 1:
                        output = output[:, 0:1]
                    
                    if model.use_normalization:
                        preds_normalized = torch.clamp(output, 0.0, 1.0)
                        preds_denorm = model.denormalize_predictions(preds_normalized)
                    else:
                        preds_denorm = output
                    
                    loss = criterion(output, Y_normalized)
                    
                    test_loss += loss.item()
                    mae = torch.abs(preds_denorm - Y_orig).mean().item()
                    rmse = torch.sqrt(torch.mean((preds_denorm - Y_orig) ** 2)).item()
                    test_mae += mae
                    test_rmse += rmse
            
            num_batches = len(test_dataloader)
            test_loss /= num_batches
            test_mae /= num_batches
            test_rmse /= num_batches
            
            writer.add_scalar('Loss/test', test_loss, epoch)
            writer.add_scalar('MAE/test', test_mae, epoch)
            writer.add_scalar('RMSE/test', test_rmse, epoch)
            
            history['test_loss'].append(test_loss)
            history['test_mae'].append(test_mae)
            history['test_rmse'].append(test_rmse)
            
            print(f"Test - MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")
    
    # Final test evaluation
    model.eval()
    final_test_mae = final_test_rmse = 0.0
    
    with torch.no_grad():
        for batch_sample_ids, X, Y, attention_mask, raw_utterances in test_dataloader:
            X, Y, attention_mask = X.to(device), Y.to(device), attention_mask.to(device)
            
            if Y.dim() > 1 and Y.shape[1] > 1:
                Y = Y.sum(dim=1, keepdim=True)
            elif Y.dim() == 1:
                Y = Y.unsqueeze(1)
            
            Y_orig = Y.clone()
            
            if model.use_normalization:
                Y_normalized = model.normalize_labels(Y_orig)
            else:
                Y_normalized = Y_orig
            
            output = model(X, attention_mask=attention_mask)
            
            if output.dim() > 1 and output.shape[1] > 1:
                output = output[:, 0:1]
            
            if model.use_normalization:
                preds_normalized = torch.clamp(output, 0.0, 1.0)
                preds_denorm = model.denormalize_predictions(preds_normalized)
            else:
                preds_denorm = output
            
            final_test_mae += torch.abs(preds_denorm - Y_orig).mean().item()
            final_test_rmse += torch.sqrt(torch.mean((preds_denorm - Y_orig) ** 2)).item()
    
    num_batches = len(test_dataloader)
    final_test_mae /= num_batches
    final_test_rmse /= num_batches
    
    print(f"\nFinal Test - MAE: {final_test_mae:.4f}, RMSE: {final_test_rmse:.4f}")
    
    # Save history - handle different lengths for test metrics
    # Test metrics are only recorded at intervals, so pad with None/NaN
    max_len = max(len(v) for v in history.values())
    for key in history:
        while len(history[key]) < max_len:
            history[key].append(None)
    
    history_df = pd.DataFrame(history)
    history_df.to_csv(f"{log_dir}/training_history.csv", index=False)
    
    writer.close()
    print(f"TensorBoard logs saved to: {log_dir}")
    
    return final_test_mae, final_test_rmse, history, early_stopper


def main():
    parser = argparse.ArgumentParser(description="Total score prediction training (baseline)")
    parser.add_argument("--dataset", type=str, default="cidh",
                       choices=["edaic", "cidh", "pdch"],
                       help="Dataset name")
    parser.add_argument("--scale", type=str, default="HAMD-13",
                       choices=["PHQ-8", "HAMD-13"],
                       help="Depression scale")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=10,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--seeds", type=str, default=None,
                       help="Comma-separated list of seeds (e.g., '800,801,802')")
    parser.add_argument("--seed_start", type=int, default=None,
                       help="Start of seed range")
    parser.add_argument("--seed_end", type=int, default=None,
                       help="End of seed range (inclusive)")
    
    args = parser.parse_args()
    
    # Custom parameters
    # args.dataset = "edaic"
    # args.dataset = "cidh"
    args.dataset = "pdch"

    # If set to None, use command line arguments or seed from config file
    custom_seeds = None  # Example: [800, 801, 802, 803, 804]
    
    # If custom_seeds is None and command line arguments are also None, use default value
    # if custom_seeds is None and args.seed_start is None and args.seed_end is None:
    #     args.seed_start = 1000
    #     args.seed_end = 1100
    
    # Determine seed list
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
        cfg = create_config_for_dataset(args.dataset)
        seeds_to_try = [cfg.seed]
    
    print(f"\n{'='*80}")
    print(f"Multi-Seed Training: Will try {len(seeds_to_try)} seed(s)")
    print(f"Seed range: {seeds_to_try[0]} to {seeds_to_try[-1]}")
    print(f"{'='*80}\n")
    
    # Get base configuration
    # Get configuration
    cfg = create_config_for_dataset(args.dataset)
    # Force total score prediction
    cfg.sum_labels = True
    cfg.prediction_mode = "regression"  # Total score only supports regression
    
    # Disable cluster constraint and task-level SPL for total score prediction
    # These are only meaningful for subscale supervision
    cfg.use_cluster_constraint = False
    cfg.use_task_spl = False
    cfg.use_normalization = False
    
    # Print configuration
    cfg.print_config()
    
    # Create main output directory for all seeds
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_run_name = f"totalscore_{cfg.dataset_name}_{cfg.scale_type}_multi_seed_{timestamp}"
    main_run_dir = f"runs/{main_run_name}"
    os.makedirs(main_run_dir, exist_ok=True)
    
    # Save main configuration
    import json
    config_dict = cfg.to_dict()
    config_dict['seeds'] = seeds_to_try
    with open(f"{main_run_dir}/config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # Store results for all seeds
    all_seed_results = []
    
    # Loop through each seed
    for seed_idx, seed in enumerate(seeds_to_try):
        print(f"\n{'='*80}")
        print(f"Training with seed {seed} ({seed_idx + 1}/{len(seeds_to_try)})")
        print(f"{'='*80}")
        
        # Set random seed FIRST (before dataloader and model creation)
        set_seed(seed)
        print(f"Random seed set to {seed}")
        
        # Create dataloaders
        print("\nCreating dataloaders...")
        print("  Loading BERT model and tokenizer (this may take a moment)...")
        train_dl, val_dl, test_dl = create_dataloaders(cfg)
        print(f"  Train: {len(train_dl.dataset)} samples")
        print(f"  Val: {len(val_dl.dataset)} samples")
        print(f"  Test: {len(test_dl.dataset)} samples")
        
        # Create model
        print("\nCreating model...")
        model = create_model(cfg)
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"  Model: MultiScaleTransformer (Total Score Only)")
        print(f"  Device: {device}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        print(f"\nTraining {cfg.scale_type} model (Total Score Prediction)")
        print(f"Mode: Total Score Only (Baseline)")
        
        # Create output directory for this seed
        seed_run_dir = os.path.join(main_run_dir, f"seed_{seed}")
        os.makedirs(seed_run_dir, exist_ok=True)
        
        # Save seed-specific configuration
        seed_config_dict = cfg.to_dict()
        seed_config_dict['seed'] = seed
        with open(f"{seed_run_dir}/config.json", 'w') as f:
            json.dump(seed_config_dict, f, indent=2)
        
        # Train
        final_test_mae, final_test_rmse, history, early_stopper = train_totalscore_with_tensorboard(
            model, train_dl, val_dl, test_dl,
            cfg,
            epochs=cfg.num_epochs,
            lr=cfg.learning_rate,
            patience=cfg.early_stopping_patience,
            log_dir=seed_run_dir
        )
        
        # Store result for this seed
        seed_result = {
            'seed': seed,
            'final_test_mae': final_test_mae,
            'final_test_rmse': final_test_rmse,
        }
        all_seed_results.append(seed_result)
        
        print(f"\nSeed {seed} completed!")
        print(f"  Test Total MAE: {final_test_mae:.4f}")
        print(f"  Test Total RMSE: {final_test_rmse:.4f}")
        print(f"  Results saved to: {seed_run_dir}")
        
        # Save intermediate results (update after each seed)
        results_df = pd.DataFrame(all_seed_results)
        results_csv_path = os.path.join(main_run_dir, "all_seeds_results.csv")
        results_df.to_csv(results_csv_path, index=False)
        print(f"  All seeds results updated: {results_csv_path}")
    
    # Final summary after all seeds
    print(f"\n{'='*80}")
    print(f"All seeds training completed!")
    print(f"{'='*80}")
    print(f"\nSummary of all {len(seeds_to_try)} seeds:")
    print(f"{'Seed':<10} {'MAE':<12} {'RMSE':<12}")
    print(f"{'-'*34}")
    for result in all_seed_results:
        print(f"{result['seed']:<10} {result['final_test_mae']:<12.4f} {result['final_test_rmse']:<12.4f}")
    
    # Calculate statistics
    mae_values = [r['final_test_mae'] for r in all_seed_results]
    rmse_values = [r['final_test_rmse'] for r in all_seed_results]
    
    print(f"\nStatistics across all seeds:")
    print(f"  MAE - Mean: {np.mean(mae_values):.4f}, Std: {np.std(mae_values):.4f}, Min: {np.min(mae_values):.4f}, Max: {np.max(mae_values):.4f}")
    print(f"  RMSE - Mean: {np.mean(rmse_values):.4f}, Std: {np.std(rmse_values):.4f}, Min: {np.min(rmse_values):.4f}, Max: {np.max(rmse_values):.4f}")
    
    # Save final results
    results_df = pd.DataFrame(all_seed_results)
    results_csv_path = os.path.join(main_run_dir, "all_seeds_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nAll seed results saved to: {results_csv_path}")
    print(f"Main run directory: {main_run_dir}")


if __name__ == "__main__":
    main()

