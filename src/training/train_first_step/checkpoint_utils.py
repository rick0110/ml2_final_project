#!/usr/bin/env python3
"""Checkpoint inspection and analysis utilities.

This script provides utilities to:
- Inspect checkpoint contents
- Compare checkpoints
- Extract metrics from training runs
- List available checkpoints
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List
from datetime import datetime

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def list_experiments():
    """List all available experiments."""
    experiments_root = PROJECT_ROOT / "experiments" / "step_1"
    
    if not experiments_root.exists():
        print("No experiments found.")
        return
    
    print("\nAvailable Experiments:")
    print("=" * 80)
    
    for exp_dir in sorted(experiments_root.iterdir()):
        if not exp_dir.is_dir():
            continue
        
        checkpoint_dir = exp_dir / "checkpoints"
        config_file = exp_dir / "config.json"
        
        num_checkpoints = len(list(checkpoint_dir.glob("*.pt"))) if checkpoint_dir.exists() else 0
        
        has_best = (checkpoint_dir / "best.pt").exists() if checkpoint_dir.exists() else False
        best_marker = " [BEST]" if has_best else ""
        
        print(f"\n{exp_dir.name}{best_marker}")
        print(f"  Path: {exp_dir}")
        print(f"  Checkpoints: {num_checkpoints}")
        
        if config_file.exists():
            import json
            with open(config_file) as f:
                config = json.load(f)
            print(f"  Epochs: {config.get('num_epochs', 'N/A')}")
            print(f"  Batch Size: {config.get('batch_size', 'N/A')}")


def list_checkpoints(experiment_name: str):
    """List checkpoints in an experiment."""
    exp_dir = PROJECT_ROOT / "experiments" / "step_1" / experiment_name
    
    if not exp_dir.exists():
        print(f"Experiment '{experiment_name}' not found.")
        return
    
    checkpoint_dir = exp_dir / "checkpoints"
    
    if not checkpoint_dir.exists():
        print(f"No checkpoints directory in {exp_dir}")
        return
    
    print(f"\nCheckpoints in: {experiment_name}")
    print("=" * 80)
    
    checkpoints = sorted(checkpoint_dir.glob("*.pt"))
    
    if not checkpoints:
        print("No checkpoints found.")
        return
    
    print(f"{'Filename':<30} {'Size':<15} {'Modified':<20}")
    print("-" * 80)
    
    for ckpt_path in checkpoints:
        size_mb = ckpt_path.stat().st_size / (1024 ** 2)
        modified = datetime.fromtimestamp(ckpt_path.stat().st_mtime)
        modified_str = modified.strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"{ckpt_path.name:<30} {size_mb:>10.2f} MB   {modified_str}")


def inspect_checkpoint(checkpoint_path: Path):
    """Inspect contents of a checkpoint."""
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"\nInspecting: {checkpoint_path}")
    print("=" * 80)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
    
    # Print basic info
    if "epoch" in checkpoint:
        print(f"Epoch: {checkpoint['epoch']}")
    
    # Print model state dict info
    if "model_state_dict" in checkpoint:
        print(f"\nModel Parameters:")
        state_dict = checkpoint["model_state_dict"]
        total_params = sum(p.numel() for p in state_dict.values())
        print(f"  Total: {total_params:,}")
        
        # Group by module
        modules = {}
        for param_name in state_dict.keys():
            module = param_name.split(".")[0] if "." in param_name else "root"
            if module not in modules:
                modules[module] = 0
            modules[module] += state_dict[param_name].numel()
        
        print(f"\n  By Module:")
        for module, count in sorted(modules.items(), key=lambda x: -x[1]):
            print(f"    {module:<30} {count:>12,}")
    
    # Print optimizer state
    if "optimizer_state_dict" in checkpoint:
        print(f"\nOptimizer:")
        opt_state = checkpoint["optimizer_state_dict"]
        print(f"  State steps: {len(opt_state.get('state', {}))}")
        if "param_groups" in opt_state:
            print(f"  Learning rate: {opt_state['param_groups'][0].get('lr', 'N/A')}")
    
    # Print metrics
    if "metrics" in checkpoint:
        print(f"\nMetrics:")
        metrics = checkpoint["metrics"]
        for key, value in sorted(metrics.items()):
            if isinstance(value, float):
                print(f"  {key:<30} {value:.6f}")
            else:
                print(f"  {key:<30} {value}")


def compare_checkpoints(paths: List[Path]):
    """Compare multiple checkpoints."""
    print(f"\nComparing {len(paths)} checkpoints:")
    print("=" * 100)
    
    data = []
    
    for ckpt_path in paths:
        if not ckpt_path.exists():
            print(f"  Skipped (not found): {ckpt_path}")
            continue
        
        try:
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            epoch = checkpoint.get("epoch", "N/A")
            metrics = checkpoint.get("metrics", {})
            
            data.append({
                "path": ckpt_path.name,
                "epoch": epoch,
                "loss": metrics.get("loss", float("nan")),
                "recon_loss": metrics.get("recon_loss", float("nan")),
                "div_loss": metrics.get("div_loss", float("nan")),
                "val_loss": metrics.get("val_loss", float("nan")),
            })
        except Exception as e:
            print(f"  Error loading {ckpt_path}: {e}")
    
    if not data:
        print("No valid checkpoints to compare.")
        return
    
    print(f"\n{'Checkpoint':<25} {'Epoch':<8} {'Train Loss':<15} {'Val Loss':<15} {'Recon':<12} {'Div':<12}")
    print("-" * 100)
    
    for d in sorted(data, key=lambda x: x["epoch"] if isinstance(x["epoch"], int) else 0):
        print(
            f"{d['path']:<25} "
            f"{str(d['epoch']):<8} "
            f"{d['loss']:<15.6f} "
            f"{d['val_loss']:<15.6f} "
            f"{d['recon_loss']:<12.6f} "
            f"{d['div_loss']:<12.6f}"
        )


def extract_metrics(experiment_name: str, metric: str = "loss"):
    """Extract specific metric from all checkpoints in an experiment."""
    exp_dir = PROJECT_ROOT / "experiments" / "step_1" / experiment_name
    checkpoint_dir = exp_dir / "checkpoints"
    
    if not checkpoint_dir.exists():
        print(f"Experiment '{experiment_name}' not found.")
        return
    
    print(f"\nExtracting '{metric}' from: {experiment_name}")
    print("=" * 60)
    
    data = []
    
    for ckpt_path in sorted(checkpoint_dir.glob("*.pt")):
        try:
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            epoch = checkpoint.get("epoch", 0)
            metrics = checkpoint.get("metrics", {})
            
            value = metrics.get(metric)
            if value is not None:
                data.append((epoch, value))
        except Exception as e:
            print(f"Error loading {ckpt_path}: {e}")
    
    if data:
        print(f"\n{'Epoch':<10} {'Value':<15}")
        print("-" * 60)
        for epoch, value in sorted(data):
            print(f"{epoch:<10} {value:<15.6f}")
        
        # Find best
        best_epoch, best_value = min(data, key=lambda x: x[1]) if "loss" in metric else max(data, key=lambda x: x[1])
        print(f"\nBest: Epoch {best_epoch} with {metric}={best_value:.6f}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Inspect and analyze training checkpoints",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # List experiments
    subparsers.add_parser("list-experiments", help="List all experiments")
    
    # List checkpoints
    list_ckpt_parser = subparsers.add_parser("list", help="List checkpoints in experiment")
    list_ckpt_parser.add_argument("experiment", help="Experiment name")
    
    # Inspect checkpoint
    inspect_parser = subparsers.add_parser("inspect", help="Inspect a checkpoint")
    inspect_parser.add_argument("checkpoint", help="Path to checkpoint file")
    
    # Compare checkpoints
    compare_parser = subparsers.add_parser("compare", help="Compare multiple checkpoints")
    compare_parser.add_argument("checkpoints", nargs="+", help="Paths to checkpoint files")
    
    # Extract metrics
    metrics_parser = subparsers.add_parser("metrics", help="Extract metrics from experiment")
    metrics_parser.add_argument("experiment", help="Experiment name")
    metrics_parser.add_argument("--metric", default="loss", help="Metric to extract")
    
    args = parser.parse_args()
    
    if args.command == "list-experiments":
        list_experiments()
    elif args.command == "list":
        list_checkpoints(args.experiment)
    elif args.command == "inspect":
        inspect_checkpoint(Path(args.checkpoint))
    elif args.command == "compare":
        compare_checkpoints([Path(p) for p in args.checkpoints])
    elif args.command == "metrics":
        extract_metrics(args.experiment, args.metric)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
