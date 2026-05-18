"""Utility functions for exploratory data analysis of mel spectrograms."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def load_mel_metadata(metadata_csv: Path) -> List[Dict[str, Any]]:
    """Load mel spectrogram metadata from CSV."""
    examples: List[Dict[str, Any]] = []
    with metadata_csv.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            examples.append({
                "mel_path": row["mel_path"],
                "duration": float(row["duration"]),
                "text": row.get("text", ""),
            })
    return examples


def load_mel_tensors(manifest: List[Dict[str, Any]], max_samples: Optional[int] = None) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Load and aggregate mel tensors from manifest.
    
    Returns:
        (features, metadata) where features has shape (n_samples, n_mels * n_frames_mean)
        aggregating time dimension by taking mean across time.
    """
    features = []
    valid_manifest = []
    
    for i, row in enumerate(manifest):
        if max_samples is not None and i >= max_samples:
            break
        try:
            data = torch.load(row["mel_path"])
            mel = data["mel"].numpy()  # shape (n_mels, T)
            # aggregate time: mean across frames
            mel_mean = mel.mean(axis=1)  # shape (n_mels,)
            features.append(mel_mean)
            valid_manifest.append(row)
        except Exception as e:
            print(f"Error loading {row['mel_path']}: {e}")
            continue
    
    return np.array(features, dtype=np.float32), valid_manifest


def analyze_mel_dimensions(features: np.ndarray, names: Optional[List[str]] = None) -> Dict[str, Any]:
    """Compute dimension statistics (mean, std, min, max) for mel spectrograms.
    
    Args:
        features: shape (n_samples, n_mels)
        names: optional list of dimension names (e.g., ['mel_0', 'mel_1', ...])
    
    Returns:
        Dictionary with per-dimension statistics.
    """
    if names is None:
        names = [f"mel_{i}" for i in range(features.shape[1])]
    
    stats = {
        "names": names,
        "mean": features.mean(axis=0),
        "std": features.std(axis=0),
        "min": features.min(axis=0),
        "max": features.max(axis=0),
        "median": np.median(features, axis=0),
        "q25": np.percentile(features, 25, axis=0),
        "q75": np.percentile(features, 75, axis=0),
    }
    return stats


def plot_dimension_histograms(features: np.ndarray, names: Optional[List[str]] = None, 
                               bins: int = 30, figsize: Tuple[int, int] = (16, 12),
                               save_path: Optional[Path] = None) -> None:
    """Plot histograms of variation in each mel dimension.
    
    Args:
        features: shape (n_samples, n_mels)
        names: optional dimension names
        bins: number of histogram bins
        figsize: figure size
        save_path: optional path to save figure
    """
    if names is None:
        names = [f"mel_{i}" for i in range(features.shape[1])]
    
    n_dims = features.shape[1]
    n_cols = 4
    n_rows = (n_dims + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for i, (ax, name) in enumerate(zip(axes, names)):
        ax.hist(features[:, i], bins=bins, alpha=0.7, edgecolor='black')
        ax.set_title(f"{name} (μ={features[:, i].mean():.2f}, σ={features[:, i].std():.2f})")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.grid(alpha=0.3)
    
    # hide unused subplots
    for ax in axes[n_dims:]:
        ax.set_visible(False)
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()


def compute_tsne(features: np.ndarray, n_components: int = 2, perplexity: int = 30,
                  random_state: int = 42) -> np.ndarray:
    """Compute t-SNE embedding of mel features.
    
    Args:
        features: shape (n_samples, n_mels)
        n_components: 2 or 3 for visualization
        perplexity: t-SNE perplexity parameter
        random_state: for reproducibility
    
    Returns:
        Embedded features with shape (n_samples, n_components)
    """
    # standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    print(f"Computing t-SNE ({features.shape[0]} samples, {features.shape[1]} dims)...")
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state, n_jobs=-1)
    embedding = tsne.fit_transform(features_scaled)
    print(f"t-SNE complete. Embedding shape: {embedding.shape}")
    
    return embedding


def plot_tsne_2d(embedding: np.ndarray, metadata: List[Dict[str, Any]], 
                 color_by: Optional[str] = None, figsize: Tuple[int, int] = (10, 8),
                 save_path: Optional[Path] = None) -> None:
    """Plot 2D t-SNE embedding.
    
    Args:
        embedding: shape (n_samples, 2) from compute_tsne
        metadata: list of metadata dicts
        color_by: 'duration' to color by duration, None for uniform color
        figsize: figure size
        save_path: optional path to save
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if color_by == "duration" and metadata:
        durations = np.array([m.get("duration", 0.0) for m in metadata])
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=durations, cmap='viridis', 
                            s=30, alpha=0.6, edgecolors='black', linewidth=0.5)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Duration (s)")
    else:
        ax.scatter(embedding[:, 0], embedding[:, 1], s=30, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("t-SNE Embedding of Mel Spectrograms")
    ax.grid(alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()


def plot_tsne_3d(embedding: np.ndarray, metadata: List[Dict[str, Any]],
                 color_by: Optional[str] = None, figsize: Tuple[int, int] = (12, 9),
                 save_path: Optional[Path] = None) -> None:
    """Plot 3D t-SNE embedding.
    
    Args:
        embedding: shape (n_samples, 3) from compute_tsne
        metadata: list of metadata dicts
        color_by: 'duration' to color by duration
        figsize: figure size
        save_path: optional path to save
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    if color_by == "duration" and metadata:
        durations = np.array([m.get("duration", 0.0) for m in metadata])
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                            c=durations, cmap='viridis', s=30, alpha=0.6, edgecolors='black', linewidth=0.5)
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5)
        cbar.set_label("Duration (s)")
    else:
        ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                  s=30, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_zlabel("t-SNE 3")
    ax.set_title("3D t-SNE Embedding of Mel Spectrograms")
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()


def plot_duration_distribution(metadata: List[Dict[str, Any]], bins: int = 30,
                               figsize: Tuple[int, int] = (10, 6),
                               save_path: Optional[Path] = None) -> None:
    """Plot histogram of audio durations."""
    durations = np.array([m.get("duration", 0.0) for m in metadata])
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(durations, bins=bins, alpha=0.7, edgecolor='black')
    ax.axvline(durations.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {durations.mean():.2f}s')
    ax.axvline(np.median(durations), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(durations):.2f}s')
    ax.set_xlabel("Duration (seconds)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Audio Durations")
    ax.legend()
    ax.grid(alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()


def compute_dimension_correlations(features: np.ndarray, names: Optional[List[str]] = None) -> np.ndarray:
    """Compute correlation matrix between mel dimensions.
    
    Args:
        features: shape (n_samples, n_mels)
        names: optional dimension names
    
    Returns:
        Correlation matrix of shape (n_mels, n_mels)
    """
    corr = np.corrcoef(features.T)
    return corr


def plot_correlation_matrix(corr_matrix: np.ndarray, names: Optional[List[str]] = None,
                            figsize: Tuple[int, int] = (12, 10),
                            save_path: Optional[Path] = None) -> None:
    """Plot correlation matrix heatmap."""
    if names is None:
        names = [f"mel_{i}" for i in range(corr_matrix.shape[0])]
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_yticklabels(names)
    ax.set_title("Correlation Matrix of Mel Dimensions")
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Correlation")
    
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()
