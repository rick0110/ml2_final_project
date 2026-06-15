"""
Utility functions for exploratory data analysis of mel spectrograms.

Responsibilities:
    - Load mel-spectrogram metadata and tensors.
    - Perform statistical analysis on mel frequency dimensions.
    - Visualize mel distributions through histograms and heatmaps.
    - Compute and plot dimensional reductions (t-SNE) for latent exploration.

Main Functions:
    - analyze_mel_dimensions: Compute basic stats per frequency bin.
    - compute_tsne: Generate 2D/3D embeddings for visualization.
    - plot_dimension_histograms: Visualize variation across channels.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def load_mel_metadata(metadata_csv: Path) -> List[Dict[str, Any]]:
    """
    Load mel spectrogram metadata from a CSV manifest.

    Args:
        metadata_csv (Path): Path to the manifest file.

    Returns:
        List[Dict[str, Any]]: List of metadata dictionaries.
    """
    examples: List[Dict[str, Any]] = []
    with metadata_csv.open("r", encoding="utf-8") as fh:
        reader: csv.DictReader = csv.DictReader(fh)
        for row in reader:
            examples.append({
                "mel_path": row["mel_path"],
                "duration": float(row["duration"]),
                "text": row.get("text", ""),
            })
    return examples


def load_mel_tensors(manifest: List[Dict[str, Any]], max_samples: Optional[int] = None) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Load and aggregate mel tensors from manifest.
    
    Aggregates the time dimension by taking the mean across frames.

    Args:
        manifest (List[Dict[str, Any]]): List of examples.
        max_samples (Optional[int]): Cap on samples to load.

    Returns:
        Tuple[np.ndarray, List[Dict[str, Any]]]: 
            - features: shape (N, n_mels)
            - valid_manifest: matched metadata.
    """
    if not manifest:
        return np.empty((0, 80), dtype=np.float32), []

    # Import the dataset loader to reuse the TacotronSTFT engine and caching logic
    try:
        from loader_tacotron import DatasetLibriSpeechTacotronVAE
    except ImportError:
        import sys
        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(PROJECT_ROOT / "src" / "data" / "loader_vae_tacotron"))
        from loader_tacotron import DatasetLibriSpeechTacotronVAE

    class DummyTextProcessor:
        def text_to_sequence(self, text):
            return [ord(c) for c in text]

    first_path = Path(manifest[0]["mel_path"])
    data_dir = first_path.parent.parent
    dataset = DatasetLibriSpeechTacotronVAE(text_processor=DummyTextProcessor(), data_dir=data_dir)

    features: List[np.ndarray] = []
    valid_manifest: List[Dict[str, Any]] = []
    
    for i, row in enumerate(manifest):
        if max_samples is not None and i >= max_samples:
            break
        try:
            mel_path = Path(row["mel_path"])
            utt_id = row.get("utt_id", mel_path.stem)
            cache_path = dataset.cache_dir / f"{utt_id}.pt"
            
            if cache_path.exists():
                mel_tensor = torch.load(cache_path, map_location="cpu", weights_only=False)
                mel = mel_tensor.numpy()
            else:
                # Load waveform and compute mel using TacotronSTFT engine dynamically, then cache it
                sample = torch.load(mel_path, map_location="cpu", weights_only=False)
                audio = sample["waveform"].squeeze(0)
                sr = sample.get("sr", 22050)
                mel_tensor = dataset.get_mel(audio, orig_freq=sr)
                torch.save(mel_tensor, cache_path)
                mel = mel_tensor.numpy()
            
            # aggregate time: mean across frames
            mel_mean: np.ndarray = mel.mean(axis=1)  # shape (n_mels,)
            features.append(mel_mean)
            valid_manifest.append(row)
        except Exception as e:
            print(f"Error loading {row['mel_path']}: {e}")
            continue
    
    return np.array(features, dtype=np.float32), valid_manifest


def analyze_mel_dimensions(features: np.ndarray, names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Compute dimension statistics for mel spectrograms.
    
    Args:
        features (np.ndarray): Aggregated mel features. Shape (N, n_mels).
        names (Optional[List[str]]): Optional channel names.
    
    Returns:
        Dict[str, Any]: Dictionary with per-dimension statistics.
    """
    if names is None:
        names = [f"mel_{i}" for i in range(features.shape[1])]
    
    stats: Dict[str, Any] = {
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
    """
    Plot histograms of variation in each mel dimension.
    
    Args:
        features (np.ndarray): Mel features. Shape (N, n_mels).
        names (Optional[List[str]]): Channel labels.
        bins (int): Histogram bins.
        figsize (Tuple[int, int]): Figure size.
        save_path (Optional[Path]): Output file path.
    """
    if names is None:
        names = [f"mel_{i}" for i in range(features.shape[1])]
    
    n_dims: int = features.shape[1]
    n_cols: int = 4
    n_rows: int = (n_dims + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes_flat = axes.flatten()
    
    for i, (ax, name) in enumerate(zip(axes_flat, names)):
        ax.hist(features[:, i], bins=bins, alpha=0.7, edgecolor='black')
        ax.set_title(f"{name} (μ={features[:, i].mean():.2f}, σ={features[:, i].std():.2f})")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.grid(alpha=0.3)
    
    # hide unused subplots
    for ax in axes_flat[n_dims:]:
        ax.set_visible(False)
    
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.show()


def compute_tsne(features: np.ndarray, n_components: int = 2, perplexity: int = 30,
                  random_state: int = 42) -> np.ndarray:
    """
    Compute t-SNE embedding of mel features.
    
    Args:
        features (np.ndarray): Input features. Shape (N, D).
        n_components (int): Dimensionality of embedding (2 or 3).
        perplexity (int): t-SNE parameter.
        random_state (int): Seed.
    
    Returns:
        np.ndarray: Embedded features. Shape (N, n_components).
    """
    # standardize features
    scaler: StandardScaler = StandardScaler()
    features_scaled: np.ndarray = scaler.fit_transform(features)
    
    print(f"Computing t-SNE ({features.shape[0]} samples, {features.shape[1]} dims)...")
    tsne: TSNE = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state, n_jobs=-1)
    embedding: np.ndarray = tsne.fit_transform(features_scaled)
    print(f"t-SNE complete. Embedding shape: {embedding.shape}")
    
    return embedding


def plot_tsne_2d(embedding: np.ndarray, metadata: List[Dict[str, Any]], 
                 color_by: Optional[str] = None, figsize: Tuple[int, int] = (10, 8),
                 save_path: Optional[Path] = None) -> None:
    """
    Plot 2D t-SNE embedding.
    
    Args:
        embedding (np.ndarray): 2D points. Shape (N, 2).
        metadata (List[Dict[str, Any]]): Metadata for coloring.
        color_by (Optional[str]): Metadata key to use for color.
        figsize (Tuple[int, int]): Figure size.
        save_path (Optional[Path]): Output path.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if color_by == "duration" and metadata:
        durations: np.ndarray = np.array([m.get("duration", 0.0) for m in metadata])
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
    """
    Plot 3D t-SNE embedding.
    
    Args:
        embedding (np.ndarray): 3D points. Shape (N, 3).
        metadata (List[Dict[str, Any]]): Metadata for coloring.
        color_by (Optional[str]): Metadata key to use for color.
        figsize (Tuple[int, int]): Figure size.
        save_path (Optional[Path]): Output path.
    """
    from mpl_toolkits.mplot3d import Axes3D # type: ignore
    
    fig: plt.Figure = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    if color_by == "duration" and metadata:
        durations: np.ndarray = np.array([m.get("duration", 0.0) for m in metadata])
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
    """
    Plot histogram of audio durations.

    Args:
        metadata (List[Dict[str, Any]]): Manifest data.
        bins (int): Bins.
        figsize (Tuple[int, int]): Size.
        save_path (Optional[Path]): Path.
    """
    durations: np.ndarray = np.array([m.get("duration", 0.0) for m in metadata])
    
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
    """
    Compute correlation matrix between mel dimensions.
    
    Args:
        features (np.ndarray): shape (N, n_mels).
        names (Optional[List[str]]): Dimension names.
    
    Returns:
        np.ndarray: Correlation matrix (n_mels, n_mels).
    """
    corr: np.ndarray = np.corrcoef(features.T)
    return corr


def plot_correlation_matrix(corr_matrix: np.ndarray, names: Optional[List[str]] = None,
                            figsize: Tuple[int, int] = (12, 10),
                            save_path: Optional[Path] = None) -> None:
    """
    Plot correlation matrix heatmap.

    Args:
        corr_matrix (np.ndarray): Matrix.
        names (Optional[List[str]]): Labels.
        figsize (Tuple[int, int]): Size.
        save_path (Optional[Path]): Path.
    """
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
