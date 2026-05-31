"""Compatibility wrapper for the shared last-model data loader."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "data" / "last-model"))

from last_model_data import collate_last_model_batch as collate_direct_tts_batch
from last_model_data import create_dataloaders