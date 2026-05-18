import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(os.getcwd())

try:
    from src.data.first_step_data_loaders.datasets import (
        LibriSpeechPTDataset,
        TTSPortugueseDataset,
        CombinedFirstStepDataset
    )
    import torch

    print("Imports successful.")

    # Instantiate datasets using default roots (or split for LibriSpeech)
    try:
        libri_ds = LibriSpeechPTDataset(split="train")
        print(f"LibriSpeech Dataset (train) length: {len(libri_ds)}")
        if len(libri_ds) > 0:
            print(f"LibriSpeech sample keys: {libri_ds[0].keys()}")
    except Exception as e:
        print(f"Error building LibriSpeech dataset: {e}")

    try:
        tts_ds = TTSPortugueseDataset()
        print(f"TTS Portuguese Dataset length: {len(tts_ds)}")
        if len(tts_ds) > 0:
            print(f"TTS Portuguese sample keys: {tts_ds[0].keys()}")
    except Exception as e:
        print(f"Error building TTS Portuguese dataset: {e}")

    try:
        # CombinedFirstStepDataset takes a sequence of datasets
        combined_ds = CombinedFirstStepDataset([libri_ds, tts_ds])
        print(f"Combined Dataset length: {len(combined_ds)}")
        if len(combined_ds) > 0:
            print(f"Combined sample keys: {combined_ds[0].keys()}")
    except Exception as e:
        print(f"Error building Combined dataset: {e}")

except ImportError as e:
    print(f"Import Error: {e}")
except Exception as e:
    print(f"Unexpected Error: {e}")
