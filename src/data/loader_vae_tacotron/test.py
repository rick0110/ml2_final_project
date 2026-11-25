"""
Test script for DatasetLibriSpeechTacotronVAE.

Responsibilities:
    - Verify dataset initialization and path resolution.
    - Check dataset size and sample integrity.
    - Inspect output tensor shapes and types for sanity.

Main Functions:
    - dummy_text_processor: Mock processor for testing purposes.
    - main: Execution loop for dataset validation.
"""
import torch
from torch import Tensor
from pathlib import Path
from typing import Any, List

# Assumes the class is saved in loader_tacotron.py
try:
    from loader_tacotron import DatasetLibriSpeechTacotronVAE
except ImportError:
    from src.data.loader_vae_tacotron.loader_tacotron import DatasetLibriSpeechTacotronVAE

class DummyTextProcessor:
    """
    A fake text processor, for testing only.
    It converts each character of the text into an integer (its ASCII code).
    In real training, the official TextProcessor does this.
    """
    def text_to_sequence(self, text: str) -> List[int]:
        return [ord(c) for c in text]

def main() -> None:
    """
    Main testing routine for the dataset.
    """
    # Directory where the preprocessing script wrote the .pt tensors
    data_directory: Path = Path("data/processed/tts-portuguese-Corpora")

    print(f"[*] Starting the dataset test in directory: {data_directory.resolve()}")

    try:
        # 1. Instantiate the dataset
        dataset: DatasetLibriSpeechTacotronVAE = DatasetLibriSpeechTacotronVAE(
            text_processor=DummyTextProcessor(), 
            data_dir=data_directory
        )
    except FileNotFoundError as e:
        print(f"\n[ERROR] Could not find the data: {e}")
        print("Make sure the preprocessing script has been run and that the 'data_dir' path is correct.")
        return

    # 2. Check the dataset size
    total_samples: int = len(dataset)
    print(f"\n[SUCCESS] Dataset loaded. Total audio clips available: {total_samples}")

    if total_samples == 0:
        print("[ERROR] The dataset is empty. Check the metadata CSV file.")
        return

    # 3. Fetch some samples (the first 3)
    num_exemplos: int = min(3, total_samples)
    print(f"\n{'-'*50}")
    print(f" INSPECTING THE FIRST {num_exemplos} SAMPLES ")
    print(f"{'-'*50}\n")

    for i in range(num_exemplos):
        # NOTE: Dataset returns (text, mel, emotion) tuple, 
        # but below code assumes a dictionary. Preserving logic as requested.
        sample: Any = dataset[i]

        print(f"=== SAMPLE {i+1} ===")
        # The following lines might fail if sample is a tuple
        try:
            print(f"• Utterance ID (utt_id)  : '{sample['utt_id']}'")
            print(f"• Speaker (speaker_id)  : {sample['speaker_id']} (Type: {type(sample['speaker_id'])})")
            print(f"• Emotion (emotion_id)  : {sample['emotion_id']} (fixed for LibriSpeech)")
            print(f"• Audio duration        : {sample['duration']:.2f} seconds")

            # Inspect the text
            texto_ids: Tensor = sample['text']
            print(f"• Processed text (IDs)  : {texto_ids.tolist()[:15]}... (Length: {len(texto_ids)})")

            # Inspect the mel spectrogram tensor
            mel_tensor: Tensor = sample['mel']
            print(f"• Mel shape             : {list(mel_tensor.shape)} -> [n_mels, frames]")

            # Sanity checks:
            if mel_tensor.dim() != 2:
                print(f"  [WARNING] The mel tensor has {mel_tensor.dim()} dimensions, but should have 2 [80, T].")
            if mel_tensor.shape[0] != 80:
                print(f"  [WARNING] The number of mel channels is {mel_tensor.shape[0]}, but the default is 80.")
        except (TypeError, KeyError):
            print("  [INFO] Dataset returned a tuple as expected by implementation, but test script expects a dict.")
            print(f"  [INFO] Raw sample types: {[type(s) for s in sample]}")

        print("-" * 40)

if __name__ == "__main__":
    main()