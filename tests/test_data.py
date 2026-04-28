"""
Unit tests for data utilities and dataset classes.

These tests use temporary directories with synthetic audio files so no real
dataset is required.
"""

import csv
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torchaudio

# tests/__init__.py adds src/ to sys.path
from data.preprocessing import (
    AudioPreprocessor,
    extract_mel_spectrogram,
    extract_pitch,
    extract_energy,
    load_audio,
)
from data.dataset import (
    TTSPortugueseDataset,
    LibriVoxPTBRDataset,
    VERBODataset,
    ProsodyTransferDataset,
    collate_fn,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SR = 22050
N_MELS = 80


def _make_wav(path: Path, duration_s: float = 0.5, sr: int = SR) -> None:
    """Write a silent WAV file to *path*."""
    n_samples = int(duration_s * sr)
    waveform = torch.zeros(1, n_samples)
    torchaudio.save(str(path), waveform, sr)


def _make_tts_portuguese_corpus(root: Path, n_utterances: int = 5) -> None:
    """Create a minimal TTS-Portuguese corpus fixture."""
    wavs_dir = root / "wavs"
    wavs_dir.mkdir(parents=True)
    rows = []
    for i in range(n_utterances):
        file_id = f"utt_{i:04d}"
        _make_wav(wavs_dir / f"{file_id}.wav")
        rows.append({"file_id": file_id, "text": f"Texto de teste {i}"})
    with open(root / "metadata.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_id", "text"])
        writer.writeheader()
        writer.writerows(rows)


def _make_librivox_ptbr(root: Path, n_files: int = 6) -> None:
    """Create a minimal LibriVox PT-BR corpus fixture."""
    for chapter in range(2):
        chapter_dir = root / f"chapter_{chapter:02d}"
        chapter_dir.mkdir(parents=True, exist_ok=True)
        for utt in range(n_files // 2):
            _make_wav(chapter_dir / f"utt_{utt:04d}.flac")


def _make_verbo(root: Path) -> None:
    """Create a minimal VERBO corpus fixture."""
    for spk in ["spk001", "spk002"]:
        for emotion in ["neutral", "happy", "sad"]:
            emot_dir = root / spk / emotion
            emot_dir.mkdir(parents=True, exist_ok=True)
            _make_wav(emot_dir / "sample_01.wav")


# ---------------------------------------------------------------------------
# Audio preprocessing tests
# ---------------------------------------------------------------------------

class TestAudioPreprocessor:
    """Tests for audio feature extraction utilities."""

    def test_load_audio_mono(self, tmp_path):
        wav_path = tmp_path / "test.wav"
        _make_wav(wav_path)
        waveform, sr = load_audio(str(wav_path), target_sr=SR)
        assert waveform.shape[0] == 1  # mono
        assert sr == SR

    def test_load_audio_resampling(self, tmp_path):
        """Loading at a different target_sr should resample."""
        wav_path = tmp_path / "test.wav"
        _make_wav(wav_path, sr=44100)
        waveform, sr = load_audio(str(wav_path), target_sr=SR)
        assert sr == SR

    def test_mel_spectrogram_shape(self, tmp_path):
        waveform = torch.randn(1, SR)
        mel = extract_mel_spectrogram(waveform, sample_rate=SR, n_mels=N_MELS)
        assert mel.dim() == 2
        assert mel.shape[0] == N_MELS

    def test_mel_spectrogram_1d_input(self):
        waveform = torch.randn(SR)
        mel = extract_mel_spectrogram(waveform, sample_rate=SR, n_mels=N_MELS)
        assert mel.shape[0] == N_MELS

    def test_pitch_shape(self):
        waveform = torch.zeros(SR)  # silent → F0 = 0
        pitch = extract_pitch(waveform, sample_rate=SR)
        assert pitch.dim() == 1
        assert pitch.shape[0] > 0

    def test_energy_shape(self):
        waveform = torch.randn(1, SR)
        energy = extract_energy(waveform)
        assert energy.dim() == 1
        assert energy.shape[0] > 0

    def test_preprocessor_process(self, tmp_path):
        wav_path = tmp_path / "test.wav"
        _make_wav(wav_path)
        preprocessor = AudioPreprocessor(sample_rate=SR, n_mels=N_MELS)
        features = preprocessor.process(str(wav_path))
        assert set(features.keys()) == {"waveform", "mel", "pitch", "energy"}
        assert features["mel"].shape[0] == N_MELS


# ---------------------------------------------------------------------------
# TTSPortugueseDataset tests
# ---------------------------------------------------------------------------

class TestTTSPortugueseDataset:
    def test_length(self, tmp_path):
        _make_tts_portuguese_corpus(tmp_path, n_utterances=4)
        ds = TTSPortugueseDataset(root=tmp_path)
        assert len(ds) == 4

    def test_sample_keys(self, tmp_path):
        _make_tts_portuguese_corpus(tmp_path, n_utterances=2)
        ds = TTSPortugueseDataset(root=tmp_path)
        sample = ds[0]
        for key in ("id", "text", "waveform", "mel", "pitch", "energy", "dataset"):
            assert key in sample, f"Missing key: {key}"

    def test_dataset_tag(self, tmp_path):
        _make_tts_portuguese_corpus(tmp_path, n_utterances=1)
        ds = TTSPortugueseDataset(root=tmp_path)
        assert ds[0]["dataset"] == "tts_portuguese"

    def test_missing_metadata_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            TTSPortugueseDataset(root=tmp_path)


# ---------------------------------------------------------------------------
# LibriVoxPTBRDataset tests
# ---------------------------------------------------------------------------

class TestLibriVoxPTBRDataset:
    def test_length(self, tmp_path):
        _make_librivox_ptbr(tmp_path, n_files=6)
        ds = LibriVoxPTBRDataset(root=tmp_path)
        assert len(ds) == 6

    def test_sample_keys(self, tmp_path):
        _make_librivox_ptbr(tmp_path, n_files=2)
        ds = LibriVoxPTBRDataset(root=tmp_path)
        sample = ds[0]
        for key in ("id", "waveform", "mel", "pitch", "energy", "dataset"):
            assert key in sample

    def test_empty_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            LibriVoxPTBRDataset(root=tmp_path)


# ---------------------------------------------------------------------------
# VERBODataset tests
# ---------------------------------------------------------------------------

class TestVERBODataset:
    def test_length(self, tmp_path):
        _make_verbo(tmp_path)
        ds = VERBODataset(root=tmp_path)
        # 2 speakers × 3 emotions × 1 file each
        assert len(ds) == 6

    def test_sample_keys(self, tmp_path):
        _make_verbo(tmp_path)
        ds = VERBODataset(root=tmp_path)
        sample = ds[0]
        for key in ("id", "speaker_id", "emotion", "waveform", "mel", "pitch", "energy", "dataset"):
            assert key in sample

    def test_emotion_filter(self, tmp_path):
        _make_verbo(tmp_path)
        ds = VERBODataset(root=tmp_path, emotions=["happy"])
        assert len(ds) == 2  # 2 speakers × 1 emotion

    def test_dataset_tag(self, tmp_path):
        _make_verbo(tmp_path)
        ds = VERBODataset(root=tmp_path)
        assert ds[0]["dataset"] == "verbo"


# ---------------------------------------------------------------------------
# ProsodyTransferDataset tests
# ---------------------------------------------------------------------------

class TestProsodyTransferDataset:
    def test_length(self, tmp_path):
        content_root = tmp_path / "tts"
        ref_root = tmp_path / "verbo"
        _make_tts_portuguese_corpus(content_root, n_utterances=3)
        _make_verbo(ref_root)
        content_ds = TTSPortugueseDataset(root=content_root)
        ref_ds = VERBODataset(root=ref_root)
        combined = ProsodyTransferDataset(content_ds, ref_ds)
        assert len(combined) == 3  # equals length of content dataset

    def test_sample_keys(self, tmp_path):
        content_root = tmp_path / "tts"
        ref_root = tmp_path / "verbo"
        _make_tts_portuguese_corpus(content_root, n_utterances=2)
        _make_verbo(ref_root)
        ds = ProsodyTransferDataset(
            TTSPortugueseDataset(root=content_root),
            VERBODataset(root=ref_root),
        )
        sample = ds[0]
        for key in ("source_waveform", "source_mel", "ref_mel", "source_id", "ref_id"):
            assert key in sample


# ---------------------------------------------------------------------------
# collate_fn tests
# ---------------------------------------------------------------------------

class TestCollate:
    def test_pads_to_max_length(self):
        """collate_fn should pad shorter tensors to match the longest."""
        batch = [
            {"mel": torch.randn(N_MELS, 50), "id": "a"},
            {"mel": torch.randn(N_MELS, 80), "id": "b"},
        ]
        result = collate_fn(batch)
        assert result["mel"].shape == (2, N_MELS, 80)

    def test_non_tensor_values_collected_as_list(self):
        batch = [{"id": "x", "mel": torch.randn(N_MELS, 10)}] * 3
        result = collate_fn(batch)
        assert isinstance(result["id"], list)
        assert len(result["id"]) == 3
