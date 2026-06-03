#!/usr/bin/env python3

from pathlib import Path
import csv
import statistics
import torch
from tqdm import tqdm


DATA_DIR = Path("data/processed/tts-portuguese-Corpora")
CSV_PATH = DATA_DIR / "mels_metadata.csv"


def fail(msg):
    raise RuntimeError(msg)


def main():

    if not CSV_PATH.exists():
        fail(f"Metadata file not found: {CSV_PATH}")

    rows = []

    with open(CSV_PATH, "r", encoding="utf8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append(row)

    if len(rows) == 0:
        fail("Empty metadata file")

    print(f"Loaded {len(rows)} metadata entries")

    durations = []
    frame_lengths = []

    missing_files = 0
    nan_count = 0
    inf_count = 0
    empty_text_count = 0

    for row in tqdm(rows):

        mel_path = Path(row["mel_path"])

        if not mel_path.exists():
            print(f"Missing file: {mel_path}")
            missing_files += 1
            continue

        sample = torch.load(mel_path, map_location="cpu")

        required_keys = {
            "waveform",
            "mel",
            "sr",
            "duration",
            "text",
        }

        missing = required_keys - set(sample.keys())

        if missing:
            fail(f"{mel_path} missing keys: {missing}")

        mel = sample["mel"]

        if not isinstance(mel, torch.Tensor):
            fail(f"{mel_path}: mel is not tensor")

        if mel.ndim != 3:
            fail(
                f"{mel_path}: expected mel ndim=3 "
                f"(1,n_mels,T), got {mel.shape}"
            )

        if mel.shape[0] != 1:
            fail(
                f"{mel_path}: expected channel dimension=1, "
                f"got {mel.shape}"
            )

        if mel.shape[1] != 80:
            print(
                f"WARNING: unexpected n_mels "
                f"{mel.shape[1]} in {mel_path}"
            )

        if torch.isnan(mel).any():
            nan_count += 1

        if torch.isinf(mel).any():
            inf_count += 1

        duration = float(sample["duration"])

        durations.append(duration)
        frame_lengths.append(mel.shape[-1])

        text = str(sample["text"]).strip()

        if len(text) == 0:
            empty_text_count += 1

        sr = int(sample["sr"])

        if sr != 22050:
            print(
                f"WARNING: {mel_path} "
                f"sample rate = {sr}"
            )

    print()
    print("========== SUMMARY ==========")

    print(f"Samples: {len(rows)}")
    print(f"Missing files: {missing_files}")
    print(f"NaN tensors: {nan_count}")
    print(f"Inf tensors: {inf_count}")
    print(f"Empty texts: {empty_text_count}")

    print()

    print(
        f"Duration min: {min(durations):.2f}s"
    )
    print(
        f"Duration mean: {statistics.mean(durations):.2f}s"
    )
    print(
        f"Duration max: {max(durations):.2f}s"
    )

    print()

    print(
        f"Frames min: {min(frame_lengths)}"
    )
    print(
        f"Frames mean: {statistics.mean(frame_lengths):.1f}"
    )
    print(
        f"Frames max: {max(frame_lengths)}"
    )

    print()

    if (
        missing_files
        or nan_count
        or inf_count
    ):
        fail("Dataset validation failed")

    print("Dataset validation passed")


if __name__ == "__main__":
    main()