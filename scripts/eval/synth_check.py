#!/usr/bin/env python3
"""
Synthesize test sentences with a trained Tacotron2-VAE checkpoint and check them.

For each sentence and each reference utterance (the style source) it:
  1. generates the mel with the Tacotron2-VAE and the audio with HiFi-GAN;
  2. transcribes the audio with Whisper and computes the word error rate (WER)
     against the requested text;
  3. records whether the decoder stopped by itself, the audio duration, the mean
     F0 and the speaking rate, so style transfer can be measured, not just heard.

Outputs (in --out-dir): one .wav per sample, a mel + alignment figure, and
results.csv with one row per sample.

Example:
    python scripts/eval/synth_check.py --checkpoint experiments/tacotron2-vae/en_vae_v1/checkpoints/epoch_6000 \
        --hifigan ckpt/hifigan_universal_v1.pt --data-dir data/processed/LJSpeech \
        --text-cleaner english_cleaners --language en --out-dir exports/eval/en_vae_v1
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List

import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "training" / "training-tacotron2-vae"))
from models.hifigan import load_hifigan, vocode  # noqa: E402
from models.tacotron2_vae.hparams import Tacotron2VAEHparams  # noqa: E402
from models.tacotron2_vae.layers import TacotronSTFT  # noqa: E402
from models.tacotron2_vae.model import Tacotron2  # noqa: E402
from text_processing import TextProcessor  # noqa: E402

SENTENCES: Dict[str, List[str]] = {
    "en": [
        "The results were published yesterday afternoon.",
        "She had a voice that could make a stone weep.",
        "The system returned an unexpected error.",
        "Please wait while the movie is loading.",
        "I never thought I would see you here again.",
        "Where did you put the keys to the car?",
    ],
    "pt": [
        "Por favor, aguarde enquanto o sistema carrega.",
        "Eu nunca pensei que veria você aqui de novo.",
        "Onde você colocou as chaves do carro?",
        "O filme começa às oito horas da noite.",
        "Ela abriu a porta devagar e olhou para dentro.",
        "Amanhã vamos viajar para o Rio de Janeiro.",
    ],
}


def words(s: str) -> List[str]:
    return re.sub(r"[^\w\s]", "", s.lower()).split()


def wer(ref: str, hyp: str) -> float:
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
    return min(1.0, d[-1] / max(len(r), 1))


def prosody(audio: np.ndarray, n_chars: int) -> Dict[str, float]:
    f0, voiced, _ = librosa.pyin(audio, fmin=60, fmax=500, sr=22050, frame_length=1024, hop_length=256)
    f0 = f0[voiced] if voiced is not None and voiced.any() else np.array([np.nan])
    dur = len(audio) / 22050
    return {"f0_hz": float(np.nanmean(f0)), "f0_std_hz": float(np.nanstd(f0)), "chars_per_s": n_chars / max(dur, 1e-3)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--hifigan", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True, help="Processed corpus with mels_metadata.csv (reference audio)")
    ap.add_argument("--text-cleaner", required=True)
    ap.add_argument("--language", choices=["en", "pt"], required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--references", nargs="*", default=None, help="utt_ids to use as style references")
    ap.add_argument("--all-references", action="store_true", help="Use every utterance of --data-dir as a reference")
    ap.add_argument("--n-references", type=int, default=3, help="If --references is not given: shortest, median, longest-rate picks")
    ap.add_argument("--style-strength", type=float, default=1.0, help="Scale of the reference latent code (1 = as is, 0 = prior mean)")
    ap.add_argument("--style-origin", type=Path, default=None,
                    help=".npy latent vector that --style-strength contracts towards (default: the prior mean, 0)")
    ap.add_argument("--whisper", default="small")
    ap.add_argument("--max-decoder-steps", type=int, default=1200)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a.out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    hp = Tacotron2VAEHparams.from_dict(ckpt["hparams"])
    hp.max_decoder_steps = a.max_decoder_steps
    model = Tacotron2(hp)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()
    tp = TextProcessor(cleaner_names=[a.text_cleaner])
    voc = load_hifigan(a.hifigan, device)
    stft = TacotronSTFT(1024, 256, 1024, sampling_rate=22050, mel_fmin=0.0, mel_fmax=8000.0)

    with open(a.data_dir / "mels_metadata.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_id = {r["utt_id"]: r for r in rows}
    if a.all_references:
        refs = rows
    elif a.references:
        refs = [by_id[u] for u in a.references]
    else:  # spread over speaking rate: slow, typical, fast
        rated = sorted(rows, key=lambda r: len(r["text"]) / float(r["duration"]))
        picks = np.linspace(0.05, 0.95, a.n_references)
        refs = [rated[int(p * (len(rated) - 1))] for p in picks]

    from faster_whisper import WhisperModel
    asr = WhisperModel(a.whisper, device="cpu", compute_type="int8")

    origin = torch.zeros(1, hp.z_latent_dim, device=device)
    if a.style_origin:
        origin = torch.from_numpy(np.load(a.style_origin)).float().view(1, -1).to(device)

    out_rows = []
    for ref in refs:
        wav = torch.load(ref["mel_path"], map_location="cpu", weights_only=True)["waveform"]
        ref_mel = stft.mel_spectrogram(wav.clamp(-1, 1)).to(device)
        ref_pros = prosody(wav[0].numpy(), len(ref["text"]))
        for si, sentence in enumerate(SENTENCES[a.language]):
            seq = torch.LongTensor(tp.text_to_sequence(sentence)).unsqueeze(0).to(device)
            with torch.inference_mode():
                _, mu, _, _ = model.vae_gst(ref_mel)  # (1, L)
                z = origin + a.style_strength * (mu - origin)
                _, mel, align = model.inference_from_latent(seq, z)
            stopped = mel.size(2) < a.max_decoder_steps
            audio = vocode(voc, mel[0].float()).cpu().numpy()
            name = f"{ref['utt_id']}__s{si}"
            sf.write(a.out_dir / f"{name}.wav", audio, 22050)
            segs, _ = asr.transcribe(librosa.resample(audio, orig_sr=22050, target_sr=16000), language=a.language, beam_size=5)
            heard = " ".join(s.text.strip() for s in segs)
            pros = prosody(audio, len(sentence))
            ali = align[0].float().cpu().numpy()
            out_rows.append({
                "sample": name, "reference": ref["utt_id"], "reference_emotion": ref.get("emotion", ""),
                "reference_speaker": ref.get("speaker", ""), "text": sentence, "heard": heard,
                "wer": round(wer(sentence, heard), 3), "stopped_by_itself": int(stopped),
                "duration_s": round(len(audio) / 22050, 2), "align_peak": round(float(ali.max(axis=1).mean()), 3),
                "latent_norm": round(float(z.norm()), 3), "latent_shift": round(float((z - origin).norm()), 3),
                "f0_hz": round(pros["f0_hz"], 1), "f0_std_hz": round(pros["f0_std_hz"], 1), "chars_per_s": round(pros["chars_per_s"], 2),
                "ref_f0_hz": round(ref_pros["f0_hz"], 1), "ref_chars_per_s": round(ref_pros["chars_per_s"], 2),
            })
            if si == 0:
                fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
                ax[0].imshow(mel[0].float().cpu().numpy(), aspect="auto", origin="lower")
                ax[0].set_title(f"Mel gerado | ref {ref['utt_id']}")
                ax[1].imshow(ali.T, aspect="auto", origin="lower")
                ax[1].set_title("Alinhamento texto x audio")
                fig.tight_layout()
                fig.savefig(a.out_dir / f"{name}.png", dpi=70)
                plt.close(fig)
            print(f"{name}: WER {out_rows[-1]['wer']:.2f} | {heard}")

    with open(a.out_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    mean_wer = float(np.mean([r["wer"] for r in out_rows]))
    stopped = float(np.mean([r["stopped_by_itself"] for r in out_rows]))
    print(f"MEAN WER {mean_wer:.3f} over {len(out_rows)} samples | stopped by itself: {stopped:.0%}")


if __name__ == "__main__":
    main()
