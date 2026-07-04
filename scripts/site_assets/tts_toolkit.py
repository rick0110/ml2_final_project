"""Shared toolkit: load Tacotron2-VAE models + WaveGlow, synthesize, extract latents/attention."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

PROJECT_ROOT = Path("/home/al.richard.viana/Documentos/ml2_final_project")
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "training" / "training-tacotron2-vae"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "models" / "waveglow"))

from models.tacotron2_vae.hparams import Tacotron2VAEHparams
from models.tacotron2_vae.model import load_tacotron2_vae_model, Tacotron2
from text_processing import TextProcessor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EXPS = {
    "baseline_lj": PROJECT_ROOT / "experiments/drive/experiments_maquina_herivelton/tacotron2-vae (1)/lj_speech_v1",
    "attn_lj": PROJECT_ROOT / "experiments/drive/experiments_with_time_intelligible_attention/tacotron2-vae (1)/lj_speech_v1",
    "verbo_v4_f1": PROJECT_ROOT / "experiments/tacotron2-vae/verbo_v4_f1",
}
BEST_CKPT = {"baseline_lj": "epoch_11200", "attn_lj": "epoch_35500", "verbo_v4_f1": "epoch_9500"}


def load_model(name: str, ckpt_name: str | None = None):
    exp_dir = EXPS[name]
    ckpt_path = exp_dir / "checkpoints" / (ckpt_name or BEST_CKPT[name])
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    hparams = Tacotron2VAEHparams.from_dict(ckpt["hparams"])
    model = load_tacotron2_vae_model(hparams, device=DEVICE)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    has_attn = not any("ref_encoder.attn" in k for k in missing)
    if not has_attn:
        # baseline: reference encoder pooled with the GRU's last hidden state
        def _encode_last(self, gru_input):
            out, hidden = self.gru(gru_input)
            return hidden.squeeze(0), None
        model.vae_gst.ref_encoder._encode = types.MethodType(
            _encode_last, model.vae_gst.ref_encoder
        )
    assert not unexpected, unexpected
    other_missing = [k for k in missing if "ref_encoder.attn" not in k]
    assert not other_missing, other_missing
    model.eval()
    tp = TextProcessor.load(exp_dir / "symbols.json")
    return model, tp, hparams, has_attn, ckpt


_WAVEGLOW = None


def load_waveglow():
    global _WAVEGLOW
    if _WAVEGLOW is None:
        from glow import WaveGlow  # noqa: F401 (needed for unpickling)
        ck = torch.load(
            PROJECT_ROOT / "local_weight_models/waveglow/waveglow_256channels_ljs_v3.pt",
            map_location=DEVICE, weights_only=False,
        )
        wg = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        wg = wg.remove_weightnorm(wg)
        _WAVEGLOW = wg.to(DEVICE).eval()
    return _WAVEGLOW


def load_mel(path: str | Path) -> Tensor:
    sample = torch.load(PROJECT_ROOT / path, map_location="cpu", weights_only=False)
    mel = sample["mel"]
    while mel.dim() > 2:
        mel = mel.squeeze(0)
    return mel.unsqueeze(0)  # (1, n_mels, T)


def load_sample(path: str | Path) -> dict:
    return torch.load(PROJECT_ROOT / path, map_location="cpu", weights_only=False)


@torch.inference_mode()
def mel_to_audio(mel: Tensor, sigma: float = 0.6) -> np.ndarray:
    wg = load_waveglow()
    audio = wg.infer(mel.to(DEVICE), sigma=sigma)
    a = audio.squeeze().float().cpu().numpy()
    return np.clip(a, -1.0, 1.0)


@torch.inference_mode()
def synth_mel_from_style(model: Tacotron2, tp: TextProcessor, text: str,
                         style_embed: Tensor) -> tuple[Tensor, Tensor]:
    """Decode text with a precomputed style embedding (1, E). Returns (mel_postnet, alignments)."""
    seq = torch.LongTensor(tp.text_to_sequence(text)).unsqueeze(0).to(DEVICE)
    emb = model.transcript_embedding(seq).transpose(1, 2)
    transcript_outputs = model.encoder.inference(emb)
    lat = style_embed.unsqueeze(1).expand_as(transcript_outputs)
    encoder_outputs = transcript_outputs + lat
    mel_outputs, _, alignments = model.decoder.inference(encoder_outputs)
    mel_post = model.postnet(mel_outputs) + mel_outputs
    return mel_post, alignments


@torch.inference_mode()
def style_from_ref(model: Tacotron2, ref_mel: Tensor):
    """Returns (style_embed (1,E), mu (1,L), weights (1,T') or None)."""
    ref = ref_mel.to(DEVICE)
    enc = model.vae_gst.ref_encoder
    out = enc._cnn_forward(ref)
    encoded, weights = enc._encode(out)
    mu = model.vae_gst.fc1(encoded)
    style = model.vae_gst.fc3(mu)  # inference uses z = mu
    return style, mu, (None if weights is None else weights)


@torch.inference_mode()
def style_from_z(model: Tacotron2, z: Tensor) -> Tensor:
    return model.vae_gst.fc3(z.to(DEVICE))


@torch.inference_mode()
def synth_from_ref(model, tp, text, ref_mel, sigma=0.6):
    style, mu, w = style_from_ref(model, ref_mel)
    mel, align = synth_mel_from_style(model, tp, text, style)
    return mel_to_audio(mel, sigma), mel, align, mu, w


def save_wav(audio: np.ndarray, path: Path, sr: int = 22050):
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32), sr)


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return (1 - t) * a + t * b


def slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Spherical interpolation. Interpolates direction on the great circle and
    the norm geometrically-linearly, so intermediate vectors keep a typical radius."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    ua, ub = a / na, b / nb
    dot = np.clip(np.dot(ua, ub), -1.0, 1.0)
    omega = np.arccos(dot)
    if omega < 1e-6:
        direction = ua
    else:
        direction = (np.sin((1 - t) * omega) * ua + np.sin(t * omega) * ub) / np.sin(omega)
    norm = (1 - t) * na + t * nb
    return direction * norm
