"""Generate all site assets: style-transfer audio, attention data, latent stats,
latent manipulation audio, and slerp/lerp interpolation audio for the 3 models."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import tts_toolkit as tk
from tts_toolkit import PROJECT_ROOT, DEVICE

DOCS = PROJECT_ROOT / "docs"
AUD = DOCS / "assets" / "audio_wav"
ANA = Path(__file__).parent / "analysis"
AUD.mkdir(parents=True, exist_ok=True)
ANA.mkdir(parents=True, exist_ok=True)

MANIFEST = {}

LJ_REFS = json.loads((Path(__file__).parent / "lj_refs.json").read_text())
LJ_TEXTS = {
    "t1": "The birch canoe slid on the smooth planks.",
    "t2": "Machine learning lets computers speak with feeling and style.",
}
VERBO_TEXTS = {
    "t1": "O aprendizado de máquina permite que computadores falem com emoção.",
}
EMOTIONS = ["neutro", "alegria", "tristeza", "raiva", "medo", "surpresa", "desgosto"]

verbo_rows = list(csv.DictReader(open(PROJECT_ROOT / "data/processed/VERBO_f1/mels_metadata.csv")))


def wav(name: str, audio: np.ndarray):
    tk.save_wav(audio, AUD / f"{name}.wav")
    print("  wav:", name, f"{len(audio)/22050:.1f}s", flush=True)


def export_gt(sample: dict, name: str):
    w = sample["waveform"].squeeze().numpy().astype(np.float32)
    tk.save_wav(np.clip(w, -1, 1), AUD / f"{name}.wav", sr=int(sample["sr"]))


# ---------------------------------------------------------------- LJ models
def lj_style_transfer():
    for style, info in LJ_REFS.items():
        export_gt(tk.load_sample(info["path"]), f"lj_ref_{style}")
    for mname, tag in [("baseline_lj", "base"), ("attn_lj", "attn")]:
        model, tp, hp, has_attn, _ = tk.load_model(mname)
        for style, info in LJ_REFS.items():
            ref = tk.load_mel(info["path"])
            for tid, text in LJ_TEXTS.items():
                audio, mel, align, mu, w = tk.synth_from_ref(model, tp, text, ref)
                wav(f"lj_{tag}_{style}_{tid}", audio)
        del model
        torch.cuda.empty_cache()


def lj_attention_and_latent():
    model, tp, hp, _, _ = tk.load_model("attn_lj")
    # attention weights on the 3 style refs
    attn_data = {}
    for style, info in LJ_REFS.items():
        ref = tk.load_mel(info["path"])
        _, mu, w = tk.style_from_ref(model, ref)
        attn_data[style] = {
            "weights": w.squeeze(0).cpu().tolist(),
            "mel": ref.squeeze(0).numpy().tolist(),
            "utt": info["utt"], "text": info["text"], "dur": info["dur"],
        }
    (ANA / "lj_attention.json").write_text(json.dumps(attn_data))

    # latent stats over a fixed LJ sample, for BOTH LJ models
    import random
    random.seed(11)
    rows = list(csv.DictReader(open(PROJECT_ROOT / "data/processed/LJSpeech-tacotron-vae/mels_metadata.csv")))
    rows = [r for r in rows if 2.0 <= float(r["duration"]) <= 8.0]
    subset = random.sample(rows, 400)
    for mname, tag in [("baseline_lj", "base"), ("attn_lj", "attn")]:
        m, tpp, _, _, _ = tk.load_model(mname)
        mus = []
        for r in subset:
            _, mu, _ = tk.style_from_ref(m, tk.load_mel(r["mel_path"]))
            mus.append(mu.squeeze(0).cpu().numpy())
        mus = np.stack(mus)
        np.savez(ANA / f"lj_mu_{tag}.npz", mu=mus)
        print(f"{tag}: mu matrix {mus.shape}, top var dims:",
              np.argsort(mus.var(0))[::-1][:5], flush=True)
        del m
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- VERBO model
def verbo_style_transfer_and_attention():
    model, tp, hp, _, _ = tk.load_model("verbo_v4_f1")
    # refs: phrase l1 for each emotion (parallel corpus)
    ref_by_emo = {}
    for e in EMOTIONS:
        r = next(r for r in verbo_rows if r["emotion_pt"] == e and r["phrase_id"] == "l1")
        ref_by_emo[e] = r
        export_gt(tk.load_sample(r["mel_path"]), f"verbo_ref_{e}")
    for e, r in ref_by_emo.items():
        ref = tk.load_mel(r["mel_path"])
        for tid, text in VERBO_TEXTS.items():
            audio, mel, align, mu, w = tk.synth_from_ref(model, tp, text, ref)
            wav(f"verbo_{e}_{tid}", audio)

    # attention weights + mu on ALL f1 utterances
    per_utt = []
    for r in verbo_rows:
        ref = tk.load_mel(r["mel_path"])
        _, mu, w = tk.style_from_ref(model, ref)
        mel = ref.squeeze(0).numpy()
        wnp = w.squeeze(0).cpu().numpy()
        # frame-energy proxy per attention window (64 mel frames per T' step)
        T = mel.shape[1]
        energy = mel.mean(0)
        bins = [energy[i * 64:(i + 1) * 64].mean() for i in range(len(wnp))]
        per_utt.append({
            "utt": r["utt_id"], "emotion": r["emotion_pt"], "phrase": r["phrase_id"],
            "weights": wnp.tolist(), "energy_bins": [float(b) for b in bins],
            "mu": mu.squeeze(0).cpu().tolist(), "dur": float(r["duration"]),
        })
    (ANA / "verbo_attention_mu.json").write_text(json.dumps(per_utt))
    # example mels for overlay figures (l1 of 3 contrasting emotions)
    ex = {}
    for e in ["alegria", "tristeza", "raiva"]:
        r = ref_by_emo[e]
        ex[e] = {"mel": tk.load_mel(r["mel_path"]).squeeze(0).numpy().tolist(),
                 "utt": r["utt_id"], "text": r["text"]}
    (ANA / "verbo_example_mels.json").write_text(json.dumps(ex))
    del model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- latent manipulation
def latent_manipulation():
    """For each model: top-2 variance dims; base z=mu(ref); set dim to mean±3σ."""
    specs = [
        ("baseline_lj", "base", "lj_mu_base.npz",
         LJ_REFS["grave_lenta"]["path"], LJ_TEXTS["t1"]),
        ("attn_lj", "attn", "lj_mu_attn.npz",
         LJ_REFS["grave_lenta"]["path"], LJ_TEXTS["t1"]),
    ]
    out_info = {}
    for mname, tag, mufile, refpath, text in specs:
        model, tp, _, _, _ = tk.load_model(mname)
        mus = np.load(ANA / mufile)["mu"]
        var = mus.var(0)
        top = np.argsort(var)[::-1][:2]
        mean, std = mus.mean(0), mus.std(0)
        ref = tk.load_mel(refpath)
        _, mu_ref, _ = tk.style_from_ref(model, ref)
        z0 = mu_ref.squeeze(0).cpu().numpy()
        audio, *_ = tk.synth_from_ref(model, tp, text, ref)
        wav(f"lat_{tag}_base", audio)
        dims_info = []
        for d in top:
            for k, lab in [(-3, "m3"), (3, "p3")]:
                z = z0.copy()
                z[d] = mean[d] + k * std[d]
                style = tk.style_from_z(model, torch.tensor(z, dtype=torch.float32).unsqueeze(0))
                mel, _ = tk.synth_mel_from_style(model, tp, text, style)
                wav(f"lat_{tag}_dim{d}_{lab}", tk.mel_to_audio(mel))
            dims_info.append({"dim": int(d), "var": float(var[d]),
                              "mean": float(mean[d]), "std": float(std[d])})
        out_info[tag] = {"top_dims": dims_info, "var_all": var.tolist()}
        del model
        torch.cuda.empty_cache()

    # verbo: mu from verbo_attention_mu.json
    model, tp, _, _, _ = tk.load_model("verbo_v4_f1")
    per_utt = json.loads((ANA / "verbo_attention_mu.json").read_text())
    mus = np.array([u["mu"] for u in per_utt])
    var = mus.var(0)
    top = np.argsort(var)[::-1][:2]
    mean, std = mus.mean(0), mus.std(0)
    r_neu = next(r for r in verbo_rows if r["emotion_pt"] == "neutro" and r["phrase_id"] == "l1")
    ref = tk.load_mel(r_neu["mel_path"])
    text = VERBO_TEXTS["t1"]
    _, mu_ref, _ = tk.style_from_ref(model, ref)
    z0 = mu_ref.squeeze(0).cpu().numpy()
    audio, *_ = tk.synth_from_ref(model, tp, text, ref)
    wav("lat_verbo_base", audio)
    dims_info = []
    for d in top:
        for k, lab in [(-3, "m3"), (3, "p3")]:
            z = z0.copy()
            z[d] = mean[d] + k * std[d]
            style = tk.style_from_z(model, torch.tensor(z, dtype=torch.float32).unsqueeze(0))
            mel, _ = tk.synth_mel_from_style(model, tp, text, style)
            wav(f"lat_verbo_dim{d}_{lab}", tk.mel_to_audio(mel))
        dims_info.append({"dim": int(d), "var": float(var[d]),
                          "mean": float(mean[d]), "std": float(std[d])})
    out_info["verbo"] = {"top_dims": dims_info, "var_all": var.tolist()}
    (ANA / "latent_info.json").write_text(json.dumps(out_info, indent=1))
    del model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- interpolation
def interpolation():
    specs = [
        ("baseline_lj", "base", LJ_REFS["grave_lenta"]["path"],
         LJ_REFS["expressiva"]["path"], LJ_TEXTS["t1"]),
        ("attn_lj", "attn", LJ_REFS["grave_lenta"]["path"],
         LJ_REFS["expressiva"]["path"], LJ_TEXTS["t1"]),
        ("verbo_v4_f1", "verbo", None, None, VERBO_TEXTS["t1"]),
    ]
    interp_info = {}
    for mname, tag, pa, pb, text in specs:
        model, tp, _, _, _ = tk.load_model(mname)
        if tag == "verbo":
            ra = next(r for r in verbo_rows if r["emotion_pt"] == "tristeza" and r["phrase_id"] == "l2")
            rb = next(r for r in verbo_rows if r["emotion_pt"] == "alegria" and r["phrase_id"] == "l2")
            pa, pb = ra["mel_path"], rb["mel_path"]
        _, mu_a, _ = tk.style_from_ref(model, tk.load_mel(pa))
        _, mu_b, _ = tk.style_from_ref(model, tk.load_mel(pb))
        a = mu_a.squeeze(0).cpu().numpy()
        b = mu_b.squeeze(0).cpu().numpy()
        angle = float(np.degrees(np.arccos(np.clip(
            np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))))
        interp_info[tag] = {
            "norm_a": float(np.linalg.norm(a)), "norm_b": float(np.linalg.norm(b)),
            "angle_deg": angle,
            "norms_lerp": [float(np.linalg.norm(tk.lerp(a, b, t))) for t in np.linspace(0, 1, 21)],
            "norms_slerp": [float(np.linalg.norm(tk.slerp(a, b, t))) for t in np.linspace(0, 1, 21)],
        }
        mels = {}
        for method in ["lerp", "slerp"]:
            fn = tk.lerp if method == "lerp" else tk.slerp
            for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
                key = f"{method}_t{int(t*100):03d}"
                if t == 0.0:
                    name = f"interp_{tag}_A"
                elif t == 1.0:
                    name = f"interp_{tag}_B"
                else:
                    name = f"interp_{tag}_{method}_t{int(t*100)}"
                if (AUD / f"{name}.wav").exists():
                    continue
                z = fn(a, b, t)
                style = tk.style_from_z(model, torch.tensor(z, dtype=torch.float32).unsqueeze(0))
                mel, _ = tk.synth_mel_from_style(model, tp, text, style)
                mels[key] = mel.squeeze(0).float().cpu().numpy()
                wav(name, tk.mel_to_audio(mel))
        np.savez(ANA / f"interp_mels_{tag}.npz", **mels)
        del model
        torch.cuda.empty_cache()
    (ANA / "interp_info.json").write_text(json.dumps(interp_info, indent=1))


if __name__ == "__main__":
    print("== LJ style transfer ==", flush=True)
    lj_style_transfer()
    print("== LJ attention + latent stats ==", flush=True)
    lj_attention_and_latent()
    print("== VERBO style transfer + attention ==", flush=True)
    verbo_style_transfer_and_attention()
    print("== latent manipulation ==", flush=True)
    latent_manipulation()
    print("== interpolation ==", flush=True)
    interpolation()
    print("DONE", flush=True)
