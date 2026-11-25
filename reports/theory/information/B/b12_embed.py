"""b12: Resemblyzer embeddings (the project's verifier) of the saved outputs and of the LibriTTS test-clean references.
Usage: python3 -I b12_embed.py <wavs_ext> <libritts_tc> <ref_ids.txt> <out.npz> [max_per_speaker]"""
import sys, os, glob, re
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "pylib"))
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
wdir, ldir, idf, out = sys.argv[1:5]; MAXU = int(sys.argv[5]) if len(sys.argv) > 5 else 30
enc = VoiceEncoder("cpu", verbose=False); rng = np.random.default_rng(0)
keys, embs = [], []
def add(key, path):
    try:
        e = enc.embed_utterance(preprocess_wav(path))
    except Exception as ex:
        print("failed", path, ex); return
    keys.append(key); embs.append(e)
outs = sorted(glob.glob(os.path.join(wdir, "*", "step_2500", "*", "*.wav")))
for p in outs:
    parts = p.split(os.sep); add("out|" + parts[-4] + "|" + parts[-2] + "|" + parts[-1][:-4], p)
print("outputs:", len(outs), flush=True)
refs = [l.strip() for l in open(idf) if l.strip()]
allw = glob.glob(os.path.join(ldir, "**", "*.wav"), recursive=True)
byspk = {}
for p in allw: byspk.setdefault(os.path.basename(p).split("_")[0], []).append(p)
for r in refs:
    spk = r.split("_")[0]
    if spk not in byspk: continue
    ps = byspk[spk]; rp = [p for p in ps if os.path.basename(p)[:-4] == r]
    if rp: add("ref|" + spk + "|" + r, rp[0])
    others = [p for p in ps if os.path.basename(p)[:-4] != r]
    for p in rng.permutation(others)[:MAXU]: add("oth|" + spk + "|" + os.path.basename(p)[:-4], p)
print("total", len(keys), flush=True)
np.savez_compressed(out, keys=np.array(keys), emb=np.array(embs))
