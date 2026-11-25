"""b20: embeddings Resemblyzer de saidas (pasta de wavs) ou de referencias LibriTTS de uma lista de locutores.
Uso: python3 -I b20_embed2.py outs <wav_dir> <out.npz>
     python3 -I b20_embed2.py refs <libritts_dir> <out.npz> <ids_ref.txt> [max_por_locutor]"""
import sys, os, glob
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "pylib"))
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
mode, src, out = sys.argv[1:4]; enc = VoiceEncoder("cpu", verbose=False); rng = np.random.default_rng(0)
keys, embs = [], []
def add(k, p):
    try: embs.append(enc.embed_utterance(preprocess_wav(p))); keys.append(k)
    except Exception as ex: print("falhou", p, ex)
if mode == "outs":
    for p in sorted(glob.glob(os.path.join(src, "*", "step_2500", "*", "*.wav"))):
        q = p.split(os.sep); add("out|" + q[-4] + "|" + q[-2] + "|" + q[-1][:-4], p)
else:
    refs = [l.strip() for l in open(sys.argv[4]) if l.strip()]; MAXU = int(sys.argv[5]) if len(sys.argv) > 5 else 30
    byspk = {}
    for p in glob.glob(os.path.join(src, "**", "*.wav"), recursive=True): byspk.setdefault(os.path.basename(p).split("_")[0], []).append(p)
    for r in refs:
        ps = byspk.get(r.split("_")[0], []); rp = [p for p in ps if os.path.basename(p)[:-4] == r]
        if rp: add("ref|" + r.split("_")[0] + "|" + r, rp[0])
    for spk, ps in byspk.items():
        others = [p for p in ps if os.path.basename(p)[:-4] not in refs]
        for p in rng.permutation(others)[:MAXU]: add("oth|" + spk + "|" + os.path.basename(p)[:-4], p)
print(mode, len(keys)); np.savez_compressed(out, keys=np.array(keys), emb=np.array(embs))
