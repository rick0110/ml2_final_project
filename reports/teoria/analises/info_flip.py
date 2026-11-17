"""Prediction: fraction of external speakers whose majority copy outcome flips when another single reference
utterance of the same speaker is used (noise SD 1 in whitened units along the radial LJ direction).
usage: python3 -I info_flip.py OUT"""
import sys, os, json, numpy as np
from scipy.stats import norm
at = json.load(open(os.path.join(sys.argv[1], "attractor.json"))); res = {}
for run in ["n8_k32_s1-wide", "n32_k32_s1-wide", "n128_k32_s1-wide"]:
    a = at[run]; d = np.array([p["d_lj_w"] for p in a["per_speaker"]]); c = np.array([p["copy"] for p in a["per_speaker"]])
    f1 = norm.cdf(-np.abs(d - a["d50_w"]) / np.sqrt(2)).mean()
    # logistic model incl. width: expected |change| of per-speaker copy prob under resampling of the reference
    b1 = 1 / a["width_w"]; rng = np.random.default_rng(0)
    dn = d[:, None] + rng.normal(0, np.sqrt(2), (len(d), 4000))
    p0 = 1 / (1 + np.exp(-b1 * (d - a["d50_w"]))); pn = 1 / (1 + np.exp(-b1 * (dn - a["d50_w"])))
    res[run] = dict(frac_flip=float(f1), mean_abs_change=float(np.abs(pn - p0[:, None]).mean()), near=int((np.abs(d - a["d50_w"]) < 1.5).sum()))
    print(run, res[run])
json.dump(res, open(os.path.join(sys.argv[1], "flip.json"), "w"), indent=1)
