"""b2: fits the kernel/posterior decoder only on the TRAINED (seen) voices of the grid and predicts the new and external ones.
Families: class covariance = W_c + Sigma_post,c + s2 * W_reg  (s2 = isotropic smoothing in the whitened space),
prior weight: K per LibriTTS voice, M_eff = exp(logM) for LJ.  Links: 'mean' (posterior mean + Lemma 1) and 'samp'.
Usage: python3 -I b2_ajuste.py out/b0.npz out/"""
import sys, os, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr
from bt_lib import *
S = load(sys.argv[1])
BASE = sorted([r for r in S if S[r]["var"] in ("", "wide") and S[r]["refs"]], key=lambda r: (S[r]["N"], S[r]["K"], r))
GEO = {r: Geo(S[r]) for r in BASE}

def covs(g, s2, post=True):
    Wreg = g.W + 0.05 * np.trace(g.W) / 32 * np.eye(32)
    Slib = g.W + g.W / np.mean(g.n) + s2 * Wreg + (np.diag(g.s2lib) if post else 0)
    Slj = g.Wlj + s2 * Wreg + (np.diag(g.s2lj) if post else 0)
    return Slib, Slj

def lam_run(g, s2, logM, post=True, K=None, C=None, means=None):
    Slib, Slj = covs(g, s2, post)
    return lam(g, C=C, Slib=Slib, Slj=Slj, logM=logM, K=K, means=means)

def nll(theta, grp, link, post, runs):
    s2, logM = np.exp(theta[0]), theta[1]
    tot = 0.0
    for r in runs:
        g = GEO[r]; k, n, idx = obs(g, grp)
        if len(k) == 0: continue
        p = np.clip(p_copy(lam_run(g, s2, logM, post)[idx], link), 1e-4, 1 - 1e-4)
        tot -= (k * np.log(p) + (n - k) * np.log(1 - p)).sum()
    return tot

