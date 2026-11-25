"""Library for the Bayesian decoder (voice estimator via a posterior over the training voices).
No side effects. Imported by the b*.py scripts (which insert this directory into sys.path)."""
import json
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

M_LJ = 2000.0          # utterances of the default voice in every coverage training run
A_LINK = 0.75 / 0.036  # ||Delta||^2 / sigma_w : 0.75 = 2(1-0.626) (4 voices at strength 0, 04_dados 6.1); 0.036 = sigma_w (02, 1.2)

def load(path):
    return json.loads(str(np.load(path, allow_pickle=False)["data"]))

class Geo:
    """geometry of a training run in raw coordinates of the latent mu (32-d)."""
    def __init__(self, r, rank=None, proj=None):
        g = r["raw"]
        self.N, self.K = r["N"], r["K"]
        self.W = np.array(g["W"]); self.Wlj = np.array(g["W_lj"])
        self.s2lib = np.array(g["s2_lib"]); self.s2lj = np.array(g["s2_lj"])
        self.means = np.array(g["means"]); self.n = np.array(r["spk_n"], float)
        self.ids = list(r["spk_ids"]); self.mlj = np.array(g["m_lj"]); self.Sb = np.array(g["Sb"])
        self.refs = r["refs"]
        self.C = np.array([x["c_raw"] for x in self.refs])
        self.P = proj  # optional: (k x 32) projection matrix
    def covs(self, noise_scale=1.0, post_scale=1.0):
        nbar = np.mean(self.n)
        Slib = noise_scale * self.W + post_scale * np.diag(self.s2lib) + self.W / nbar  # mean estimated from n_i utterances
        Slj = noise_scale * self.Wlj + post_scale * np.diag(self.s2lj)
        return Slib, Slj

def gauss_ll(X, mus, S):
    """log N(x; mu_j, S) for all x (n,p) and mu_j (m,p) -> (n,m)"""
    L = np.linalg.cholesky(S); ld = 2 * np.log(np.diag(L)).sum()
    Xi = np.linalg.solve(L, X.T).T; Mi = np.linalg.solve(L, mus.T).T
    d2 = (Xi ** 2).sum(1)[:, None] + (Mi ** 2).sum(1)[None, :] - 2 * Xi @ Mi.T
    return -0.5 * d2 - 0.5 * ld - 0.5 * X.shape[1] * np.log(2 * np.pi), d2

def lam(geo, C=None, T=1.0, logM=np.log(M_LJ), K=None, means=None, Slib=None, Slj=None, P=None, return_parts=False):
    """posterior log-ratio (nats): Lambda = log sum_i K N(c;v_i) - log M N(c;v0), likelihood tempered by 1/T.
    copy (posterior-mean estimator) <=> Lambda > 0.  P: optional projection (k x 32) applied to everything."""
    C = geo.C if C is None else C; K = geo.K if K is None else K
    means = geo.means if means is None else means
    if Slib is None or Slj is None: Slib, Slj = geo.covs()
    mlj = geo.mlj
    if P is not None:
        C = C @ P.T; means = means @ P.T; mlj = mlj @ P.T; Slib = P @ Slib @ P.T; Slj = P @ Slj @ P.T
    lli, d2i = gauss_ll(C, means, Slib); ll0, d20 = gauss_ll(C, mlj[None, :], Slj)
    A = logsumexp(np.log(K) + lli / T, axis=1); B = logM + ll0[:, 0] / T
    if return_parts:
        return A - B, dict(cov=A - np.log(K), lik0=ll0[:, 0] / T, d2nn=d2i.min(1), d20=d20[:, 0], lli=lli / T)
    return A - B

def rho_of(L):
    return 1.0 / (1.0 + np.exp(np.clip(L, -50, 50)))

def p_copy(L, link="mean", a=A_LINK):
    r = rho_of(L)
    if link == "mean":   # posterior mean + utterance noise on the margin (Lemma 1)
        return norm.cdf(a * (0.5 - r))
    if link == "samp":   # posterior sampling: each utterance picks one voice
        return 1 - r
    raise ValueError(link)

def obs(geo, grp, key="r1", spk=False):
    """(k, n) per reference of the group"""
    k = []; n = []; idx = []
    for j, x in enumerate(geo.refs):
        if x["grp"] != grp or key not in x: continue
        s = x[key]; kk = s["k_spk"] if spk else s["k"]
        if kk < 0: continue
        k.append(kk); n.append(s["n"]); idx.append(j)
    return np.array(k, float), np.array(n, float), np.array(idx, int)
