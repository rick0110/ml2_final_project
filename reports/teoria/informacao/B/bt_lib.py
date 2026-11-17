"""Biblioteca do decodificador bayesiano (estimador de voz por posterior sobre as vozes de treino).
Sem efeitos colaterais. Importada pelos scripts b*.py (que inserem este diretorio em sys.path)."""
import json
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

M_LJ = 2000.0          # frases da voz padrao em todo treino de cobertura
A_LINK = 0.75 / 0.036  # ||Delta||^2 / sigma_w : 0.75 = 2(1-0.626) (4 vozes com forca 0, 04_dados 6.1); 0.036 = sigma_w (02, 1.2)

def load(path):
    return json.loads(str(np.load(path, allow_pickle=False)["data"]))

class Geo:
    """geometria de um treino em coordenadas cruas do latente mu (32-d)."""
    def __init__(self, r, rank=None, proj=None):
        g = r["raw"]
        self.N, self.K = r["N"], r["K"]
        self.W = np.array(g["W"]); self.Wlj = np.array(g["W_lj"])
        self.s2lib = np.array(g["s2_lib"]); self.s2lj = np.array(g["s2_lj"])
        self.means = np.array(g["means"]); self.n = np.array(r["spk_n"], float)
        self.ids = list(r["spk_ids"]); self.mlj = np.array(g["m_lj"]); self.Sb = np.array(g["Sb"])
        self.refs = r["refs"]
        self.C = np.array([x["c_raw"] for x in self.refs])
        self.P = proj  # opcional: matriz (k x 32) de projecao
    def covs(self, noise_scale=1.0, post_scale=1.0):
        nbar = np.mean(self.n)
        Slib = noise_scale * self.W + post_scale * np.diag(self.s2lib) + self.W / nbar  # media estimada com n_i frases
        Slj = noise_scale * self.Wlj + post_scale * np.diag(self.s2lj)
        return Slib, Slj

def gauss_ll(X, mus, S):
    """log N(x; mu_j, S) para todos x (n,p) e mu_j (m,p) -> (n,m)"""
    L = np.linalg.cholesky(S); ld = 2 * np.log(np.diag(L)).sum()
    Xi = np.linalg.solve(L, X.T).T; Mi = np.linalg.solve(L, mus.T).T
    d2 = (Xi ** 2).sum(1)[:, None] + (Mi ** 2).sum(1)[None, :] - 2 * Xi @ Mi.T
    return -0.5 * d2 - 0.5 * ld - 0.5 * X.shape[1] * np.log(2 * np.pi), d2

def lam(geo, C=None, T=1.0, logM=np.log(M_LJ), K=None, means=None, Slib=None, Slj=None, P=None, return_parts=False):
    """log-razao de posterior (nats): Lambda = log sum_i K N(c;v_i) - log M N(c;v0), verossimilhanca temperada por 1/T.
    copia (estimador de media posterior) <=> Lambda > 0.  P: projecao opcional (k x 32) aplicada a tudo."""
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
    if link == "mean":   # media posterior + ruido de frase na margem (Lema 1)
        return norm.cdf(a * (0.5 - r))
    if link == "samp":   # amostragem da posterior: cada frase escolhe uma voz
        return 1 - r
    raise ValueError(link)

def obs(geo, grp, key="r1", spk=False):
    """(k, n) por referencia do grupo"""
    k = []; n = []; idx = []
    for j, x in enumerate(geo.refs):
        if x["grp"] != grp or key not in x: continue
        s = x[key]; kk = s["k_spk"] if spk else s["k"]
        if kk < 0: continue
        k.append(kk); n.append(s["n"]); idx.append(j)
    return np.array(k, float), np.array(n, float), np.array(idx, int)
