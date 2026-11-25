"""Modelo populacional (teoria com entradas medidas): vozes gaussianas no espaco branqueado de um treino de referencia.
Sem efeitos colaterais."""
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm
from bt_lib import Geo, A_LINK

class Pop:
    def __init__(self, r, reg=0.05):
        g = Geo(r); self.g = g
        Wreg = g.W + reg * np.trace(g.W) / 32 * np.eye(32)
        L = np.linalg.cholesky(Wreg); Li = np.linalg.inv(L); self.Li = Li
        w = lambda S: Li @ S @ Li.T
        self.I = w(g.W)                                     # ~ identidade (ruido de frase, dentro do locutor)
        self.Plib = w(np.diag(g.s2lib)); self.Plj = w(np.diag(g.s2lj)); self.Wlj = w(g.Wlj)
        Sb = w(g.Sb); ev, V = np.linalg.eigh(Sb); ev = np.clip(ev, 0, None)
        self.Sb = (V * ev) @ V.T; self.Sb_half = V * np.sqrt(ev)
        self.mpop = (g.means @ Li.T).mean(0); self.v0 = Li @ g.mlj
        self.C_ext = g.C @ Li.T                               # referencias reais (externas etc.) branqueadas
        self.grp = [x["grp"] for x in g.refs]
    def draw(self, n, rng):
        return self.mpop + rng.standard_normal((n, 32)) @ self.Sb_half.T
    def noise(self, n, rng, R=1):
        Lc = np.linalg.cholesky(self.I + 1e-9 * np.eye(32))
        return rng.standard_normal((n, 32)) @ Lc.T / np.sqrt(R)

def _ll(X, mus, S):
    L = np.linalg.cholesky(S); ld = 2 * np.log(np.diag(L)).sum()
    Xi = np.linalg.solve(L, X.T).T; Mi = np.linalg.solve(L, mus.T).T
    d2 = (Xi ** 2).sum(1)[:, None] + (Mi ** 2).sum(1)[None, :] - 2 * Xi @ Mi.T
    return -0.5 * d2 - 0.5 * ld

def Lam(p, C, V, K, s2, logM, P=None, scale_c=1.0, post=True):
    """C: codigos (n,32) ; V: vozes treinadas (N,32). P: projecao (k,32) opcional."""
    Slib = p.I * (1 + 1.0 / K) + s2 * np.eye(32) + (p.Plib if post else 0)
    Slj = p.Wlj + s2 * np.eye(32) + (p.Plj if post else 0)
    v0 = p.v0
    if P is not None:
        C = C @ P.T; V = V @ P.T; v0 = P @ v0; Slib = P @ Slib @ P.T; Slj = P @ Slj @ P.T
    A = logsumexp(np.log(K) + _ll(C, V, Slib), axis=1)
    B = logM + _ll(C, v0[None, :], Slj)[:, 0]
    return A - B

def pc(L, link="mean", a=A_LINK):
    r = 1 / (1 + np.exp(np.clip(L, -50, 50)))
    return norm.cdf(a * (0.5 - r)) if link == "mean" else 1 - r
