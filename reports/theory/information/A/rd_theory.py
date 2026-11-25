"""Funcoes teoricas (teoria2/A).
- gchisq_cdf_imhof: P(sum_j w_j (Z_j + a_j)^2 <= t)  (Imhof 1961)
- smallball_saddle: mesma probabilidade pela aproximacao de ponto de sela (Lugannani-Rice);
  o expoente de Chernoff e a 'taxa do livro-codigo aleatorio' R_c(t) = -log2 P.
- waterfill: D(R) da agua reversa para fonte gaussiana de espectro lam.
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import norm


def gchisq_cdf_imhof(t, w, a2):
    """w: pesos (variancias), a2: nao-centralidades a_j^2 (em unidades de desvio). t escalar."""
    w = np.asarray(w, float); a2 = np.asarray(a2, float)
    if t <= 0:
        return 0.0

    def f(u):
        wu = w * u
        th = 0.5 * np.sum(np.arctan(wu) + a2 * wu / (1 + wu ** 2)) - 0.5 * t * u
        lr = np.sum(0.25 * np.log1p(wu ** 2) + 0.5 * a2 * wu ** 2 / (1 + wu ** 2))
        return np.sin(th) / (u * np.exp(lr))
    val, err = quad(f, 0, np.inf, limit=800, epsabs=1e-13, epsrel=1e-10)
    return float(min(max(0.5 - val / np.pi, 0.0), 1.0))


def cgf(s, w, a2):
    """K(s) = log E exp(s Q), Q = sum w (Z+a)^2, valido para s < 1/(2 max w)."""
    d = 1 - 2 * s * w
    return np.sum(-0.5 * np.log(d) + a2 * w * s / d)


def cgf1(s, w, a2):
    d = 1 - 2 * s * w
    return np.sum(w / d + a2 * w / d ** 2)


def cgf2(s, w, a2):
    d = 1 - 2 * s * w
    return np.sum(2 * w ** 2 / d ** 2 + 4 * a2 * w ** 2 / d ** 3)


def smallball_saddle(t, w, a2, return_chernoff=False):
    """P(Q <= t) para t abaixo da media (cauda inferior) por Lugannani-Rice.
    Retorna (P, chernoff_nats) onde chernoff = s t - K(s) >= 0 e' o expoente: P <= exp(-chernoff)."""
    w = np.asarray(w, float); a2 = np.asarray(a2, float)
    mean = np.sum(w * (1 + a2))
    if t <= 0:
        return (0.0, np.inf) if return_chernoff else 0.0
    if t >= mean:
        # usar Imhof fora da cauda inferior
        p = gchisq_cdf_imhof(t, w, a2)
        return (p, 0.0) if return_chernoff else p
    # resolver K'(s) = t, s<0
    lo = -1.0
    while cgf1(lo, w, a2) > t:
        lo *= 2
    s = brentq(lambda s: cgf1(s, w, a2) - t, lo, 0.0 - 1e-15)
    ex = s * t - cgf(s, w, a2)
    wh = np.sign(s) * np.sqrt(2 * ex)
    uh = s * np.sqrt(cgf2(s, w, a2))
    p = norm.cdf(wh) + norm.pdf(wh) * (1 / wh - 1 / uh)
    p = float(max(p, 1e-300))
    return (p, ex) if return_chernoff else p


def waterfill(lam, R_bits):
    """D(R) para fonte gaussiana de variancias lam (soma em todas as componentes)."""
    lam = np.sort(np.asarray(lam, float))[::-1]
    lam = lam[lam > 0]

    def rate(th):
        return 0.5 * np.sum(np.log2(np.maximum(lam / th, 1.0)))
    if R_bits <= 0:
        return lam.sum(), lam.max()
    th = brentq(lambda th: rate(th) - R_bits, 1e-300 if R_bits > 1e3 else lam.min() * 1e-30, lam.max())
    return np.sum(np.minimum(lam, th)), th
