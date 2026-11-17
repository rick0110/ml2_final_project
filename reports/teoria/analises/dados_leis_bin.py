"""Leis para a fracao NAO copiada q(N,K) e versao mediada pela margem. Cada lei: fit(N,K,q,mu) -> funcao pred(N,K).
mix = (u_s, sig) define q = media_s Phi(-(mu + u_s)/sig) (ligacao probit da margem com locutores fixos)."""
import numpy as np
from scipy.optimize import nnls, least_squares
from scipy.stats import norm

def _lin(X, y, nonneg):
    return nnls(X, y)[0] if nonneg else np.linalg.lstsq(X, y, rcond=None)[0]

def law_lin(cols, nonneg=True):
    def fit(N, K, q, mu, mix):
        b = _lin(cols(N, K), q, nonneg)
        return lambda N2, K2: cols(np.atleast_1d(N2).astype(float), np.atleast_1d(K2).astype(float)) @ b
    return fit

def law_pow(withK=True):
    A = np.arange(0.01, 1.5001, 0.01); B = np.arange(-0.5, 1.5001, 0.02) if withK else np.array([0.0])
    lN = None
    def fit(N, K, q, mu, mix):
        best = (np.inf, None)
        lq = np.log(np.clip(q, 1e-3, None))
        for b in B:
            x = N[None, :] ** -A[:, None] * K[None, :] ** -b
            c = (x * q).sum(1) / (x * x).sum(1)
            sse = ((c[:, None] * x - q) ** 2).sum(1); i = int(np.argmin(sse))
            if sse[i] < best[0]: best = (sse[i], (c[i], A[i], b))
        c, a, b = best[1]
        f = lambda N2, K2: c * np.atleast_1d(N2).astype(float) ** -a * np.atleast_1d(K2).astype(float) ** -b
        f.par = (c, a, b); return f
    return fit

def law_probit(cols):
    def fit(N, K, q, mu, mix):
        X = cols(N, K); b0 = np.linalg.lstsq(X, norm.ppf(np.clip(q, .02, .98)), rcond=None)[0]
        b = least_squares(lambda b: norm.cdf(X @ b) - q, b0).x
        return lambda N2, K2: norm.cdf(cols(np.atleast_1d(N2).astype(float), np.atleast_1d(K2).astype(float)) @ b)
    return fit

def law_margin(cols, alpha_grid=None):
    """ajusta lei linear (ou com expoente em grade) para mu, converte para q pela mistura probit."""
    def fit(N, K, q, mu, mix):
        u, sig = mix
        if alpha_grid is None:
            b = np.linalg.lstsq(cols(N, K), mu, rcond=None)[0]; g = lambda N2, K2: cols(N2, K2) @ b
        else:
            best = (np.inf, None)
            for a in alpha_grid:
                X = cols(N, K, a); b = np.linalg.lstsq(X, mu, rcond=None)[0]; s = ((X @ b - mu) ** 2).sum()
                if s < best[0]: best = (s, (a, b))
            a, b = best[1]; g = lambda N2, K2: cols(N2, K2, a) @ b
        def pred(N2, K2):
            m = g(np.atleast_1d(N2).astype(float), np.atleast_1d(K2).astype(float))
            return norm.cdf(-(m[:, None] + u[None, :]) / sig).mean(1)
        return pred
    return fit

one = lambda N: np.ones_like(N, dtype=float)
LAWS_NK = {
    "L1: a + A/N + B/(NK)": (law_lin(lambda N, K: np.c_[one(N), 1 / N, 1 / (N * K)]), 3),
    "L2: a + A/sqrtN + C/sqrtK": (law_lin(lambda N, K: np.c_[one(N), N ** -.5, K ** -.5]), 3),
    "L3: c N^-alpha K^-beta": (law_pow(True), 3),
    "L4: Phi(a + b lnN + c lnK) direto": (law_probit(lambda N, K: np.c_[one(N), np.log(N), np.log(K)]), 3),
    "L4m: margem a+b lnN+c lnK -> probit": (law_margin(lambda N, K: np.c_[one(N), np.log(N), np.log(K)]), 3),
    "L4m': margem a+b/sqrtN+c/sqrtK -> probit": (law_margin(lambda N, K: np.c_[one(N), N ** -.5, K ** -.5]), 3),
}
LAWS_N = {
    "L1n: a + A/N": (law_lin(lambda N, K: np.c_[one(N), 1 / N]), 2),
    "L2n: a + A/sqrtN": (law_lin(lambda N, K: np.c_[one(N), N ** -.5]), 2),
    "L3n: c N^-alpha": (law_pow(False), 2),
    "L4n: Phi(a + b lnN) direto": (law_probit(lambda N, K: np.c_[one(N), np.log(N)]), 2),
    "L4mn: margem a+b lnN -> probit": (law_margin(lambda N, K: np.c_[one(N), np.log(N)]), 2),
    "L4mn': margem a+b/sqrtN -> probit": (law_margin(lambda N, K: np.c_[one(N), N ** -.5]), 2),
}

def mixture(Ym):
    """Ym: treinos x locutores x frases. u_s = efeito fixo do locutor; sig^2 = var residual dentro do treino-locutor + var da interacao treino x locutor."""
    run_m = Ym.mean((1, 2)); sm = Ym.mean(2)                        # treino x locutor
    u = (sm - run_m[:, None]).mean(0)
    inter = sm - run_m[:, None] - u[None, :]
    a, b = sm.shape
    s2_rs = (inter ** 2).sum() / ((a - 1) * (b - 1)) if a > 1 else 0.0
    s2_w = Ym.var(2, ddof=1).mean()
    return u, np.sqrt(s2_w + max(s2_rs - s2_w / Ym.shape[2], 0))

def loco(fit, N, K, q, mu, mix, cfg):
    pred = np.empty_like(q)
    for c in np.unique(cfg):
        tr = cfg != c
        pred[~tr] = fit(N[tr], K[tr], q[tr], mu[tr], mix)(N[~tr], K[~tr])
    return pred

def evaluate(laws, N, K, q, mu, mix, cfg):
    out = {}
    n = len(q)
    for nm, (fit, k) in laws.items():
        f = fit(N, K, q, mu, mix); r = f(N, K) - q; sse = (r ** 2).sum(); kk = k + 1
        e = loco(fit, N, K, q, mu, mix, cfg) - q
        out[nm] = dict(k=k, rmse=np.sqrt(sse / n), aicc=n * np.log(sse / n) + 2 * kk + 2 * kk * (kk + 1) / max(n - kk - 1, 1e-9),
                       bic=n * np.log(sse / n) + kk * np.log(n), cv=np.sqrt(np.mean(e ** 2)), f=f)
    return out
