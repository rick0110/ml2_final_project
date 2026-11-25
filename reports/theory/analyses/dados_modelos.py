"""Motor de ajuste: leis lineares, separaveis (grade nos expoentes + minimos quadrados) e probit.
Todas ajustadas por minimos quadrados nao ponderados no nivel do treino (como fit_law.py)."""
import numpy as np
from scipy.optimize import nnls, least_squares
from scipy.stats import norm

ALPHAS = np.round(np.arange(0.02, 2.0001, 0.02), 4)
BETAS = np.round(np.arange(-0.5, 1.5001, 0.05), 4)

class Model:
    def __init__(self, name, k, kind, basis=None, grid=None, nonneg=False):
        self.name, self.k, self.kind, self.basis, self.grid, self.nonneg = name, k, kind, basis, grid, nonneg
        self._cache = {}
    def _X(self, N, K):
        key = (N.tobytes(), K.tobytes())
        if key not in self._cache:
            if self.kind == "lin":
                self._cache[key] = self.basis(N, K)
            elif self.kind == "sep":
                self._cache[key] = np.stack([self.basis(N, K, th) for th in self.grid])
            else:
                self._cache[key] = None
        return self._cache[key]
    def fit(self, N, K, y):
        N = np.asarray(N, float); K = np.asarray(K, float)
        if self.kind == "lin":
            X = self._X(N, K)
            if self.nonneg:
                b, _ = nnls(X, y)
            else:
                b = np.linalg.lstsq(X, y, rcond=None)[0]
            return ("lin", b)
        if self.kind == "sep":
            X = self._X(N, K)
            A = np.einsum("gni,gnj->gij", X, X) + 1e-12 * np.eye(X.shape[2])
            r = np.einsum("gni,n->gi", X, y)
            b = np.linalg.solve(A, r[..., None])[..., 0]
            sse = ((np.einsum("gni,gi->gn", X, b) - y) ** 2).sum(1)
            g = int(np.argmin(sse))
            return ("sep", b[g], self.grid[g])
        if self.kind == "probit":   # q = Phi(-(a + b lnN + c lnK)) (ou sem K)
            X = self.basis(N, K)
            yc = np.clip(y, 0.02, 0.98)
            b0 = np.linalg.lstsq(X, -norm.ppf(yc), rcond=None)[0]
            f = least_squares(lambda b: norm.cdf(-(X @ b)) - y, b0)
            return ("probit", f.x)
    def predict(self, par, N, K):
        N = np.atleast_1d(np.asarray(N, float)); K = np.atleast_1d(np.asarray(K, float))
        if par[0] == "lin":
            return self.basis(N, K) @ par[1]
        if par[0] == "sep":
            return self.basis(N, K, par[2]) @ par[1]
        return norm.cdf(-(self.basis(N, K) @ par[1]))
    def params(self, par):
        if par[0] == "sep":
            return np.concatenate([np.atleast_1d(par[1]), np.atleast_1d(par[2])])
        return np.asarray(par[1])

one = lambda N: np.ones_like(N)
def margin_models():
    return [
        Model("a + b/N", 2, "lin", lambda N, K: np.c_[one(N), 1 / N]),
        Model("a + b/N + c/(NK)  [L1]", 3, "lin", lambda N, K: np.c_[one(N), 1 / N, 1 / (N * K)]),
        Model("a + b/sqrt(N)", 2, "lin", lambda N, K: np.c_[one(N), N ** -.5]),
        Model("a + b/sqrt(N) + c/sqrt(K)  [L2]", 3, "lin", lambda N, K: np.c_[one(N), N ** -.5, K ** -.5]),
        Model("a + b ln N", 2, "lin", lambda N, K: np.c_[one(N), np.log(N)]),
        Model("a + b ln N + c ln K", 3, "lin", lambda N, K: np.c_[one(N), np.log(N), np.log(K)]),
        Model("a + b N^-alpha", 3, "sep", lambda N, K, th: np.c_[one(N), N ** -th], ALPHAS),
        Model("a + b N^-alpha + c ln K", 4, "sep", lambda N, K, th: np.c_[one(N), N ** -th, np.log(K)], ALPHAS),
        Model("a + c ln K (so K)", 2, "lin", lambda N, K: np.c_[one(N), np.log(K)]),
        Model("constante", 1, "lin", lambda N, K: np.c_[one(N)]),
    ]
AB = np.array([(a, b) for a in ALPHAS[ALPHAS <= 1.5] for b in BETAS])
def frac_models(with_K=True):
    """Leis para a fracao NAO copiada q."""
    ms = [
        Model("L1n: a + A/N", 2, "lin", lambda N, K: np.c_[one(N), 1 / N], nonneg=True),
        Model("L2n: a + A/sqrt(N)", 2, "lin", lambda N, K: np.c_[one(N), N ** -.5], nonneg=True),
        Model("L3n: c N^-alpha", 2, "sep", lambda N, K, th: np.c_[N ** -th], ALPHAS),
        Model("L4n: Phi(-(a + b ln N))", 2, "probit", lambda N, K: np.c_[one(N), np.log(N)]),
    ]
    if with_K:
        ms += [
            Model("L1: a + A/N + B/(NK)", 3, "lin", lambda N, K: np.c_[one(N), 1 / N, 1 / (N * K)], nonneg=True),
            Model("L2: a + A/sqrt(N) + C/sqrt(K)", 3, "lin", lambda N, K: np.c_[one(N), N ** -.5, K ** -.5], nonneg=True),
            Model("L3: c N^-alpha K^-beta", 3, "sep", lambda N, K, th: np.c_[N ** -th[0] * K ** -th[1]], AB),
            Model("L4: Phi(-(a + b ln N + c ln K))", 3, "probit", lambda N, K: np.c_[one(N), np.log(N), np.log(K)]),
            Model("constante", 1, "lin", lambda N, K: np.c_[one(N)]),
        ]
    return ms
def seen_models():
    return [
        Model("a + C/K + A/N", 3, "lin", lambda N, K: np.c_[one(N), 1 / K, 1 / N], nonneg=True),
        Model("a + C/K", 2, "lin", lambda N, K: np.c_[one(N), 1 / K], nonneg=True),
        Model("a + C/sqrt(K)", 2, "lin", lambda N, K: np.c_[one(N), K ** -.5], nonneg=True),
        Model("a + C/sqrt(K) + A/sqrt(N)", 3, "lin", lambda N, K: np.c_[one(N), K ** -.5, N ** -.5], nonneg=True),
        Model("c K^-beta N^-alpha", 3, "sep", lambda N, K, th: np.c_[N ** -th[0] * K ** -th[1]],
              np.array([(a, b) for a in np.arange(-0.5, 1.001, 0.02) for b in np.arange(0, 2.001, 0.02)])),
        Model("Phi(-(a + b ln N + c ln K))", 3, "probit", lambda N, K: np.c_[one(N), np.log(N), np.log(K)]),
        Model("Phi(-(a + c ln K))", 2, "probit", lambda N, K: np.c_[one(N), np.log(K)]),
    ]

def config_ids(N, K):
    keys = [f"{int(n)}x{int(k)}" for n, k in zip(N, K)]
    u = sorted(set(keys), key=keys.index)
    return np.array([u.index(x) for x in keys]), u

def loco(model, N, K, y, cfg):
    """Leave-one-configuration-out: devolve predicoes fora da amostra por treino."""
    pred = np.empty_like(y)
    for c in np.unique(cfg):
        tr = cfg != c
        par = model.fit(N[tr], K[tr], y[tr])
        pred[~tr] = model.predict(par, N[~tr], K[~tr])
    return pred

def summary(model, N, K, y, cfg):
    n = len(y)
    par = model.fit(N, K, y)
    sse = float(((model.predict(par, N, K) - y) ** 2).sum())
    k = model.k + 1
    pred = loco(model, N, K, y, cfg)
    e = pred - y
    # erro CV por configuracao (media das sementes) e por treino
    cv_run = float(np.sqrt(np.mean(e ** 2)))
    cv_cfg = float(np.sqrt(np.mean([np.mean(e[cfg == c]) ** 2 for c in np.unique(cfg)])))
    return dict(model=model.name, k=model.k, rmse=np.sqrt(sse / n), aic=n * np.log(sse / n) + 2 * k,
                aicc=n * np.log(sse / n) + 2 * k + 2 * k * (k + 1) / max(n - k - 1, 1e-9),
                bic=n * np.log(sse / n) + k * np.log(n), cv_rmse_run=cv_run, cv_rmse_cfg=cv_cfg, par=model.params(par))
