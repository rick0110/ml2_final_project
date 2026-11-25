"""Biblioteca comum (teoria2/A): leitura dos latentes e resultados salvos, geometria branqueada.
Uso sempre com python3 -I; caminhos passados como argumento. Somente leitura dos dados."""
import csv
import numpy as np

RUNS_WIDE = {8: 'n8_k32_s1-wide', 32: 'n32_k32_s1-wide', 128: 'n128_k32_s1-wide'}


def load_latents(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in ['mu', 'logvar', 'speaker', 'emotion', 'source', 'utt_id']}


def load_results(path):
    """devolve dict speaker -> dict(copy=fracao copiada, margin=media, n, f0) para refs externas,
    e o mesmo para seen/unseen."""
    rows = list(csv.DictReader(open(path)))
    out = {}
    for r in rows:
        key = (r['reference_emotion'], r['reference_speaker'])
        d = out.setdefault(key, {'c': [], 'm': [], 'f0': float(r['ref_f0_hz'] or 'nan'),
                                 'sref': [], 'sdef': []})
        sr, sd = float(r['sim_ref']), float(r['sim_default'])
        d['c'].append(sr > sd); d['m'].append(sr - sd); d['sref'].append(sr); d['sdef'].append(sd)
    res = {}
    for k, d in out.items():
        res[k] = dict(copy=float(np.mean(d['c'])), margin=float(np.mean(d['m'])), n=len(d['c']),
                      f0=d['f0'], margins=np.array(d['m']), sdef=float(np.mean(d['sdef'])))
    return res


class Geometry:
    """Espaco branqueado pela covariancia dentro do locutor (LibriTTS de treino) de um run."""

    def __init__(self, lat, add_post=True):
        mu, src, spk = lat['mu'].astype(float), lat['source'], lat['speaker']
        tr = src != 'refs'
        lib = tr & (spk != 'lj')
        self.spk_ids = sorted(set(spk[lib]))
        groups = [mu[lib & (spk == s)] for s in self.spk_ids]
        groups = [g for g in groups if len(g) >= 2]
        self.spk_ids = [s for s in self.spk_ids if (lib & (spk == s)).sum() >= 2]
        resid = np.concatenate([g - g.mean(0) for g in groups])
        dfw = sum(len(g) - 1 for g in groups)
        W = resid.T @ resid / dfw
        self.W_raw = W
        self.post_var = np.exp(lat['logvar'][tr]).mean(0)
        Wc = W + (np.diag(self.post_var) if add_post else 0)
        ev, U = np.linalg.eigh(Wc)
        self.T = U @ np.diag(ev ** -0.5) @ U.T  # branqueador simetrico
        self.Wc = Wc
        x = mu @ self.T  # T simetrico
        self.x = x
        self.lat = lat
        self.n_s = np.array([(lib & (spk == s)).sum() for s in self.spk_ids])
        self.means = np.array([x[lib & (spk == s)].mean(0) for s in self.spk_ids])
        ljm = tr & (spk == 'lj')
        self.lj = x[ljm].mean(0)
        self.lj_cov = np.cov(x[ljm].T)
        self.n_lj = ljm.sum()
        # covariancia entre locutores por componentes de variancia (desbalanceado)
        n = self.n_s.astype(float)
        S = len(n)
        gm = (n[:, None] * self.means).sum(0) / n.sum()
        dev = self.means - gm
        MSB = (n[:, None, None] * np.einsum('si,sj->sij', dev, dev)).sum(0) / (S - 1)
        Wx = self.T @ W @ self.T  # dentro do locutor (sem ruido de amostragem do posterior) branqueado
        Wx_all = np.eye(len(gm))  # com ruido do posterior (se add_post)
        n0 = (n.sum() - (n ** 2).sum() / n.sum()) / (S - 1)
        self.Sb = (MSB - (Wx if add_post else Wx_all)) / n0
        self.Sb_raw_means = np.cov(self.means.T)
        self.center = gm
        self.Wx = Wx

    def refs(self, kind='external'):
        lat = self.lat
        m = (lat['source'] == 'refs') & (lat['emotion'] == kind)
        return lat['speaker'][m], self.x[m]
