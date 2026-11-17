"""Tarefa 8 (complemento: mais sementes e grade densa em N; so cenario conservador).
Tarefa 8: poder por simulacao com componentes de variancia medidos.
Modelo por amostra (escala da margem): m = mu(N) + u_s + v_rs + w_rt + x_st + z_t + e + semente_r ; copiado = m > 0.
Componentes (DP): locutor, locutor x frase, frase, residuo e treino x frase da ANOVA de 3 vias das 38 externas (out/02.txt);
treino x locutor: 0.046 (conservador; inclui interacao locutor x N real) ou 0.0148 (so semente); efeito principal de semente 0.0127.
mu(N) calibrado para que a fracao nao copiada da populacao siga cada hipotese:
  H026: q = 0.744 N^-0.26      H05: q = a + A N^-0.5      Hpiso: q = a + A/N      (as tres ajustadas aos 3 pontos medidos nas 38 vozes)
Analise de cada experimento simulado: q_hat por N (media das sementes), ajuste das 3 formas de 2 parametros (c N^-alpha com alpha livre;
a + A/sqrtN; a + A/N), escolha por menor SSE; IC de alpha por bootstrap nao e feito: reporta-se a dispersao de alpha_hat entre simulacoes.
Teste de piso: diferenca q(512) - q(2048) com EP por bootstrap de locutores; 'ganho detectado' se IC95 exclui 0.
Uso: python3 -I dados_08_poder.py <out_dir> [nsim]"""
import sys, os, itertools
import numpy as np, pandas as pd
from scipy.stats import norm
out = sys.argv[1]; NSIM = int(sys.argv[2]) if len(sys.argv) > 2 else 300
rng = np.random.default_rng(88)
SD = dict(spk=0.0764, spk_sent=0.0143, sent=0.0092, run_sent=0.0062, res=0.0314, seed=0.0127)
NS = np.array([8, 32, 128, 512, 2048.])
qobs = np.array([1 - 0.5663, 1 - 0.6979, 1 - 0.7895]); Nobs = np.array([8, 32, 128.])
def fitlin(cols, N, q): X = np.c_[np.ones_like(N), cols(N)]; return np.linalg.lstsq(X, q, rcond=None)[0]
b05 = fitlin(lambda N: N ** -.5, Nobs, qobs); b1 = fitlin(lambda N: 1 / N, Nobs, qobs)
H = {"H026: 0.744 N^-0.26": lambda N: 0.744 * N ** -0.26,
     f"H05: {b05[0]:.3f} + {b05[1]:.3f}/sqrtN": lambda N: b05[0] + b05[1] * N ** -.5,
     f"Hpiso: {b1[0]:.3f} + {b1[1]:.3f}/N": lambda N: b1[0] + b1[1] / N}
print("q verdadeiro (fracao NAO copiada) por hipotese em N =", NS.astype(int))
for k, f in H.items(): print(f"  {k:28s}", np.round(f(NS), 3))
def fits(N, q):
    lq = np.log(np.clip(q, 1e-3, None)); bA = np.polyfit(np.log(N), lq, 1); sA = ((np.exp(np.polyval(bA, np.log(N))) - q) ** 2).sum()
    r = [sA]
    for cols in [lambda N: N ** -.5, lambda N: 1 / N]:
        X = np.c_[np.ones_like(N), cols(N)]; b = np.linalg.lstsq(X, q, rcond=None)[0]; r.append(((X @ b - q) ** 2).sum())
    return np.array(r), -bA[0]
def simulate(hyp, S, T, seeds, sd_rs, nsim, NS=NS):
    q_true = H[hyp](NS)
    tot = np.sqrt(sum(v ** 2 for v in SD.values()) + sd_rs ** 2)
    mu = -tot * norm.ppf(q_true)
    pick = np.zeros(3); alphas = []; floor_det = 0; pick_mix = np.zeros(3)
    for _ in range(nsim):
        u = rng.normal(0, SD["spk"], S); z = rng.normal(0, SD["sent"], T); x = rng.normal(0, SD["spk_sent"], (S, T))
        qh = np.zeros((seeds, len(NS))); qm = np.zeros_like(qh); spkq = np.zeros((seeds, len(NS), S))
        for j in range(seeds):
            for i in range(len(NS)):
                m = (mu[i] + rng.normal(0, SD["seed"]) + u[:, None] + rng.normal(0, sd_rs, S)[:, None] + rng.normal(0, SD["run_sent"], T)[None, :]
                     + z[None, :] + x + rng.normal(0, SD["res"], (S, T)))
                c = m < 0; qh[j, i] = c.mean(); spkq[j, i] = c.mean(1)
                ms = m.mean(1); sw = np.sqrt(m.var(1, ddof=1).mean()); qm[j, i] = norm.cdf(-ms / sw).mean()
        qbar = qh.mean(0); sse, a = fits(NS, qbar); pick[np.argmin(sse)] += 1; alphas.append(a)
        sse2, _ = fits(NS, qm.mean(0)); pick_mix[np.argmin(sse2)] += 1
        # teste de ganho entre 512 e 2048 (pareado por locutor, media das sementes)
        if 512 in NS and 2048 in NS:
            i5, i2 = list(NS).index(512), list(NS).index(2048)
            dq = spkq[:, i5].mean(0) - spkq[:, i2].mean(0)
            v_spk = np.var([dq[rng.integers(0, S, S)].mean() for _ in range(200)])
            slope = norm.pdf(norm.ppf(np.clip(qbar[[i5, i2]], .01, .99))) / tot      # dq/dmu
            v_seed = ((SD["seed"] * slope) ** 2).sum() / seeds                  # variancia de semente (suposta conhecida do piloto)
            floor_det += (dq.mean() - 1.96 * np.sqrt(v_spk + v_seed)) > 0
    a = np.array(alphas)
    return dict(P_escolhe=pick / nsim, P_escolhe_mix=pick_mix / nsim, alpha_med=np.median(a), alpha_lo=np.percentile(a, 2.5), alpha_hi=np.percentile(a, 97.5),
                P_ganho_512_2048=floor_det / nsim)
rows = []
DENSE = np.array([8, 16, 32, 64, 128, 256, 512, 1024, 2048.])
designs = [(38, 25, 5), (100, 25, 5), ("densa", 100, 25, 1), ("densa", 100, 25, 2), ("densa", 38, 25, 2), ("densa", 100, 25, 3)]
_unused = [(4, 25, 1), (4, 25, 3), (38, 25, 1), (38, 25, 2), (38, 25, 3), (100, 10, 1), (100, 25, 1), (100, 25, 2), (100, 25, 3), (200, 10, 2), (200, 25, 2), (200, 25, 3)]
hn = list(H)
for sd_rs, lab in [(0.046, "conservador")]:
    for des in designs:
        ns = NS
        if des[0] == "atual": ns = np.array([8, 32, 128.]); des = des[1:]
        if des[0] == "densa": ns = DENSE; des = des[1:]
        S, T, seeds = des
        for k, hyp in enumerate(hn):
            r = simulate(hyp, S, T, seeds, sd_rs, NSIM if S <= 100 else NSIM // 2, ns)
            rows.append(dict(cenario=lab, S=S, T=T, sementes=seeds, verdade=hyp.split(":")[0], P_correta=r["P_escolhe"][k], P_correta_mix=r["P_escolhe_mix"][k],
                             escolhas="/".join(f"{p:.2f}" for p in r["P_escolhe"]), alpha_med=r["alpha_med"], alpha_lo=r["alpha_lo"], alpha_hi=r["alpha_hi"],
                             P_ganho_512_2048=r["P_ganho_512_2048"], N_grade="/".join(str(int(v)) for v in ns), treinos=len(ns) * seeds, amostras_aval=len(ns) * seeds * S * T))
        print(pd.DataFrame(rows[-3:]).round(3).to_string(header=(len(rows) == 3)), flush=True)
P = pd.DataFrame(rows); P.to_csv(os.path.join(out, "poder_extra.csv"), index=False)
pd.set_option("display.width", 250); print(P.round(3).to_string())
