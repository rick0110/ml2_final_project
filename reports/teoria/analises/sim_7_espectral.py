"""Regime (c) espectral: vozes v in R^D com covariancia diag(lambda_j), lambda_j = j^-alpha.
Decodificador linear W aprendido por ridge de pares (codigo z_i = v_i, saida y_i = v_i + ruido s2_eff);
alvo W = I. Erro numa voz nova: E|(W_hat - I) v*|^2 = sum_j lambda_j (1-W_jj)^2 + variancia.
Previsoes: ridge otimo com ruido ~ N^-(alpha-1)/alpha ; sem ruido (ridgeless, N<D) ~ N^-(alpha-1)."""
import numpy as np
rng=np.random.default_rng(3); D=1500
def err(alpha,N,s2,lam,reps=4):
    l=np.arange(1,D+1)**(-alpha); e=[]
    for _ in range(reps):
        Z=rng.normal(size=(N,D))*np.sqrt(l); Y=Z+rng.normal(scale=np.sqrt(s2),size=Z.shape)
        if N<D:  # forma dual
            Kz=Z@Z.T; A=np.linalg.solve(Kz+lam*np.eye(N),Y); Wt=Z.T@A   # W^T (D x D) = Z^T (ZZ^T+lam)^-1 Y
        else:
            Wt=np.linalg.solve(Z.T@Z+lam*np.eye(D),Z.T@Y)
        Dm=Wt-np.eye(D); e.append(np.sum(l[:,None]*Dm**2))   # E|(W-I)^T... v*|^2 com v* ~ N(0,diag l)
    return np.mean(e)
Ns=np.array([32,64,128,256,512,1024])
for alpha in (1.5,2.0,3.0):
    for s2 in (0.0,0.05):
        lams=np.geomspace(1e-6,10,15) if s2>0 else [1e-8]
        E=np.array([min(err(alpha,N,s2,la,reps=2) for la in lams) for N in Ns])
        sl=np.polyfit(np.log(Ns),np.log(E),1)[0]
        pred=(alpha-1)/alpha if s2>0 else alpha-1
        print(f'alpha={alpha} s2={s2}: inclinacao={sl:.3f}  previsto={-pred:.3f}  E={np.round(E,4)}')
