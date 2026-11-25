"""Collapse of the 16 training runs (4 new voices, q per run) onto candidate scaling variables:
 X1 = N (speakers only); X2 = N/(1+K0/K) (parametric regime, K0 free); X3 = N*K/M (fraction of
 non-default data, coverage regime with attractor, law of large numbers); X4 = mixed: N if N<N*, NK/M above.
q ~ a + b ln X (+ c ln^2 X). RMSE compared with seed noise (~0.06 per run)."""
import numpy as np, pandas as pd
from scipy.optimize import least_squares
G=pd.read_csv('dados_grade.csv'); U=G[G.grp=='unseen']; N=U.N.values.astype(float); K=U.K.values.astype(float); q=U.q.values
M=2000.
def fit(X,extra=0):
    def r(p): 
        x=np.log(X(p[3:])) if extra else np.log(X(None)); return p[0]+p[1]*x+p[2]*x**2-q
    p0=[0.5,-0.05,0]+([1.0]*extra); s=least_squares(r,p0); return np.sqrt(np.mean(s.fun**2)),s.x
for nm,X,e in [('N',lambda p:N,0),('N K (total utterances)',lambda p:N*K,0),('N/(1+K0/K)',lambda p:N/(1+np.abs(p[0])/K),1),
               ('N K/M',lambda p:N*K/M,0),('min(N*, ...) mixed: N^(1-w) (NK)^w',lambda p:N**(1-p[0])*(N*K)**p[0],1)]:
    rm,p=fit(X,e); print(f'{nm:38s} rmse={rm:.4f} params={np.round(p,3)}')
m=N>=32
print('only N>=32 (many-neighbour regime):')
for nm,X in [('N',N[m]),('N K',(N*K)[m])]:
    A=np.c_[np.ones(m.sum()),np.log(X)]; b=np.linalg.lstsq(A,q[m],rcond=None)[0]; print(f'  {nm:6s} rmse={np.sqrt(np.mean((A@b-q[m])**2)):.4f}')
