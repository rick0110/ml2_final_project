"""Colapso dos 16 treinos (4 vozes novas, q por treino) em variaveis de escala candidatas:
 X1 = N (so locutores); X2 = N/(1+K0/K) (regime parametrico, K0 livre); X3 = N*K/M (fracao de dados
 nao-padrao, regime de cobertura com atrator, lei dos grandes numeros); X4 = misto: N se N<N*, NK/M acima.
q ~ a + b ln X (+ c ln^2 X). RMSE comparado ao ruido de semente (~0.06 por treino)."""
import numpy as np, pandas as pd
from scipy.optimize import least_squares
G=pd.read_csv('dados_grade.csv'); U=G[G.grp=='unseen']; N=U.N.values.astype(float); K=U.K.values.astype(float); q=U.q.values
M=2000.
def fit(X,extra=0):
    def r(p): 
        x=np.log(X(p[3:])) if extra else np.log(X(None)); return p[0]+p[1]*x+p[2]*x**2-q
    p0=[0.5,-0.05,0]+([1.0]*extra); s=least_squares(r,p0); return np.sqrt(np.mean(s.fun**2)),s.x
for nm,X,e in [('N',lambda p:N,0),('N K (total de frases)',lambda p:N*K,0),('N/(1+K0/K)',lambda p:N/(1+np.abs(p[0])/K),1),
               ('N K/M',lambda p:N*K/M,0),('min(N*, ...) misto: N^(1-w) (NK)^w',lambda p:N**(1-p[0])*(N*K)**p[0],1)]:
    rm,p=fit(X,e); print(f'{nm:38s} rmse={rm:.4f} params={np.round(p,3)}')
m=N>=32
print('so N>=32 (regime de muitos vizinhos):')
for nm,X in [('N',N[m]),('N K',(N*K)[m])]:
    A=np.c_[np.ones(m.sum()),np.log(X)]; b=np.linalg.lstsq(A,q[m],rcond=None)[0]; print(f'  {nm:6s} rmse={np.sqrt(np.mean((A@b-q[m])**2)):.4f}')
