"""Leis candidatas ajustadas aos 3 pontos das 38 vozes, previsoes para N=247,1000,2456 (K=32),
N minimo para 90%/95% de copia; teste do elo binario por voz; previsoes do modelo cobertura+atrator
(sim_3, nw, d=16, alpha=1) na grade N x K comparadas aos 16 treinos (4 vozes novas)."""
import numpy as np, pandas as pd
from scipy.optimize import least_squares, brentq
from scipy.stats import norm
import sim_3_atrator as S
Nw=np.array([8.,32.,128.]); qw=np.array([.434,.302,.211])
laws={
 'potencia c N^-b':        (lambda p,N:p[0]*N**-p[1],[0.7,0.26]),
 'a + A/N':                (lambda p,N:p[0]+p[1]/N,[0.2,2]),
 'a + A/sqrt(N)':          (lambda p,N:p[0]+p[1]/np.sqrt(N),[0.1,1]),
 'probit: Phi(a - b lnN)': (lambda p,N:norm.cdf(p[0]-p[1]*np.log(N)),[0.3,0.2]),
 'probit-raiz: Phi(-sqrt(a+b lnN))':(lambda p,N:norm.cdf(-np.sqrt(np.abs(p[0])+np.abs(p[1])*np.log(N))),[0.0,0.1]),
}
print('lei | parametros | rmse | q(247) q(1000) q(2456) | N p/ 90% | N p/ 95%')
for k,(f,p0) in laws.items():
    r=least_squares(lambda p:f(p,Nw)-qw,p0); p=r.x
    def nmin(t):
        h=lambda l:f(p,np.exp(l))-t
        return np.exp(brentq(h,np.log(8),np.log(1e15))) if h(np.log(1e15))<0 else np.inf
    print(f'{k:34s} {np.round(p,3)} {np.sqrt(np.mean(r.fun**2)):.4f} | '+' '.join(f'{f(p,n):.3f}' for n in (247,1000,2456))+f' | {nmin(.10):.3g} | {nmin(.05):.3g}')
print('modelo cobertura+atrator (sim_3 nw d=16 a=1): q(247,1000,2456,10000)=.177 .115 .086 .062 -> N90~1.7e3, N95~2e4 (extrap. b=0.25)')
# elo binario por voz
M=pd.read_csv('dados_margem_por_locutor_wide.csv',index_col=0); M.columns=[8,32,128]
P=pd.read_csv('dados_naocopiado_por_locutor_wide.csv',index_col=0); P.columns=[8,32,128]
sw=0.036
print('\nelo binario: q observado vs E_s Phi(-mu_s/sigma_w) com sigma_w=0.036 (margens medias por voz)')
for N in (8,32,128):
    print(f'  N={N}: q obs={P[N].mean():.3f}  previsto={np.mean(norm.cdf(-M[N].values/sw)):.3f}  probit gaussiano das margens={norm.cdf(-M[N].mean()/np.sqrt(M[N].var()+sw**2)):.3f}'
          f'  fracao de vozes com 10%<copia<90%: {np.mean((P[N]>.1)&(P[N]<.9)):.2f}')
# grade
G=pd.read_csv('dados_grade.csv'); U=G[G.grp=='unseen'].groupby(['N','K']).q.mean().reset_index()
Pm=dict(dec='nw',d=16,ell=0.5,amp2=4.0,delta=2.0,sb2=0.05,s2=0.5,K=32,sref=0.2,seta=0.1,alpha=1.0)
qm=[]
for _,row in U.iterrows():
    Q=dict(Pm); Q['K']=int(row.K); q,_,_=S.simulate(Q,(int(row.N),),seeds=4,ntest=300,nutt=15); qm.append(q[int(row.N)])
U['modelo']=qm
X=np.c_[np.ones(len(U)),U.modelo]; b=np.linalg.lstsq(X,U.q,rcond=None)[0]; res=U.q-X@b
print('\ngrade (4 vozes novas, media das sementes) vs modelo (sem reajuste; regressao linear q_obs = a + b q_modelo):')
print(U.round(3).to_string(index=False)); print(f'  a={b[0]:.3f} b={b[1]:.3f} rmse={np.sqrt(np.mean(res**2)):.3f} corr={np.corrcoef(U.q,U.modelo)[0,1]:.2f}')
print('  ruido esperado de uma media de 1-2 sementes em 4 vozes: ~0.05-0.08')
