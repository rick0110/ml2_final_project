"""Fit of the theoretical forms to the mean margin (grid, 16 training runs) and predictions for N=247,1000,2456 (K=32)
on the 38 voices, through the semi-parametric shift link. Uses the csv files produced by sim_0_dados.py."""
import numpy as np, pandas as pd
from scipy.optimize import least_squares, brentq
from scipy.stats import norm
G=pd.read_csv('dados_grade.csv'); U=G[G.grp=='unseen']; S=G[G.grp=='seen']
N,K,m=U.N.values.astype(float),U.K.values.astype(float),U.m.values
forms={
 'a1  : mu - A/N - B/(NK)':            (lambda p:p[0]-p[1]/N-p[2]/(N*K),[.05,.3,1]),
 'a1/2: mu - c*sqrt((1+K0/K)/N)':      (lambda p:p[0]-p[1]*np.sqrt((1+abs(p[2])/K)/N),[.05,.3,5]),
 'b-ad: mu - A*N^-b - C/K':            (lambda p:p[0]-p[1]*N**-abs(p[2])-p[3]/K,[.1,.3,.5,.3]),
 'b-ad2: mu - A*N^-b - C/sqrt(K)':     (lambda p:p[0]-p[1]*N**-abs(p[2])-p[3]/np.sqrt(K),[.1,.3,.5,.1]),
 'b-ot: mu - c*(N*K^t)^-b':            (lambda p:p[0]-p[1]*(N*K**abs(p[2]))**-abs(p[3]),[.1,.3,.3,.5]),
 'log : a + b lnN + c lnK':            (lambda p:p[0]+p[1]*np.log(N)+p[2]*np.log(K),[0,.03,.01]),
 'so N: mu - A*N^-b':                  (lambda p:p[0]-p[1]*N**-abs(p[2]),[.1,.3,.5]),
}
print('== new voices of the grid (16 training runs), mean margin; expected seed noise ~0.010-0.015 ==')
for k,(f,p0) in forms.items():
    r=least_squares(lambda p:f(p)-m,p0); n=len(m); p=len(p0)
    rm=np.sqrt(np.mean(r.fun**2)); print(f'{k:34s} rmse={rm:.4f}  rmse_corr(df)={np.sqrt(np.sum(r.fun**2)/(n-p)):.4f}  p={np.round(r.x,3)}')
# without N=2
msk=N>2
print('-- without N=2 --')
for k,(f,p0) in forms.items():
    r=least_squares(lambda p:(f(p)-m)[msk],p0); print(f'{k:34s} rmse={np.sqrt(np.mean(r.fun**2)):.4f} p={np.round(r.x,3)}')
# NxK interaction: effect of K (per doubling) at each N
print('\neffect of K on the margin (per factor 4 in K), new voices:')
for n_ in (8,32,128):
    u=U[U.N==n_].groupby('K').m.mean(); ks=sorted(u.index)
    print(' N=%d'%n_, {f'{a}->{b}':round((u[b]-u[a])/np.log(b/a)*np.log(4),4) for a,b in zip(ks[:-1],ks[1:])})
print('\n== trained: mean margin ==')
N2,K2,m2=S.N.values.astype(float),S.K.values.astype(float),S.m.values
for k,f,p0 in [('mu - C/K - A/N',lambda p:p[0]-p[1]/K2-p[2]/N2,[.1,1,.1]),('mu - C/sqrt(K) - A/N',lambda p:p[0]-p[1]/np.sqrt(K2)-p[2]/N2,[.1,1,.1]),
               ('mu - C*K^-g + A*lnN (competition)',lambda p:p[0]-p[1]*K2**-abs(p[2])+p[3]*np.log(N2),[.2,.5,.5,0])]:
    r=least_squares(lambda p:f(p)-m2,p0); print(f'{k:34s} rmse={np.sqrt(np.mean(r.fun**2)):.4f} p={np.round(r.x,3)}')
print('\ntrained-new gap (margin), per (N,K):'); 
gp=G.pivot_table(index=['N','K'],columns='grp',values='m'); gp['gap']=gp.seen-gp.unseen; print(gp.round(3).to_string())

print('\n== 38 voices, K=32: predictions ==')
M=pd.read_csv('dados_margem_por_locutor_wide.csv',index_col=0); M.columns=[8,32,128]
Ns=np.array([8.,32.,128.]); sw=0.036
def q_of(mu, base):  # semi-parametric shift link: shifts the per-speaker margins of the N=128 run
    return np.mean(norm.cdf(-(base+ (mu-base.mean()))/sw))
def fitpred(Mb):
    mu=Mb.mean(0).values; base=Mb[128].values; out={}
    laws={'a, nu=1 (1/N)':lambda n,p:p[0]-p[1]/n,'a, nu=1/2 (1/sqrt N)':lambda n,p:p[0]-p[1]/np.sqrt(n),
          'b/c beta=0.25':lambda n,p:p[0]-p[1]*n**-.25,'log N':lambda n,p:p[0]+p[1]*np.log(n)}
    for k,f in laws.items():
        A=np.array([[1,-(f(n,[0,-1]))] for n in Ns]) if k!='log N' else np.array([[1,np.log(n)] for n in Ns])
        if k!='log N': A=np.array([[1,f(n,[0,1])] for n in Ns])
        p=np.linalg.lstsq(A,mu,rcond=None)[0]
        g=(lambda n,p=p,f=f,k=k: p[0]+p[1]*(np.log(n) if k=='log N' else f(n,[0,1])))
        qs=[q_of(g(n),base) for n in (8,32,128,247,1000,2456,1e9)]
        def nmin(t):
            h=lambda ln: q_of(g(np.exp(ln)),base)-t
            return np.exp(brentq(h,np.log(8),np.log(1e12))) if h(np.log(1e12))<0 else np.inf
        out[k]=qs+[nmin(.10),nmin(.05),np.sqrt(np.mean((A@p-mu)**2))]
    return out
o=fitpred(M)
print('law | q(8) q(32) q(128) | q(247) q(1000) q(2456) | q(inf) | N for 90% | N for 95% | margin rmse')
rng=np.random.default_rng(1); bs={k:[] for k in o}
for _ in range(2000):
    i=rng.integers(0,len(M),len(M)); ob=fitpred(M.iloc[i])
    for k in ob: bs[k].append(ob[k])
for k,v in o.items():
    b=np.array(bs[k]); lo,hi=np.percentile(b,[2.5,97.5],axis=0)
    print(f'{k:22s}', ' '.join(f'{x:.3f}' for x in v[:3]),'|',' '.join(f'{x:.3f} [{l:.2f},{h:.2f}]' for x,l,h in zip(v[3:6],lo[3:6],hi[3:6])),'|',f'{v[6]:.3f} [{lo[6]:.2f},{hi[6]:.2f}]','|',f'{v[7]:.3g}','|',f'{v[8]:.3g}','|',f'{v[9]:.4f}')
print('observed: 0.434 0.302 0.211')
