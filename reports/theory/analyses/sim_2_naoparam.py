"""Regime (b): 'coverage' decoder (nearest neighbour and Nadaraya-Watson) in d dimensions.
Voices v ~ U[0,1]^d; target rendering g(v) = sin(A v + c) in R^D (smooth); observation per speaker:
mean of K utterances, noise s2_eff = sb2 + s2/K. Measures the excess error on a new voice vs N and fits the exponent.
Predictions: noiseless 1-NN ~ N^(-2/d) (g Lipschitz);  noisy 1-NN: floor s2_eff (K becomes additive noise);
NW with optimal bandwidth: ~ (s2_eff/N)^(4/(4+d)) (symmetric order-2 kernel, smooth g: s=2)."""
import numpy as np
from scipy.spatial import cKDTree
rng=np.random.default_rng(1); D=8
def gfun(A,c,v): return np.sin(v@A.T+c)
def exp_nn(d,N,s2e,reps=30,nt=400):
    out=[]
    for _ in range(reps):
        A=rng.normal(scale=2.0,size=(D,d)); c=rng.uniform(0,6.3,D)
        V=rng.uniform(size=(N,d)); Y=gfun(A,c,V)+rng.normal(scale=np.sqrt(s2e),size=(N,D)) if s2e>0 else gfun(A,c,V)
        T=rng.uniform(size=(nt,d)); _,j=cKDTree(V).query(T); out.append(np.mean((Y[j]-gfun(A,c,T))**2))
    return np.mean(out)
def exp_nw(d,N,s2e,reps=20,nt=300):
    hs=np.geomspace(0.02,1.0,16); err=np.zeros(len(hs))
    for _ in range(reps):
        A=rng.normal(scale=2.0,size=(D,d)); c=rng.uniform(0,6.3,D)
        V=rng.uniform(size=(N,d)); Y=gfun(A,c,V)+rng.normal(scale=np.sqrt(s2e),size=(N,D))
        T=rng.uniform(size=(nt,d)); d2=((T[:,None,:]-V[None])**2).sum(-1); G=gfun(A,c,T)
        for a,h in enumerate(hs):
            Wt=np.exp(-(d2-d2.min(1,keepdims=True))/(2*h*h)); Wt/=Wt.sum(1,keepdims=True)
            err[a]+=np.mean((Wt@Y-G)**2)
    return (err/reps).min()
Ns=np.array([16,32,64,128,256,512,1024,2048])
print('noiseless 1-NN: slope of log(error) vs log N, predicted -2/d')
for d in (2,4,8):
    e=np.array([exp_nn(d,N,0) for N in Ns]); sl=np.polyfit(np.log(Ns[2:]),np.log(e[2:]),1)[0]
    print(f'  d={d}: {sl:.3f}  (predicted {-2/d:.3f})  errors={np.round(e,4)}')
print('1-NN with noise s2_eff=0.05: error -> floor ~ s2_eff (K only enters the floor)')
for d in (4,):
    e=np.array([exp_nn(d,N,0.05) for N in Ns]); print(f'  d={d}: errors={np.round(e,4)}')
print('Nadaraya-Watson optimal bandwidth, s2_eff=0.1: predicted -4/(4+d)')
Ns2=np.array([32,64,128,256,512,1024,2048])
for d in (2,4,8):
    e=np.array([exp_nw(d,N,0.1) for N in Ns2]); sl=np.polyfit(np.log(Ns2[1:]),np.log(e[1:]),1)[0]
    print(f'  d={d}: {sl:.3f}  (predicted {-4/(4+d):.3f})  errors={np.round(e,4)}')
print('NW, effect of K via s2_eff = sb2 + s2/K (sb2=0.02, s2=0.5), d=4: does the error depend on N/s2_eff?')
for N,K in ((64,8),(64,128),(256,8),(256,128),(1024,8)):
    s2e=0.02+0.5/K; e=exp_nw(4,N,s2e,reps=10); print(f'  N={N:5d} K={K:4d} s2_eff={s2e:.3f} N/s2_eff={N/s2e:8.0f} error={e:.4f}')
