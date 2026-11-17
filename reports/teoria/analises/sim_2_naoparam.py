"""Regime (b): decodificador de 'cobertura' (vizinho mais proximo e Nadaraya-Watson) em d dimensoes.
Vozes v ~ U[0,1]^d; renderizacao alvo g(v) = sin(A v + c) em R^D (suave); observacao por locutor:
media de K frases, ruido s2_eff = sb2 + s2/K. Mede o excesso de erro numa voz nova vs N e ajusta o expoente.
Previsoes: 1-NN sem ruido ~ N^(-2/d) (g Lipschitz);  1-NN com ruido: piso s2_eff (K vira ruido aditivo);
NW com banda otima: ~ (s2_eff/N)^(4/(4+d)) (nucleo simetrico de ordem 2, g suave: s=2)."""
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
print('1-NN sem ruido: inclinacao de log(erro) vs log N, previsto -2/d')
for d in (2,4,8):
    e=np.array([exp_nn(d,N,0) for N in Ns]); sl=np.polyfit(np.log(Ns[2:]),np.log(e[2:]),1)[0]
    print(f'  d={d}: {sl:.3f}  (previsto {-2/d:.3f})  erros={np.round(e,4)}')
print('1-NN com ruido s2_eff=0.05: erro -> piso ~ s2_eff (K so entra no piso)')
for d in (4,):
    e=np.array([exp_nn(d,N,0.05) for N in Ns]); print(f'  d={d}: erros={np.round(e,4)}')
print('Nadaraya-Watson banda otima, s2_eff=0.1: previsto -4/(4+d)')
Ns2=np.array([32,64,128,256,512,1024,2048])
for d in (2,4,8):
    e=np.array([exp_nw(d,N,0.1) for N in Ns2]); sl=np.polyfit(np.log(Ns2[1:]),np.log(e[1:]),1)[0]
    print(f'  d={d}: {sl:.3f}  (previsto {-4/(4+d):.3f})  erros={np.round(e,4)}')
print('NW, efeito de K via s2_eff = sb2 + s2/K (sb2=0.02, s2=0.5), d=4: erro depende de N/s2_eff?')
for N,K in ((64,8),(64,128),(256,8),(256,128),(1024,8)):
    s2e=0.02+0.5/K; e=exp_nw(4,N,s2e,reps=10); print(f'  N={N:5d} K={K:4d} s2_eff={s2e:.3f} N/s2_eff={N/s2e:8.0f} erro={e:.4f}')
