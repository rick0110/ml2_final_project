"""Variants of the coverage+attractor model (nw, d=16, alpha=1, fitted to q(8,32,128) of the 38 voices):
 (1) weight of the default voice M vs K: prediction 'q depends on K/M' (besides s2/K);
 (2) fixed budget N*K: best split;
 (3) pseudo-speakers made by shifting along a single direction (pitch): do they count as speakers?;
 (4) mean of R references;
 (5) parametric module of dimension k (PCA subspace of the N voices): floor q_floor(k) and exponent of the excess;
 (6) two stages with voice leakage into the units (weight w of the leaked channel)."""
import numpy as np, sim_3_atrator as S
P=dict(dec='nw',d=16,ell=0.5,amp2=4.0,delta=2.0,sb2=0.05,s2=0.5,K=32,sref=0.2,seta=0.1,alpha=1.0)
d=P['d']; sc=np.arange(1,d+1)**(-P['alpha']/2); v0=np.zeros(d); v0[0]=P['delta']
def q_custom(render,seeds=4,ntest=300,nutt=15,sref=None):
    sref=P['sref'] if sref is None else sref; qs=[]
    for sd in range(seeds):
        r=np.random.default_rng(sd); rt=np.random.default_rng(10_000+sd)
        T=rt.normal(size=(ntest,d))*sc; C=T[:,None,:]+rt.normal(scale=sref,size=(ntest,nutt,d)); E=rt.normal(scale=P['seta'],size=(ntest,nutt,d))
        Y=render(r,C.reshape(-1,d)).reshape(ntest,nutt,d)+E
        qs.append(1-np.mean(np.linalg.norm(Y-T[:,None],axis=-1)<np.linalg.norm(Y-v0,axis=-1)))
    return np.mean(qs)
def nw_render(N,K=32,M=2000,aug=0.0,naug=0):
    def f(r,C):
        V=r.normal(size=(N,d))*sc; Yb=V+r.normal(scale=np.sqrt(P['sb2']+P['s2']/K),size=V.shape)
        if naug:
            sh=np.linspace(-aug,aug,naug+1); sh=sh[sh!=0]
            e0=np.eye(d)[0]; V=np.vstack([V]+[V+s*e0 for s in sh]); Yb=np.vstack([Yb]+[Yb+s*e0 for s in sh])
        S.M=M; Q=dict(P); Q['K']=K; return S.predict(Q,V,Yb,v0,C)
    return f
def lin_render(N,k,K=32):
    def f(r,C):
        V=r.normal(size=(N,d))*sc; Yb=V+r.normal(scale=np.sqrt(P['sb2']+P['s2']/K),size=V.shape)
        X=np.vstack([Yb,v0[None]]); mu=X.mean(0); U=np.linalg.svd(X-mu,full_matrices=False)[2][:k]
        return mu+(C-mu)@U.T@U
    return f
def q_floor(k,n=200000):
    rr=np.random.default_rng(5); T=rr.normal(size=(n,d))*sc; mu=np.zeros(d)
    Pk=np.zeros(d); Pk[:k]=1   # true subspace = first k coordinates (contains v0)
    out=np.sqrt((((T-mu)*(1-Pk))**2).sum(1)); inn=np.sqrt((((T-v0)*Pk)**2).sum(1)); return np.mean(out>inn)
if __name__=="__main__":
    print('(1) weight of the default voice M and utterances per speaker K (q, new voices):')
    for N in (32,128):
        for K,M in ((32,2000),(64,2000),(32,1000),(128,2000),(32,500),(8,2000),(8,500)):
            print(f'   N={N:4d} K={K:4d} M={M:5d} K/M={K/M:.3f}: q={q_custom(nw_render(N,K,M)):.3f}')
    print('(2) budget N*K=4096 and 16384 (M=2000):')
    for B in (4096,16384):
        for K in (2,4,8,16,32,64,128):
            N=B//K; print(f'   B={B} N={N:5d} K={K:4d}: q={q_custom(nw_render(N,K),seeds=3):.3f}')
    print('(3) pseudo-speakers: each trained voice gets copies shifted only in pitch (coord. 0)')
    for N in (8,32):
        print(f'   N={N}: real q={q_custom(nw_render(N)):.3f} | +4 copies/voice (+-0.5) q={q_custom(nw_render(N,aug=0.5,naug=4)):.3f} | +4 copies (+-1.0) q={q_custom(nw_render(N,aug=1.0,naug=4)):.3f} | real with 5N voices q={q_custom(nw_render(5*N)):.3f}')
    print('(4) mean of R references (code noise sref^2/R):')
    for N in (8,128):
        print(f'   N={N}: '+' '.join(f'R={R}: {q_custom(nw_render(N),sref=P["sref"]/np.sqrt(R)):.3f}' for R in (1,4,16)))
    print('(5) parametric module of dimension k: yhat = mu_hat + P_k (c - mu_hat), P_k = PCA of the N voices (noisy)')
    for k in (2,4,8,12,16):
        qs=[q_custom(lin_render(N,k),seeds=3) for N in (8,32,128,1000)]
        print(f'   k={k:2d}: q(N=8,32,128,1000)='+' '.join(f'{x:.3f}' for x in qs)+f' | predicted floor q_floor(k)={q_floor(k):.3f}')
    print('(6) two stages: yhat = (1-w) B_Nvoz(c) + w v_A ; B = nw without a strong default voice (M_B=K), v_A = v0 (text voice)')
    def two_stage(Nv,w):
        base=nw_render(Nv,32,M=32)
        return lambda r,C:(1-w)*base(r,C)+w*v0
    for w in (0.0,0.1,0.2,0.3,0.4):
        print(f'   w={w:.1f}: q(Nvoz=8,32,128,1000,2456)='+' '.join(f'{q_custom(two_stage(N,w),seeds=3):.3f}' for N in (8,32,128,1000,2456)))
    print('   single-stage reference (M=2000): '+' '.join(f'{q_custom(nw_render(N),seeds=3):.3f}' for N in (8,32,128,1000,2456)))
