"""(i) Floor of the parametric module q_floor(k) = P(|P_perp(v-mu)| > |P_k(v-v0)|) for spectra measured in the latent
(alpha_raw~2, alpha_snr~1.05; d=32) and offset of the default voice delta (in units of sqrt(lambda_1)) along the 1st axis;
(ii) N* = prod_j (1+2 lambda_j/ell^2)^(1/2): number of speakers beyond which a new voice has on average >=1
trained neighbour within scale ell (end of the dependence on the distance to the nearest neighbour)."""
import numpy as np
rng=np.random.default_rng(0)
for alpha in (2.0,1.05):
    l=np.arange(1,33)**(-alpha); T=rng.normal(size=(200000,32))*np.sqrt(l)
    for delta in (1.0,2.0):
        v0=np.zeros(32); v0[0]=delta; row=[]
        for k in (1,2,4,6,8,12,16):
            out=np.sqrt((T[:,k:]**2).sum(1)); inn=np.sqrt(((T[:,:k]-v0[:k])**2).sum(1)); row.append(np.mean(out>inn))
        print(f'alpha={alpha} delta={delta}: q_floor(k=1,2,4,6,8,12,16)='+' '.join(f'{x:.3f}' for x in row)+f' | explained var. top-6={l[:6].sum()/l.sum():.2f}')
for d,alpha,ell in ((16,1.0,0.5),(32,1.0,0.3),(6,0.0,0.75),(8,0.0,1.0)):
    l=np.arange(1,d+1)**(-alpha); print(f'N*(d={d},alpha={alpha},ell={ell}) = {np.prod(np.sqrt(1+2*l/ell**2)):.0f}')
