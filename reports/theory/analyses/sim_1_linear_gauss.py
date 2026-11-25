"""Regime (a): linear-Gaussian random-effects model.
Voice of speaker i: u_i ~ N(0, I_d) (code of dimension d); output y_ij = W u_i + b_i + e_ij in R^D,
b_i ~ N(0, sb2 I_D) (idiosyncrasy outside the code), e_ij ~ N(0, s2 I_D). Default voice: u=0, known
intercept (fixed by the M=2000 utterances). Decoder: W estimated by OLS on the per-speaker means.
Trained voices: BLUP (per-speaker lookup shrunk toward W u_i).
Formulas tested (per output coordinate):
  new:      E = sb2 + (sb2 + s2/K) d/(N-d-1)                         [exact]
  trained:  E ~= sb2 (s2/K)/(sb2+s2/K) + (1-lam)^2 (sb2+s2/K) d/N  [approx.]
  noisy reference (R utterances, noise sx2 on the code): + |W|^2 sx2/R   (encoder floor)"""
import numpy as np
rng=np.random.default_rng(0)
d,D,sb2,s2=4,24,0.05,0.5
def run(N,K,reps=400,sx2=0.0,R=1):
    eu=[];es=[]
    for _ in range(reps):
        W=rng.normal(size=(D,d))/np.sqrt(d)
        U=rng.normal(size=(N,d)); B=rng.normal(scale=np.sqrt(sb2),size=(N,D))
        Ybar=U@W.T+B+rng.normal(scale=np.sqrt(s2/K),size=(N,D))
        What=np.linalg.lstsq(U,Ybar,rcond=None)[0].T
        # new voice
        us=rng.normal(size=(200,d)); bs=rng.normal(scale=np.sqrt(sb2),size=(200,D))
        z=us+rng.normal(scale=np.sqrt(sx2/R),size=us.shape) if sx2>0 else us
        eu.append(np.mean(((z@What.T)-(us@W.T+bs))**2))
        # trained (BLUP)
        lam=sb2/(sb2+s2/K); pred=U@What.T; yhat=pred+lam*(Ybar-pred)
        es.append(np.mean((yhat-(U@W.T+B))**2))
    return np.mean(eu),np.mean(es),np.std(eu)/np.sqrt(reps)
print('N K | new sim  formula | trained sim  approx')
rows=[]
for N in (8,16,32,64,128,256):
    for K in (2,8,32,128):
        eu,es,se=run(N,K)
        fu=sb2+(sb2+s2/K)*d/(N-d-1)
        lam=sb2/(sb2+s2/K); fs=sb2*(s2/K)/(sb2+s2/K)+(1-lam)**2*(sb2+s2/K)*d/N
        rows.append((N,K,eu,es)); print(f'{N:4d} {K:4d} | {eu:.4f} {fu:.4f} | {es:.4f} {fs:.4f}')
rows=np.array(rows)
# fit a + A/N + B/(NK) on the new voices, N>=32 (away from the curvature of d/(N-d-1))
m=rows[:,0]>=32; N,K,e=rows[m,0],rows[m,1],rows[m,2]
X=np.c_[np.ones_like(N),1/N,1/(N*K)]; p=np.linalg.lstsq(X,e,rcond=None)[0]
print('\nfit new a+A/N+B/(NK) (N>=32):',np.round(p,4),' predicted a=sb2=%.3f A=d*sb2=%.3f B=d*s2=%.3f'%(sb2,d*sb2,d*s2))
print('  -> d estimated by A/a = %.2f (true %d); K0=B/A=%.1f (true s2/sb2=%.1f)'%(p[1]/p[0],d,p[2]/p[1],s2/sb2))
# local exponent in N of the excess (new - floor)
for K in (8,128):
    mm=rows[:,1]==K; ex=rows[mm,2]-sb2; NN=rows[mm,0]
    sl=np.polyfit(np.log(NN[2:]),np.log(ex[2:]),1)[0]; print(f'K={K}: slope log(excess) vs log N (N>=32) = {sl:.3f} (predicted -1)')
# trained: fit a'+C/K+A'/N
N,K,e=rows[:,0],rows[:,1],rows[:,3]; X=np.c_[np.ones_like(N),1/K,1/N]; p=np.linalg.lstsq(X,e,rcond=None)[0]
print("trained a'+C/K+A'/N:",np.round(p,4),' (C predicted ~ s2=%.2f when K>>s2/sb2)'%s2)
print('\nencoder floor: N=128,K=32, code noise sx2=0.3')
for R in (1,4,16):
    eu,_,_=run(128,32,reps=200,sx2=0.3,R=R); print(f'  R={R:2d}: new error={eu:.4f}  predicted={sb2+(sb2+s2/32)*d/123+0.3/R:.4f}')
