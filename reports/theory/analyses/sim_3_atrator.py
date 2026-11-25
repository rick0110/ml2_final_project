"""Mechanistic model: coverage + attractor of the default voice.
Voice space R^d, v ~ N(0, diag(j^-alpha)) (alpha=0: isotropic); coordinate 0 = 'pitch' (F0). Default voice v0 = delta*e0 (female, high-pitched).
Two 'coverage' decoders (they render the code c as a local average of the trained voices):
 gp: Gaussian-process regression (RBF, scale ell, amplitude amp2) of the residual r(v)=g(v)-v0, g(v)=v,
     prior mean v0 (fine-tuning from a single-speaker checkpoint) + anchor at v0 (M utterances, residual 0);
 nw: Nadaraya-Watson per UTTERANCE: the default voice weighs M utterances, each trained speaker weighs K utterances:
     yhat(c) = [M k(c,v0) v0 + K sum_i k(c,v_i) ybar_i] / [M k(c,v0) + K sum_i k(c,v_i)].
Trained speaker: mean of K utterances, noise s2_eff = sb2 + s2/K. Test: c = v* + xi (sd sref) per utterance,
output yhat(c) + eta (sd seta); 'copied' if |output - v*| < |output - v0|.
Usage: python sim_3_atrator.py grade gp|nw      or   python sim_3_atrator.py run "dict(...)" """
import sys, itertools, numpy as np
from scipy.stats import spearmanr
from scipy.linalg import cho_factor, cho_solve
M=2000
sq=lambda A,B:np.maximum((A*A).sum(1)[:,None]+(B*B).sum(1)[None]-2*A@B.T,0)
def predict(P,Vtr,Ybar,v0,C):
    ell=P['ell']
    if P['dec']=='gp':
        X=np.vstack([Vtr,v0[None]]); R=np.vstack([Ybar-v0,np.zeros((1,len(v0)))])
        noise=np.r_[np.full(len(Vtr),P['sb2']+P['s2']/P['K']),P['s2']/M]
        Kx=P['amp2']*np.exp(-sq(X,X)/(2*ell**2))+np.diag(noise)
        return v0+P['amp2']*np.exp(-sq(C,X)/(2*ell**2))@cho_solve(cho_factor(Kx),R)
    out=np.empty_like(C)
    for i in range(0,len(C),2000):
        c=C[i:i+2000]; L=-sq(c,Vtr)/(2*ell**2)+np.log(P['K']); L0=-((c-v0)**2).sum(1)/(2*ell**2)+np.log(M)
        mx=np.maximum(L.max(1),L0); W=np.exp(L-mx[:,None]); w0=np.exp(L0-mx); Z=W.sum(1)+w0
        out[i:i+2000]=(W@Ybar+w0[:,None]*v0)/Z[:,None]
    return out
def simulate(P,Ns,seeds=4,ntest=400,nutt=25,ret_voice=False):
    d=P['d']; v0=np.zeros(d); v0[0]=P['delta']; out={N:[] for N in Ns}; per={N:[] for N in Ns}
    for sd in range(seeds):
        sc=np.arange(1,d+1)**(-P.get('alpha',0)/2); r=np.random.default_rng(sd); Vall=r.normal(size=(max(Ns),d))*sc
        Yall=Vall+r.normal(scale=np.sqrt(P['sb2']+P['s2']/P['K']),size=Vall.shape)
        rt=np.random.default_rng(10_000+sd); T=rt.normal(size=(ntest,d))*sc
        C=T[:,None,:]+rt.normal(scale=P['sref'],size=(ntest,nutt,d)); E=rt.normal(scale=P['seta'],size=(ntest,nutt,d))
        for N in Ns:
            Y=predict(P,Vall[:N],Yall[:N],v0,C.reshape(-1,d)).reshape(ntest,nutt,d)+E
            mm=np.linalg.norm(Y-v0,axis=-1)-np.linalg.norm(Y-T[:,None],axis=-1)
            cop=(mm>0).mean(1); out[N].append(1-cop.mean())
            if ret_voice:
                per[N].append(dict(cop=cop,m=mm.mean(1),sw=mm.std(1),nn=np.sqrt(sq(T,Vall[:N])).min(1),
                                   dist0=np.linalg.norm(T-v0,axis=1),pitch=T[:,0]))
    return {N:np.mean(v) for N,v in out.items()},{N:np.std(v) for N,v in out.items()},per
def icc_ratio(per):
    return np.mean([x['m'].std()/np.sqrt(np.mean(x['sw']**2)) for x in per])
obs=np.array([.434,.302,.211])
if __name__=='__main__' and sys.argv[1]=='grade':
    dec=sys.argv[2]; best=[]
    alphas=(0.0,) if len(sys.argv)<4 else tuple(float(a) for a in sys.argv[3].split(','))
    ds=(4,6,8,12,16) if len(sys.argv)<4 else (16,32)
    grid=itertools.product(ds,(0.3,0.5,1.0,1.5,2.5),(1.0,2.0,3.0),(0.1,0.3),alphas)
    for d,ell,delta,seta,alpha in grid:
        P=dict(dec=dec,d=d,ell=ell,amp2=4.0,delta=delta,sb2=0.05,s2=0.5,K=32,sref=0.2,seta=seta,alpha=alpha)
        q,_,per=simulate(P,(8,32,128),seeds=2,ntest=200,nutt=15,ret_voice=True)
        qq=np.array([q[8],q[32],q[128]]); best.append((np.sqrt(np.mean((qq-obs)**2)),P,qq,icc_ratio(per[32])))
    best.sort(key=lambda t:t[0])
    for b in best[:10]: print(round(b[0],4),{k:b[1][k] for k in ('dec','d','ell','delta','seta','alpha')},np.round(b[2],3),'sd_between/sd_within=%.1f'%b[3])
    sys.exit()
if __name__=='__main__' and sys.argv[1]=='run':
    P=dict(dec='nw',d=8,ell=1.0,amp2=4.0,delta=2.0,sb2=0.05,s2=0.5,K=32,sref=0.2,seta=0.1); P.update(eval(sys.argv[2]))
    print('parameters:',P)
    Ns=(8,32,128,247,1000,2456,10000) if P['dec']=='nw' else (8,32,128,247,1000,2456)
    q,qs,per=simulate(P,Ns,seeds=6,ntest=300,ret_voice=True)
    print('N     : '+' '.join(f'{N:6d}' for N in Ns)); print('q     : '+' '.join(f'{q[N]:6.3f}' for N in Ns)); print('sd seed: '+' '.join(f'{qs[N]:6.3f}' for N in Ns))
    lN=np.log(Ns); lq=np.log([q[N] for N in Ns]); print('local exponent -dlnq/dlnN between points:',np.round(-np.diff(lq)/np.diff(lN),3))
    print('observed (38 voices): q(8,32,128)=',obs,'; margin sd_between/sd_within ~ 2.5 (0.09/0.036)')
    for N in (8,32,128): print(f'  N={N}: sd_between/sd_within = {icc_ratio(per[N]):.2f}')
    print('per-voice rank correlations (mean over seeds), copy vs:')
    for N in (8,32,128,1000):
        c=np.nanmean([(spearmanr(x['cop'],x['dist0'])[0],spearmanr(x['cop'],x['pitch'])[0],spearmanr(x['cop'],x['nn'])[0]) for x in per[N]],0)
        print(f'  N={N:4d}: dist. default voice {c[0]:+.2f} | pitch {c[1]:+.2f} | dist. trained neighbour {c[2]:+.2f}')
    print('  observed: dist.default +.47/+.58/+.53; F0 -.53/-.52/-.37; neighbour -.33/-.40/+.13 (N=8/32/128)')
    for N in (8,32,128):
        cop=np.concatenate([x['cop'] for x in per[N]]); print(f'  N={N}: fraction of voices with 10%<copy<90%: {np.mean((cop>.1)&(cop<.9)):.2f}; copy=0: {np.mean(cop==0):.2f}; copy=1: {np.mean(cop==1):.2f}')
    for a,b in ((8,32),(32,128),(8,128)):
        print(f'  rank corr. copy N={a} vs {b}: {np.nanmean([spearmanr(per[a][s]["cop"],per[b][s]["cop"])[0] for s in range(6)]):.2f}')
    print('  corr(margin gain 8->128, margin at 8) = %.2f (observed -0.55)'%np.mean([np.corrcoef(per[128][s]['m']-per[8][s]['m'],per[8][s]['m'])[0,1] for s in range(6)]))
    print('panel noise (same voices, different training seeds):')
    for N in (8,32,128):
        Cc=np.array([x['cop'] for x in per[N]])
        for nv in (4,38):
            blocks=range(0,Cc.shape[1]-nv,nv)
            print(f'  N={N:3d} panel {nv:2d} voices: sd between seeds={np.mean([np.std(Cc[:,i:i+nv].mean(1)) for i in blocks]):.3f}  mean range={np.mean([np.ptp(Cc[:,i:i+nv].mean(1)) for i in blocks]):.3f}')
    if P['dec']=='nw':
        # effect of the default-voice weight M and of K (predictions for new training runs)
        print('effect of K and M (nw), q per N:')
        for K in (8,32,128):
            Q=dict(P); Q['K']=K; qq,_,_=simulate(Q,(8,32,128,1000),seeds=4,ntest=400); print(f'  K={K:3d}: '+' '.join(f'{qq[N]:.3f}' for N in (8,32,128,1000)))
