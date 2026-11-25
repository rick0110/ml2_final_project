"""Reads (read-only) the results.csv files: statistics of the margin m = sim_ref - sim_default
used to calibrate the binary link. Writes nothing to the repository."""
import pandas as pd, numpy as np, glob, re, os
from scipy.stats import norm, skew
R='/home/user/ml2_final_project/reports/coverage/results'
def load(run, sub='strength_1.0'):
    d=pd.read_csv(f'{R}/{run}/step_2500/{sub}/results.csv'); d['m']=d.sim_ref-d.sim_default; return d
print('== wide evaluation (38 external voices), K=32 ==')
W={}
for N in (8,32,128):
    d=load(f'n{N}_k32_s1-wide'); x=d[d.reference_emotion=='external']
    g=x.groupby('reference_speaker'); mu=g.m.mean(); W[N]=mu
    f0=g.ref_f0_hz.mean()
    print(f'N={N}: q={np.mean(x.m<=0):.3f} mean m={x.m.mean():.4f} median_spk={mu.median():.4f} sd_between={mu.std():.4f} sd_within={np.sqrt(g.m.var().mean()):.4f} skew_spk={skew(mu):.2f} '
          f'corr(m_s, f0_ref)={np.corrcoef(mu,f0)[0,1]:.2f} probit_gauss={norm.cdf(-x.m.mean()/x.m.std()):.3f}')
    if 'sim_spk' in x: pass
M=pd.DataFrame(W)
print('between-run corr of the per-speaker margin:\n',M.corr().round(2))
print('sd of the paired per-speaker shift: 8->32 %.3f, 32->128 %.3f'%((M[32]-M[8]).std(),(M[128]-M[32]).std()))
print('corr(shift 8->128, margin at N=8) = %.2f'%np.corrcoef(M[128]-M[8],M[8])[0,1])
# exponent beta by speaker bootstrap
rng=np.random.default_rng(0); B=[];Bp=[]
Q={N:None for N in W}
per={}
for N in (8,32,128):
    d=load(f'n{N}_k32_s1-wide'); x=d[d.reference_emotion=='external']
    per[N]=(x.m<=0).groupby(x.reference_speaker).mean()
P=pd.DataFrame(per)
for _ in range(10000):
    i=rng.integers(0,len(M),len(M)); mm=M.values[i].mean(0); qq=P.values[i].mean(0)
    r=(mm[2]-mm[1])/(mm[1]-mm[0]);  B.append(-np.log(r)/np.log(4) if r>0 else np.nan)
    z=norm.ppf(qq); r2=(z[2]-z[1])/(z[1]-z[0]); Bp.append(-np.log(r2)/np.log(4) if r2>0 else np.nan)
for nm,b in (('mean margin',B),('probit(q)',Bp)):
    b=np.array(b); print(f'beta via {nm}: median {np.nanmedian(b):.2f}, 95% CI [{np.nanpercentile(b,2.5):.2f}, {np.nanpercentile(b,97.5):.2f}], fraction without solution {np.isnan(b).mean():.2f}')
M.to_csv('dados_margem_por_locutor_wide.csv'); P.to_csv('dados_naocopiado_por_locutor_wide.csv')
print('\n== grid: new (4) and trained voices, all base training runs ==')
rows=[]
for p in sorted(glob.glob(f'{R}/n*_k*_s[12]')):
    run=os.path.basename(p); N,K,s=map(int,re.match(r'n(\d+)_k(\d+)_s(\d)',run).groups())
    try: d=load(run)
    except Exception as e: continue
    for grp in ('unseen','seen'):
        x=d[d.reference_emotion==grp]
        if len(x)==0: continue
        g=x.groupby('reference_speaker').m
        rows.append(dict(run=run,N=N,K=K,seed=s,grp=grp,q=np.mean(x.m<=0),m=x.m.mean(),sd_e=g.mean().std(),sd_d=np.sqrt(g.var().mean()),n=len(x)))
G=pd.DataFrame(rows); print(G.round(3).to_string()); G.to_csv('dados_grade.csv',index=False)
# seed noise per speaker
dif=[]
for (N,K),_ in G[G.grp=='unseen'].groupby(['N','K']):
    try:
        a=load(f'n{N}_k{K}_s1'); b=load(f'n{N}_k{K}_s2')
    except Exception: continue
    for grp in ('unseen',):
        ma=a[a.reference_emotion==grp].groupby('reference_speaker').m.mean(); mb=b[b.reference_emotion==grp].groupby('reference_speaker').m.mean()
        j=ma.index.intersection(mb.index); dif+=list((ma[j]-mb[j]).values)
dif=np.array(dif); print('seed pairs: n=%d per-speaker margin diff.: sd=%.3f -> sigma_seed=%.3f'%(len(dif),dif.std(),dif.std()/np.sqrt(2)))
