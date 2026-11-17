"""Regime (a): modelo linear-gaussiano de efeitos aleatorios.
Voz do locutor i: u_i ~ N(0, I_d) (codigo de dimensao d); saida y_ij = W u_i + b_i + e_ij em R^D,
b_i ~ N(0, sb2 I_D) (idiossincrasia fora do codigo), e_ij ~ N(0, s2 I_D). Voz padrao: u=0, intercepto
conhecido (fixado pelas M=2000 frases). Decodificador: W estimado por MQO nas medias por locutor.
Vozes treinadas: BLUP (lookup por locutor encolhido para W u_i).
Formulas testadas (por coordenada de saida):
  nova:     E = sb2 + (sb2 + s2/K) d/(N-d-1)                         [exato]
  treinada: E ~= sb2 (s2/K)/(sb2+s2/K) + (1-lam)^2 (sb2+s2/K) d/N  [aprox.]
  referencia ruidosa (R frases, ruido sx2 no codigo): + |W|^2 sx2/R   (piso do codificador)"""
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
        # nova voz
        us=rng.normal(size=(200,d)); bs=rng.normal(scale=np.sqrt(sb2),size=(200,D))
        z=us+rng.normal(scale=np.sqrt(sx2/R),size=us.shape) if sx2>0 else us
        eu.append(np.mean(((z@What.T)-(us@W.T+bs))**2))
        # treinadas (BLUP)
        lam=sb2/(sb2+s2/K); pred=U@What.T; yhat=pred+lam*(Ybar-pred)
        es.append(np.mean((yhat-(U@W.T+B))**2))
    return np.mean(eu),np.mean(es),np.std(eu)/np.sqrt(reps)
print('N K | nova sim  formula | treinada sim  aprox')
rows=[]
for N in (8,16,32,64,128,256):
    for K in (2,8,32,128):
        eu,es,se=run(N,K)
        fu=sb2+(sb2+s2/K)*d/(N-d-1)
        lam=sb2/(sb2+s2/K); fs=sb2*(s2/K)/(sb2+s2/K)+(1-lam)**2*(sb2+s2/K)*d/N
        rows.append((N,K,eu,es)); print(f'{N:4d} {K:4d} | {eu:.4f} {fu:.4f} | {es:.4f} {fs:.4f}')
rows=np.array(rows)
# ajuste a + A/N + B/(NK) nas novas, N>=32 (fora da curvatura de d/(N-d-1))
m=rows[:,0]>=32; N,K,e=rows[m,0],rows[m,1],rows[m,2]
X=np.c_[np.ones_like(N),1/N,1/(N*K)]; p=np.linalg.lstsq(X,e,rcond=None)[0]
print('\najuste novas a+A/N+B/(NK) (N>=32):',np.round(p,4),' previsto a=sb2=%.3f A=d*sb2=%.3f B=d*s2=%.3f'%(sb2,d*sb2,d*s2))
print('  -> d estimado por A/a = %.2f (verdadeiro %d); K0=B/A=%.1f (verdadeiro s2/sb2=%.1f)'%(p[1]/p[0],d,p[2]/p[1],s2/sb2))
# expoente local em N do excesso (nova - piso)
for K in (8,128):
    mm=rows[:,1]==K; ex=rows[mm,2]-sb2; NN=rows[mm,0]
    sl=np.polyfit(np.log(NN[2:]),np.log(ex[2:]),1)[0]; print(f'K={K}: inclinacao log(excesso) vs log N (N>=32) = {sl:.3f} (previsto -1)')
# treinadas: ajuste a'+C/K+A'/N
N,K,e=rows[:,0],rows[:,1],rows[:,3]; X=np.c_[np.ones_like(N),1/K,1/N]; p=np.linalg.lstsq(X,e,rcond=None)[0]
print("treinadas a'+C/K+A'/N:",np.round(p,4),' (C previsto ~ s2=%.2f quando K>>s2/sb2)'%s2)
print('\npiso do codificador: N=128,K=32, ruido de codigo sx2=0.3')
for R in (1,4,16):
    eu,_,_=run(128,32,reps=200,sx2=0.3,R=R); print(f'  R={R:2d}: erro nova={eu:.4f}  previsto={sb2+(sb2+s2/32)*d/123+0.3/R:.4f}')
