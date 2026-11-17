"""Modulo parametrico de dimensao k (sim_4, item 5) em N grande: piso e expoente do excesso, com e sem ruido
de codigo/frase (sref, seta). Excesso = q(N) - q(N=20000)."""
import numpy as np, sim_4_variantes as V
for sref,seta in ((0.2,0.1),(0.0,0.0)):
    V.P['sref']=sref; V.P['seta']=seta
    for k in (4,8):
        Ns=(16,32,128,512,2048,20000); q=np.array([V.q_custom(V.lin_render(N,k),seeds=6,ntest=500) for N in Ns])
        ex=q[:-1]-q[-1]; sl=-np.diff(np.log(np.maximum(ex,1e-4)))/np.diff(np.log(Ns[:-1]))
        print(f'sref={sref} seta={seta} k={k}: q={np.round(q,4)} piso formula={V.q_floor(k):.4f} expoentes locais do excesso={np.round(sl,2)}')
