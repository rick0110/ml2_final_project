import sys, numpy as np, collections
z=np.load(sys.argv[1],allow_pickle=False)
for k in z.files: print(k,z[k].shape,z[k].dtype)
C=collections.Counter
print(C(z['emotion'].tolist())); print(C(z['source'].tolist()))
print(C(zip(z['source'].tolist(),z['emotion'].tolist())))
sp=z['speaker']; print(len(set(sp.tolist())), C(sp.tolist()).most_common(5))
for s in set(z['source'].tolist()):
    m=z['source']==s; c=C(sp[m].tolist()); print(s, len(c), sorted(c.values())[:3], sorted(c.values())[-3:], z['utt_id'][m][:2])
print(z['mu'][:2,:5], z['logvar'][:2,:5])
