import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, sys
d=pd.read_csv('/tmp/irep/replication-dataset/IREP_data_release.csv'); d.columns=[c.strip('"') for c in d.columns]
lab=[c for c in d.columns if c not in('item_id','platform','rater','error.other')]
pools=['Budapest','Mexico','KualaLumpur']; P=3
s=d[d.platform.isin(pools)]; cnt=s.groupby(['item_id','platform']).size().unstack(); items=cnt.index[(cnt[pools]>=2).all(axis=1)]; s=s[s.item_id.isin(items)]; n=len(items)
def tau(X,noise,mode):
    C=np.cov(X.T,ddof=1)
    if mode=='avg': return np.array([C[p,p]-np.mean([C[p,q] for q in range(P) if q!=p])-noise[p] for p in range(P)])
    # 'pair' variant: use the larger of the two pair covariances? no: use pairs separately -> min over q
    return np.array([C[p,p]-max(C[p,q] for q in range(P) if q!=p)-noise[p] for p in range(P)])
def run(seed,B,noise_mode,mode,ddof_noise=1):
    rng=np.random.default_rng(seed); fl=np.zeros(3,int); q5s=[]
    for c in lab:
        g=s.groupby(['item_id','platform'])[c]; X=g.mean().unstack()[pools].values; V=g.var(ddof=ddof_noise).unstack()[pools]; K=g.count().unstack()[pools]
        noise=V.mean().values*(1/K).mean().values if noise_mode=='v64' else (V/K).mean().values
        lo=np.percentile([tau(X[rng.integers(0,n,n)],noise,mode) for _ in range(B)],5,axis=0); fl+=lo>0; q5s.append(lo)
    return fl,np.array(q5s)
for nm,mode in (('v64','avg'),('meanVK','avg')):
    fl,_=run(0,500,nm,mode); print('noise=%-7s cov=%s seed0 B=500 ->'%(nm,mode),dict(zip(pools,fl.tolist())))
fl,q=run(0,3000,'v64','avg'); print('B=3000 seed0',dict(zip(pools,fl.tolist())))
for j,p in enumerate(pools):
    idx=np.argsort(np.abs(q[:,j]))[:6]; print(p,'labels closest to 0 (5th pct):',[(lab[i],round(q[i,j],4)) for i in idx])
