# IRep with the estimator described in v64's parameter table: 3 city pools (International excluded), items with 2 raters in each pool,
# 31 labels, pool covariance = mean of its two pairs (the 'avg cov' estimator), 500 item-bootstrap draws, default_rng(int(__import__("sys").argv[1])).
# tau_hat^2 = Var(ybar_p) - mean_q Cov(ybar_p, ybar_q) - noise_p ; noise_p = mean within-item variance (ddof=1) * mean(1/k).
# A pool is flagged in a class when the lower end of the 90% percentile interval (5th percentile) is above 0 (one-sided 5%).
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
d=pd.read_csv('/tmp/irep/replication-dataset/IREP_data_release.csv'); d.columns=[c.strip('"') for c in d.columns]
lab=[c for c in d.columns if c not in('item_id','platform','rater','error.other')]
pools=['Budapest','Mexico','KualaLumpur']; P=3
s=d[d.platform.isin(pools)]; cnt=s.groupby(['item_id','platform']).size().unstack(); items=cnt.index[(cnt[pools]>=2).all(axis=1)]; s=s[s.item_id.isin(items)]; n=len(items)
print('items',n,'labels',len(lab))
rng=np.random.default_rng(int(__import__("sys").argv[1])); B=int(__import__("sys").argv[2]); flag=np.zeros(P,int); flag_pt=np.zeros(P,int); skipped=0
def tau(X,noise):
    C=np.cov(X.T,ddof=1); return np.array([C[p,p]-np.mean([C[p,q] for q in range(P) if q!=p])-noise[p] for p in range(P)])
for c in lab:
    g=s.groupby(['item_id','platform'])[c]; X=g.mean().unstack()[pools].values; V=g.var(ddof=1).unstack()[pools]; K=g.count().unstack()[pools]
    noise=V.mean().values*(1/K).mean().values
    pt=tau(X,noise); lo=np.percentile([tau(X[rng.integers(0,n,n)],noise) for _ in range(B)],5,axis=0)
    flag+=lo>0; flag_pt+=pt>0
print('flagged classes (lower 5th percentile > 0) of',len(lab),':',dict(zip(pools,flag.tolist())))
print('point estimate tau_hat^2 > 0:',dict(zip(pools,flag_pt.tolist())))
print('v64 reported: Kuala Lumpur 25, Budapest 15, Mexico 8')
