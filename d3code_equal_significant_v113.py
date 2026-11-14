import pandas as pd, numpy as np, itertools
R=pd.read_csv('d3repo/dataset/d3-ratings.csv'); H=pd.read_csv('d3repo/dataset/d3-raters.csv')
R=R.merge(H[['rater_id','Region']],on='rater_id'); R=R[R.rating_raw>=0]
regs=sorted(R.Region.unique())
stat=R.groupby(['Region','item_id']).rating_raw.agg(m='mean',v=lambda x: x.var(ddof=1) if len(x)>1 else np.nan,k='count').reset_index()
S={r:stat[(stat.Region==r)&(stat.k>=2)].set_index('item_id') for r in regs}
def diffs(a,b):
    c=np.cov(a.m.values,b.m.values,ddof=1)[0,1]; o=[]
    for s in (a,b):
        V=np.var(s.m.values,ddof=1); sig=np.nanmean(s.v.values); kb=np.mean(1/s.k.values)
        o.append((1-sig*kb/V, c/V))
    return o[0][0]-o[1][0], o[0][1]-o[1][1]
def run(B,alpha,seed=0):
    rng=np.random.default_rng(seed); rev=sig=0
    for p,q in itertools.combinations(regs,2):
        items=S[p].index.intersection(S[q].index); a=S[p].loc[items]; b=S[q].loc[items]
        d1,d2=diffs(a,b); r=d1*d2<0; rev+=r
        bs=np.array([diffs(a.iloc[ix],b.iloc[ix]) for ix in (rng.integers(0,len(items),len(items)) for _ in range(B))])
        ok=True
        for j in (0,1):
            lo,hi=np.percentile(bs[:,j],[100*alpha/2,100*(1-alpha/2)])
            ok&=(lo>0 or hi<0)
        sig+= r and ok and np.sign(bs[:,0].mean())!=np.sign(bs[:,1].mean())
    return rev,sig
print('90% intervals, 500 draws:',run(500,0.10))
print('Bonferroni 0.10/28, 1500 draws:',run(1500,0.10/28))
