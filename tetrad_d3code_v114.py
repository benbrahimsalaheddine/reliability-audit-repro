# Tetrad test on real D3code: 8 regional panels, items rated in all 8 regions (>=1 rating each; panel mean per item).
import pandas as pd, numpy as np, itertools, sys
R=pd.read_csv('d3repo/dataset/d3-ratings.csv'); H=pd.read_csv('d3repo/dataset/d3-raters.csv')
R=R.merge(H[['rater_id','Region']],on='rater_id'); R=R[R.rating_raw>=0]
M=R.groupby(['item_id','Region']).rating_raw.mean().unstack()
K=R.groupby(['item_id','Region']).rating_raw.count().unstack()
ok=(K>=2).all(axis=1); M=M[ok]; regs=list(M.columns); n=len(M); P=len(regs)
print('items',n,'panels',P)
quads=list(itertools.combinations(range(P),4)); m=3*len(quads)
def tet(X):
    C=np.cov(X.T,ddof=1); out=[]
    for p,q,r,s in quads:
        a=C[p,q]*C[r,s]; b=C[p,r]*C[q,s]; c=C[p,s]*C[q,r]
        out+= [a-b,a-c,b-c]
    return np.array(out)
X=M.values; t0=tet(X)
rng=np.random.default_rng(20261111); B=int(sys.argv[1]) if len(sys.argv)>1 else 300
bs=np.array([tet(X[rng.integers(0,n,n)]) for _ in range(B)])
se=bs.std(axis=0,ddof=1)
T=np.max(np.abs(bs-t0)/se,axis=1); q=np.percentile(T,95)
z=np.abs(t0)/se; nz=z>q
print('B',B,'critical value (studentized max, 95%)',round(float(q),2))
print('differences',m,'non-zero (simultaneous 5% level):',int(nz.sum()),' share',round(float(nz.mean()),3),' largest |t|',round(float(z.max()),1))
sc=np.mean(np.abs(np.cov(X.T,ddof=1))[np.triu_indices(P,1)])**2
print('median |tetrad| / mean|cov|^2 =',round(float(np.median(np.abs(t0))/sc),3),' max',round(float(np.max(np.abs(t0))/sc),3))
cnt=np.zeros(P); tot=np.zeros(P); i=0
for p,q_,r,s_ in quads:
    for _ in range(3):
        for zz in (p,q_,r,s_):
            tot[zz]+=1
            if nz[i]: cnt[zz]+=1
        i+=1
for zz in np.argsort(-cnt/tot): print(regs[zz], int(cnt[zz]),'/',int(tot[zz]), round(cnt[zz]/tot[zz],2))
