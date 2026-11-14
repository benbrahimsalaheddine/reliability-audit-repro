# (1) Jigsaw (3 pools, equal loadings): order by rho_k and by rho_star, pair gaps with 90% item-bootstrap intervals (500 draws, default_rng(0)).
# (2) D3code pair North America - Sinosphere on its own items (>=2 raters in both regions): r, rho_k, rho_star, tau_hat^2 per panel.
# Definitions as in assumption tables: rho_k=(Var-noise)/Var ; rho_star=Cov/Var ; tau2=Var-Cov-noise ; noise = mean within-item var * mean(1/k).
import pandas as pd, numpy as np, itertools
rng=np.random.default_rng(0)
def panels(R,minK):
    g=R.groupby(['item','panel']).y; M=g.mean().unstack(); K=g.count().unstack(); V=g.var(ddof=1).unstack()
    ok=(K>=minK).all(axis=1); return M[ok],K[ok],V[ok]
def stats(X,noise):
    C=np.cov(X.T,ddof=1); out={}
    for a,b in itertools.permutations(range(X.shape[1]),2):
        out[(a,b)]=(( C[a,a]-noise[a])/C[a,a], C[a,b]/C[a,a])
    return out,C
J=pd.read_csv('/tmp/jig/specialized_rater_pools_data.csv',usecols=['id','toxic_score','rater_group']).dropna(subset=['toxic_score'])
J=J.rename(columns={'id':'item','rater_group':'panel','toxic_score':'y'})
M,K,V=panels(J,2); P=list(M.columns); n=len(M); noise=V.mean().values*(1/K).mean().values; X=M.values
print('Jigsaw pools',P,'items',n)
base,C=stats(X,noise)
for a in range(3): print('  %-17s rho_k %.3f  rho_star(with each other pool) %s'%(P[a],base[(a,(a+1)%3)][0],', '.join('%s %.3f'%(P[b],base[(a,b)][1]) for b in range(3) if b!=a)))
B=10000; gaps={}
for _ in range(B):
    s=rng.integers(0,n,n); Xb=X[s]; bs,_=stats(Xb,noise)
    for a,b in itertools.combinations(range(3),2):
        gaps.setdefault((a,b),[]).append((bs[(a,a+1 if a+1!=3 else 0)][0] if False else None))
# explicit pair gaps
def gap(X):
    Cb=np.cov(X.T,ddof=1); res={}
    for a,b in itertools.combinations(range(3),2):
        rk=((Cb[a,a]-noise[a])/Cb[a,a])-((Cb[b,b]-noise[b])/Cb[b,b]); rs=Cb[a,b]/Cb[a,a]-Cb[a,b]/Cb[b,b]; res[(a,b)]=(rk,rs)
    return res
g0=gap(X); rng=np.random.default_rng(0); bg={k:[] for k in g0}
for _ in range(B):
    gb=gap(X[rng.integers(0,n,n)])
    for k in gb: bg[k].append(gb[k])
for k,(rk,rs) in g0.items():
    arr=np.array(bg[k]); lo=np.percentile(arr,5,axis=0); hi=np.percentile(arr,95,axis=0)
    sig=[(lo[i]>0 or hi[i]<0) for i in (0,1)]
    print('  pair %s - %s : gap rho_k %+.3f [%+.3f,%+.3f] ; gap rho_star %+.3f [%+.3f,%+.3f] ; reversed in point estimate: %s ; both gaps exclude 0: %s'%(P[k[0]],P[k[1]],rk,lo[0],hi[0],rs,lo[1],hi[1],np.sign(rk)!=np.sign(rs),all(sig)))
R=pd.read_csv('d3repo/dataset/d3-ratings.csv'); H=pd.read_csv('d3repo/dataset/d3-raters.csv'); R=R.merge(H[['rater_id','Region']],on='rater_id'); R=R[R.rating_raw>=0]
R=R.rename(columns={'item_id':'item','Region':'panel','rating_raw':'y'})[['item','panel','y']]
R=R[R.panel.isin(['North America','Sinosphere'])]; M,K,V=panels(R,2); P=list(M.columns); n=len(M); noise=V.mean().values*(1/K).mean().values; C=np.cov(M.values.T,ddof=1)
print('\nD3code North America - Sinosphere, own items (>=2 raters in both):',n,'items')
print('  r = %.3f ; covariance %.3f'%(C[0,1]/np.sqrt(C[0,0]*C[1,1]),C[0,1]))
for a in range(2): print('  %-14s var %.3f noise %.3f rho_k %.3f rho_star %.3f tau2_hat %.3f'%(P[a],C[a,a],noise[a],(C[a,a]-noise[a])/C[a,a],C[0,1]/C[a,a],C[a,a]-C[0,1]-noise[a]))
rel=[(C[a,a]-noise[a]) for a in range(2)]; print('  shared share of reliable variance: Cov/(mean reliable var) = %.3f ; Cov/sqrt(prod reliable var) = %.3f'%(C[0,1]/np.mean(rel),C[0,1]/np.sqrt(rel[0]*rel[1])))
