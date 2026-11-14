# Calibration of the studentized-max tetrad test at the D3code design: P=8, n=3622, 210 differences.
import numpy as np, itertools
P=8; n=3622; L=np.array([0.97,0.83,0.99,1.0,0.88,0.62,0.88,0.93])
quads=list(itertools.combinations(range(P),4)); idx=np.array(quads)
def tet(X):
    C=np.cov(X.T,ddof=1); out=[]
    p,q,r,s=idx.T
    a=C[p,q]*C[r,s]; b=C[p,r]*C[q,s]; c=C[p,s]*C[q,r]
    return np.concatenate([a-b,a-c,b-c])
def run(rng,gamma,B=400,tau2=0.3,k=3):
    mu=rng.normal(size=n); X=np.empty((n,P))
    for p in range(P): X[:,p]=L[p]*mu+rng.normal(scale=np.sqrt(tau2),size=n)+rng.normal(scale=np.sqrt(1/k),size=n)
    if gamma>0:
        sh=rng.normal(scale=np.sqrt(gamma),size=n); X[:,0]+=sh; X[:,1]+=sh
    t0=tet(X); bs=np.array([tet(X[rng.integers(0,n,n)]) for _ in range(B)])
    se=bs.std(axis=0,ddof=1); q=np.percentile(np.max(np.abs(bs-t0)/se,axis=1),95)
    return bool((np.abs(t0)/se>q).any())
rng=np.random.default_rng(20261111); R=200
for g in (0.0,0.05,0.2):
    r=np.mean([run(rng,g) for _ in range(R)]); print('gamma',g,'rate',round(float(r),3),'reps',R)
