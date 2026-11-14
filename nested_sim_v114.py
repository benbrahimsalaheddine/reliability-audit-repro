# Nested-document calibration, SummEval design: 100 documents x 16 summaries, experts k=3, crowd k=5.
# Rerun written from the description in Appendix "Nested-document calibration" (the original code is not in the project).
import numpy as np, sys
rng=np.random.default_rng(0 if len(sys.argv)<6 else int(sys.argv[5])); D,S=100,16; n=D*S; k1,k2=3,5; sig2=1.0; sigs2=1.0
doc=np.repeat(np.arange(D),S)
def stat(Y1,Y2,idx):
    a=Y1[idx]; b=Y2[idx]; m1=a.mean(1); m2=b.mean(1)
    v1=a.var(axis=1,ddof=1).mean(); kb=1/a.shape[1]
    return np.var(m1,ddof=1)-np.cov(m1,m2,ddof=1)[0,1]-v1*kb
def lower(Y1,Y2,mode,B):
    vals=np.empty(B)
    for b in range(B):
        if mode=='item': idx=rng.integers(0,n,n)
        else:
            dsel=rng.integers(0,D,D); idx=(dsel[:,None]*S+np.arange(S)[None,:]).ravel()
        vals[b]=stat(Y1,Y2,idx)
    return np.percentile(vals,5)
def rep(icc,tau2,B):
    de=rng.normal(scale=np.sqrt(icc*sigs2),size=D)[doc]; ie=rng.normal(scale=np.sqrt((1-icc)*sigs2),size=n)
    mu=de+ie
    Y1=mu[:,None]+rng.normal(scale=np.sqrt(tau2),size=n)[:,None]+rng.normal(scale=np.sqrt(sig2),size=(n,k1))
    Y2=mu[:,None]+rng.normal(scale=np.sqrt(tau2),size=n)[:,None]+rng.normal(scale=np.sqrt(sig2),size=(n,k2))
    al=np.arange(n)
    return lower(Y1,Y2,'item',B)>0, lower(Y1,Y2,'doc',B)>0
R=int(sys.argv[1]); B=int(sys.argv[2])
ICC=[float(x) for x in sys.argv[3].split(',')] if len(sys.argv)>3 else [0.0,0.3,0.6]
TAU=[float(x) for x in sys.argv[4].split(',')] if len(sys.argv)>4 else [0.0,0.3]
for icc in ICC:
    for tau2 in TAU:
        r=np.array([rep(icc,tau2,B) for _ in range(R)]).mean(0)
        print('intra-doc corr',icc,'tau2',tau2,'item bootstrap',round(float(r[0]),3),'document bootstrap',round(float(r[1]),3),'reps',R,'draws',B)
