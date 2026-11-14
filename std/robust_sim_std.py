# Robustness of the tau^2 test and estimate to violations of A1, A2, A3, A6, A7, A11, A12, in one place.
# Two panels, equal loadings, n=200 items, mean k=4, sigma_s^2=1, sigma^2=1 (mean), tau^2 = 0 (false positives) or 0.3 (power).
# Test: one-sided 5% item bootstrap (B=250, percentile) on tau_hat_0^2. 1000 replications per cell. default_rng(0).
import numpy as np, time
rng=np.random.default_rng(0); n=200; B=10000; R=5000
def std_exp(size): return rng.exponential(1,size)-1          # skewed, mean 0, variance 1
def ar1(ro,n):
    e=rng.normal(0,1,n); x=np.empty(n); x[0]=e[0]
    for i in range(1,n): x[i]=ro*x[i-1]+np.sqrt(1-ro**2)*e[i]
    return x
def gen(tau2,sc):
    mu=std_exp(n) if sc=='A11 skewed' else rng.normal(0,1,n)
    ks=[np.full(n,4),np.full(n,4)]
    if sc=='A12 unequal k': ks=[rng.integers(1,8,n)+1,rng.integers(1,8,n)+1]   # 2..8, mean 5
    ys=[]; gshare=rng.normal(0,np.sqrt(0.15),n) if sc=='A3 shared comp.' else 0
    for p in range(2):
        t=np.sqrt(tau2)
        if sc=='A1 AR(1) 0.5': c=t*ar1(0.5,n)
        elif sc=='A11 skewed': c=t*std_exp(n)
        else: c=t*rng.normal(0,1,n)
        sig2=2.5 if (sc=='A6 unequal var.' and p==1) else 1.0
        rows=[]
        for i in range(n):
            k=ks[p][i]
            if sc=='A2 var. by item': sd=np.sqrt(sig2*(1+0.8*mu[i]**2)/1.8)
            else: sd=np.sqrt(sig2)
            e=(std_exp(k) if sc=='A11 skewed' else rng.normal(0,1,k))*sd
            a=np.exp(rng.normal(0,0.3,k)) if sc=='A7 rater scales' else 1.0
            y=a*(mu[i]+c[i]+gshare[i] if np.ndim(gshare) else mu[i]+c[i])+e
            if sc=='A11 ordinal 1-5': y=np.clip(np.round(3+0.9*y),1,5)
            rows.append(y)
        m=np.array([r.mean() for r in rows]); w=np.array([r.var(ddof=1) for r in rows]); kk=np.array([len(r) for r in rows])
        ys.append((m,w,kk))
    return ys
def tau0(ys,idx):
    (m0,w0,k0),(m1,w1,k1)=ys[0],ys[1]
    a=m0[idx]; b=m1[idx]; a=a-a.mean(1,keepdims=True); b=b-b.mean(1,keepdims=True)
    N=a.shape[1]; v=(a*a).sum(1)/(N-1); c=(a*b).sum(1)/(N-1)
    s2=w0[idx].mean(1); kap=(1.0/k0[idx]).mean(1)
    return v-c-s2*kap
scs=['base','A2 var. by item','A11 skewed','A11 ordinal 1-5','A12 unequal k','A1 AR(1) 0.5','A6 unequal var.','A3 shared comp.','A7 rater scales']
print('scenario                 tau2  flag rate   mean est   (bias)')
for sc in scs:
    for tau2 in (0.0,0.3):
        t0=time.time(); fl=0; ests=[]
        for r in range(R):
            ys=gen(tau2,sc); pt=tau0(ys,np.arange(n)[None,:])[0]
            lo=np.percentile(tau0(ys,rng.integers(0,n,(B,n))),5); fl+=lo>0; ests.append(pt)
        ref=tau2   # target: panel-specific variance of panel 0 on the observed scale (not rescaled)
        print(f'{sc:22s} {tau2:5.1f}  {100*fl/R:6.1f}%   {np.mean(ests):7.3f}   ({np.mean(ests)-ref:+.3f})   [{time.time()-t0:.0f}s]',flush=True)
