# False-positive rate of the per-panel test, resumable and vectorised. Model: ybar_p,i = l_p mu_i + e_p,i, Var(mu)=1, noise 0.25, tau^2 = 0 for every panel.
# Two panels: tau_hat_p = Var(ybar_p) - Cov(ybar_p,ybar_q) - noise (Eq. 3.8). Three panels: triad estimate (Eq. 3.26).
# One-sided 5% test: tau_hat > 1.645 * bootstrap SE (item bootstrap, B = 10000). Chunks of CH replications, seed [0, cell, chunk].
import numpy as np, json, os, sys, time
noise=0.25; B=10000; R=5000; CH=int(os.environ.get('CH',250)); SEED=0; BS=2000
def taus(S,P):
    out=[]
    for p in (0,1):
        v=S[...,p,p]-noise
        if P==2: out.append(v-S[...,p,1-p])
        else:
            o=[q for q in range(P) if q!=p]; out.append(v-S[...,p,o[0]]*S[...,p,o[1]]/S[...,o[0],o[1]])
    return out
def cov(Y):  # Y (...,n,P)
    Yc=Y-Y.mean(-2,keepdims=True); return np.einsum('...ni,...nj->...ij',Yc,Yc)/(Y.shape[-2]-1)
cells=[(n,r,P) for n in (200,1000) for r in (1.0,1.3,1.6) for P in (2,3)]
f='./std/ckpt/W1.jsonl'
rows=[json.loads(l) for l in open(f)] if os.path.exists(f) else []
done={(r['cell'],r['c']) for r in rows}
cl=[int(a) for a in sys.argv[1:]] if len(sys.argv)>1 else range(len(cells))
for ci in cl:
    n,ratio,P=cells[ci]
    for c in range(R//CH):
        if (ci,c) in done: continue
        rng=np.random.default_rng([SEED,ci,c]); l=np.array([ratio]+[1.0]*(P-1))
        rej=np.zeros(2); mean=np.zeros(2); t0=time.time()
        for _ in range(CH):
            mu=rng.standard_normal(n); Y=mu[:,None]*l+np.sqrt(noise)*rng.standard_normal((n,P))
            t=taus(cov(Y),P); bt=[[],[]]
            for s in range(0,B,BS):
                idx=rng.integers(0,n,(BS,n)); tb=taus(cov(Y[idx]),P)
                for p in (0,1): bt[p].append(tb[p])
            for p in (0,1):
                se=np.std(np.concatenate(bt[p]),ddof=1); mean[p]+=t[p]; rej[p]+=t[p]>1.645*se
        open(f,'a').write(json.dumps({'cell':ci,'n':n,'ratio':ratio,'P':P,'c':c,'rej':rej.tolist(),'sum':mean.tolist(),'sec':time.time()-t0})+'\n')
rows=[json.loads(l) for l in open(f)]
for ci,(n,ratio,P) in enumerate(cells):
    rr=[r for r in rows if r['cell']==ci]; m=CH*len(rr)
    if m: print('n=%4d ratio %.1f P=%d : mean tau_hat^2 larger %+.3f smaller %+.3f ; flagged larger %.3f smaller %.3f  (%d/%d chunks)'%(n,ratio,P,sum(r['sum'][0] for r in rr)/m,sum(r['sum'][1] for r in rr)/m,sum(r['rej'][0] for r in rr)/m,sum(r['rej'][1] for r in rr)/m,len(rr),R//CH))
