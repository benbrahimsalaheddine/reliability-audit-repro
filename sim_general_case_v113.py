import numpy as np, sys, time
def run(ells, tau2, test, n=200, k=4, sig2=1.0, B=250, R=1000, seed=0, mode='pair'):
    rng=np.random.default_rng(seed); P=len(ells); ells=np.array(ells,float); flags=0
    for r in range(R):
        mu=rng.normal(size=n)
        y=ells[:,None,None]*mu[None,:,None]+rng.normal(scale=np.sqrt(tau2),size=(P,n,1))+rng.normal(scale=np.sqrt(sig2),size=(P,n,k))
        m=y.mean(2); w=y.var(2,ddof=1)  # (P,n)
        idx=rng.integers(0,n,size=(B,n))
        mb=m[:,idx]            # P,B,n
        wb=w[:,idx].mean(2)    # P,B
        mc=mb-mb.mean(2,keepdims=True)
        C=np.einsum('pbn,qbn->bpq',mc,mc)/(n-1)   # B,P,P
        p=test
        V=C[:,p,p]
        if mode=='pair':
            q=1 if p==0 else 0
            sig=C[:,p,q]
        else:
            others=[a for a in range(P) if a!=p]; q,s=others[0],others[1]
            sig=C[:,p,q]*C[:,p,s]/C[:,q,s]
        tau=V-sig-wb[p]/k
        if np.percentile(tau,5)>0: flags+=1
    return flags/R
if __name__=='__main__':
    t=time.time()
    R=int(sys.argv[1])
    out={}
    for name,ells,mode,tp in [('2p equal, test p1',(1,1),'pair',0),('2p 1.0/0.62, test larger-loading panel',(1.0,0.62),'pair',0),('2p 1.0/0.62, test smaller-loading panel',(1.0,0.62),'pair',1),('3p 1.0/0.84/0.62 triad, test p1',(1.0,0.84,0.62),'tri',0),('3p triad, test p3 (smallest)',(1.0,0.84,0.62),'tri',2)]:
        f0=run(ells,0.0,tp,R=R,mode=mode); f1=run(ells,0.3,tp,R=R,mode=mode,seed=1)
        print(name,'FPR',f0,'power',f1,flush=True)
    print('time',time.time()-t)
