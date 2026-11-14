# FaeU W1: with unequal loadings and NO panel-specific component (tau^2 = 0), how often does the one-sided test flag the panel with the larger loading?
# Model: ybar_p,i = l_p mu_i + e_p,i, Var(mu)=1, Var(e_p)=0.25 (noise of the panel mean), tau^2 = 0 for every panel. n items, 400 replications, item bootstrap with 200 draws, one-sided 5% test (tau_hat > 1.645 * bootstrap SE).
# Two panels: tau_hat_p = Var(ybar_p) - Cov(ybar_p,ybar_q) - noise (Eq. 3.8).  Three panels: tau_hat_p = Var(ybar_p) - noise - triad estimate of l_p^2 s2 (Eq. 3.26).  Seed 0.
import numpy as np
rng=np.random.default_rng(0); noise=0.25; R=400; B=200
def tau2(Y,p,P):
    S=np.cov(Y.T,ddof=1); v=S[p,p]-noise
    if P==2: return v-S[p,1-p]
    o=[q for q in range(P) if q!=p]; return v-S[p,o[0]]*S[p,o[1]]/S[o[0],o[1]]
for n in (200,1000):
    for ratio in (1.0,1.3,1.6):
        for P in (2,3):
            l=np.array([ratio]+[1.0]*(P-1)); rej=np.zeros(P); mean=np.zeros(P)
            for _ in range(R):
                mu=rng.standard_normal(n); Y=mu[:,None]*l+np.sqrt(noise)*rng.standard_normal((n,P))
                for p in (0,1):
                    t=tau2(Y,p,P); mean[p]+=t/R
                    se=np.std([tau2(Y[ix],p,P) for ix in (rng.integers(0,n,n) for _ in range(B))],ddof=1); rej[p]+=(t>1.645*se)/R
            print('n=%4d loading ratio %.1f P=%d : mean tau_hat^2 larger-loading panel %+.3f, smaller-loading panel %+.3f ; flagged at nominal 5%%: larger %.3f, smaller %.3f'%(n,ratio,P,mean[0],mean[1],rej[0],rej[1]),flush=True)
