# Recompute of Table 2 (SummEval per dimension): one-way rho_1 and rho_k per panel, rho* (fidelity to the common signal),
# and tau^2 with 90 % item-bootstrap interval. Experts k=3, crowd k=5, 1,600 summaries. Seed 0, 1,000 item draws.
import json,sys,numpy as np
rng=np.random.default_rng(0)
A=[json.loads(l) for l in open(sys.argv[1])]; n=len(A)
def oneway(Y):
    k=Y.shape[1]; m=Y.mean(1); g=m.mean()
    msb=k*((m-g)**2).sum()/(len(Y)-1); msw=((Y-m[:,None])**2).sum()/(len(Y)*(k-1))
    return (msb-msw)/(msb+(k-1)*msw),(msb-msw)/msb
def est(Ey,Cy):
    mE,mC=Ey.mean(1),Cy.mean(1); c=np.cov(mE,mC)[0,1]
    sE=Ey.var(1,ddof=1).mean(); sC=Cy.var(1,ddof=1).mean(); vE,vC=mE.var(ddof=1),mC.var(ddof=1)
    return vE-c-sE/Ey.shape[1], vC-c-sC/Cy.shape[1], c/vE, c/vC
print('dimension   r1E   r1C   rkE   rkC   rho*E  rho*C   tauE [90% CI]        tauC [90% CI]   (rho*E, rho*C 90% CI)')
for d in ['coherence','consistency','fluency','relevance']:
    Ey=np.array([[x[d] for x in a['expert_annotations']] for a in A],float)
    Cy=np.array([[x[d] for x in a['turker_annotations']] for a in A],float)
    r1E,rkE=oneway(Ey); r1C,rkC=oneway(Cy); t=est(Ey,Cy)
    bs=np.array([est(Ey[i],Cy[i]) for i in (rng.integers(0,n,n) for _ in range(10000))])
    q=np.percentile(bs,[5,95],axis=0)
    print(f'{d:11s} {r1E:.3f} {r1C:.3f} {rkE:.3f} {rkC:.3f} {t[2]:.3f} {t[3]:.3f}  {t[0]:.2f} [{q[0,0]:.2f},{q[1,0]:.2f}]  {t[1]:.2f} [{q[0,1]:.2f},{q[1,1]:.2f}]  rho*E [{q[0,2]:.2f},{q[1,2]:.2f}] rho*C [{q[0,3]:.2f},{q[1,3]:.2f}]')
