# Per-panel and per-pair summary tables for D3code (8 regions) and CREHate (5 nations).
# Items rated by >=2 raters in every panel. Panel mean per item. Loadings from the triad identity (median over pairs).
import pandas as pd, numpy as np, itertools
def load_d3():
    R=pd.read_csv('d3repo/dataset/d3-ratings.csv'); H=pd.read_csv('d3repo/dataset/d3-raters.csv')
    R=R.merge(H[['rater_id','Region']],on='rater_id'); R=R[R.rating_raw>=0]
    return R.rename(columns={'item_id':'item','Region':'panel','rating_raw':'y'})[['item','panel','y']]
def load_cre():
    R=pd.read_csv('crehate/data/CREHate_raw_annotations.csv')
    return R.rename(columns={'ID':'item','nationality':'panel','annotation':'y'})[['item','panel','y']]
def run(name,R):
    g=R.groupby(['item','panel']).y
    M=g.mean().unstack(); K=g.count().unstack(); V=g.var(ddof=1).unstack()
    ok=(K>=2).all(axis=1); M,K,V=M[ok],K[ok],V[ok]; P=list(M.columns); n=len(M)
    C=np.cov(M.values.T,ddof=1)
    rows=[]
    for a,p in enumerate(P):
        trip=[C[a,b]*C[a,c]/C[b,c] for b,c in itertools.combinations([x for x in range(len(P)) if x!=a],2)]
        l2s=float(np.median(trip)); var=C[a,a]
        s2=float(V[p].mean()); noise=s2*float((1/K[p]).mean()); kbar=float(K[p].mean())
        rows.append(dict(dataset=name,panel=p,items=n,raters_per_item=round(kbar,2),var_mean=round(var,3),
          rho_k=round((var-noise)/var,3),
          l2s2=round(l2s,3),rho_star=round(l2s/var,3),tau2=round(var-l2s-noise,3)))
    T=pd.DataFrame(rows)
    # relative loading: sqrt(l2s2 / max l2s2)
    T['rel_loading']=np.sqrt(T.l2s2/T.l2s2.max()).round(2)
    pr=[]
    for (a,p),(b,q) in itertools.combinations(list(enumerate(P)),2):
        pr.append(dict(dataset=name,panel_1=p,panel_2=q,items=n,cov=round(C[a,b],3),r=round(C[a,b]/np.sqrt(C[a,a]*C[b,b]),3)))
    return T,pd.DataFrame(pr)
out=[run('D3code',load_d3()),run('CREHate',load_cre())]
A=pd.concat([o[0] for o in out]); B=pd.concat([o[1] for o in out])
print(A.to_string(index=False)); print(); print(B.to_string(index=False))
A.to_csv('table_panels_v121.csv',index=False); B.to_csv('table_pairs_v121.csv',index=False)
