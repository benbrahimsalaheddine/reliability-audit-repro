import pandas as pd, numpy as np, itertools
R=pd.read_csv('d3repo/dataset/d3-ratings.csv'); H=pd.read_csv('d3repo/dataset/d3-raters.csv')
R=R.merge(H[['rater_id','Region']],on='rater_id'); R=R[R.rating_raw>=0]
regs=sorted(R.Region.unique()); print(regs)
g=R.groupby(['Region','item_id']).rating_raw
stat=g.agg(m='mean',v=lambda x: x.var(ddof=1) if len(x)>1 else np.nan,k='count').reset_index()
def panel_stats(items, reg):
    s=stat[(stat.Region==reg)&(stat.item_id.isin(items))].set_index('item_id').loc[items]
    return s
# pair analysis, equal loadings, full sample
def pair(reg_p,reg_q,boot=None,rng=None,loadratio=None):
    sp=stat[(stat.Region==reg_p)&(stat.k>=2)].set_index('item_id'); sq=stat[(stat.Region==reg_q)&(stat.k>=2)].set_index('item_id')
    items=sp.index.intersection(sq.index)
    if boot is not None: items=items[rng.integers(0,len(items),len(items))]
    a=sp.loc[items]; b=sq.loc[items]
    out={}
    c=np.cov(a.m.values,b.m.values,ddof=1)[0,1]
    res={}
    for nm,s in (('p',a),('q',b)):
        V=np.var(s.m.values,ddof=1); sig=np.nanmean(s.v.values); kb=np.mean(1/s.k.values)
        res[nm]=(V,1-sig*kb/V)
    return res,c,len(items)
def reverse(res,c,ratio=1.0):
    (Vp,rkp),(Vq,rkq)=res['p'],res['q']
    fp=ratio*c/Vp; fq=(1/ratio)*c/Vq
    return (rkp-rkq)*(fp-fq)<0, rkp-rkq, fp-fq
cnt=0; rows=[]
for p,q in itertools.combinations(regs,2):
    res,c,n=pair(p,q); rv,drk,df=reverse(res,c); cnt+=rv; rows.append((p,q,n,rv,drk,df))
print('equal loadings full sample reversals',cnt,'/28')
# all-8 subset
ids=None
for reg in regs:
    s=set(stat[(stat.Region==reg)&(stat.k>=2)].item_id)
    ids=s if ids is None else ids&s
ids=np.array(sorted(ids)); print('all-eight subset',len(ids))
M=np.vstack([stat[(stat.Region==r)&(stat.item_id.isin(ids))].set_index('item_id').loc[ids].m.values for r in regs])
W=np.vstack([stat[(stat.Region==r)&(stat.item_id.isin(ids))].set_index('item_id').loc[ids].v.values for r in regs])
K=np.vstack([stat[(stat.Region==r)&(stat.item_id.isin(ids))].set_index('item_id').loc[ids].k.values for r in regs])
def sub_stats(ix):
    m=M[:,ix]; C=np.cov(m,ddof=1); V=np.diag(C)
    sig=np.nanmean(W[:,ix],1); kb=np.mean(1/K[:,ix],1); rk=1-sig*kb/V
    P=len(regs); l2=np.zeros(P)
    for p in range(P):
        vals=[C[p,q]*C[p,r]/C[q,r] for q in range(P) for r in range(q+1,P) if q!=p and r!=p]
        l2[p]=np.median(vals)
    l=np.sqrt(np.maximum(l2,1e-12)); l=l/l.max()
    return C,V,rk,l2,l
def counts(ix):
    C,V,rk,l2,l=sub_stats(ix); P=len(regs); eq=0; fr=0; flags=[]
    for p,q in itertools.combinations(range(P),2):
        fe_p=C[p,q]/V[p]; fe_q=C[p,q]/V[q]
        eq+= (rk[p]-rk[q])*(fe_p-fe_q)<0
        ff_p=l2[p]/V[p]; ff_q=l2[q]/V[q]
        # free: fidelity = l_p^2 sigma_s^2 / V_p, with l^2 sigma^2 = triad product
        fr+= (rk[p]-rk[q])*(ff_p-ff_q)<0
        flags.append((rk[p]-rk[q])*(ff_p-ff_q)<0)
    return eq,fr,flags,l
eq,fr,flags,l=counts(np.arange(len(ids)))
print('subset equal',eq,'free point',fr,'weights',dict(zip(regs,np.round(l,2))))
rng=np.random.default_rng(0); B=10000; acc=np.zeros(28); 
for b in range(B):
    ix=rng.integers(0,len(ids),len(ids)); _,_,fl,_=counts(ix); acc+=np.array(fl)
print('free: pairs reversing in >=95% of resamples:',int((acc/B>=0.95).sum()),' >=90%:',int((acc/B>=0.90).sum()),' max share',acc.max()/B)

print('--- full-sample pairs with estimated loading ratio ---')
C,V,rk,l2,l=sub_stats(np.arange(len(ids)))
lw=dict(zip(regs,l)); n_rev=0; pts=[]
for p,q in itertools.combinations(regs,2):
    res,c,n=pair(p,q); ratio=lw[p]/lw[q]
    rv,drk,df=reverse(res,c,ratio); n_rev+=rv
    if drk<0: drk,df=-drk,-df
    pts.append((round(drk,3),round(df,3)))
print('general (estimated loadings) point reversals on full pair samples:',n_rev,'/28')
print(pts)
