import sys; sys.argv=['x']
exec(open('./d3code_equal_significant_v113.py').read().split("print('90%")[0])
import itertools
def run2(B,alpha,seed=0):
    rng=np.random.default_rng(seed); out={'two':0,'one':0}
    for p,q in itertools.combinations(regs,2):
        items=S[p].index.intersection(S[q].index); a=S[p].loc[items]; b=S[q].loc[items]
        d1,d2=diffs(a,b)
        if d1*d2>=0: continue
        bs=np.array([diffs(a.iloc[ix],b.iloc[ix]) for ix in (rng.integers(0,len(items),len(items)) for _ in range(B))])
        for name,(lo_q,hi_q) in {'two':(100*alpha/2,100*(1-alpha/2)),'one':(100*alpha,100*(1-alpha))}.items():
            ok=True
            for j in (0,1):
                lo,hi=np.percentile(bs[:,j],[lo_q,hi_q]); ok&=(lo>0 or hi<0)
            out[name]+=ok and np.sign(bs[:,0].mean())!=np.sign(bs[:,1].mean())
    return out
print(run2(4000,0.10/28))
