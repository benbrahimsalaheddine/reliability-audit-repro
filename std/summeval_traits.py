# SummEval: does the G-Eval gap (experts vs crowd) survive when (a) surface traits of the summary or (b) the other three dimensions (halo) are removed?
# Traits: length, extractiveness vs the source, mean word length, summarising system. Halo: each series (G-Eval, experts, crowd) loses its own other 3 dimensions.
# Seed 0, B = 10,000 document-bootstrap draws, 90% percentile interval.
import json,re,numpy as np
rng=np.random.default_rng(0); B=10000
ANN='./data/summeval/model_annotations.aligned.jsonl'; RES='./data/geval/results'
ann={(a['id'],a['model_id']):a for a in map(json.loads,open(ANN))}
num=re.compile(r'-?\d+(?:\.\d+)?'); tok=lambda s:re.findall(r"[a-z0-9']+",s.lower())
DIM=[('coherence','coh'),('consistency','con'),('fluency','flu'),('relevance','rel')]
def feats(summ,src):
    w=tok(summ); s=set(tok(src)); n=max(len(w),1); st=tok(src); sb=set(zip(st,st[1:])); bg=list(zip(w,w[1:]))
    return [np.log(n), np.mean([x in s for x in w]) if w else 0, np.mean([b in sb for b in bg]) if bg else 0, np.mean([len(x) for x in w]) if w else 0]
def resid(y,X):
    if X is None: return y
    X=np.column_stack([np.ones(len(y)),X]); return y-X@np.linalg.lstsq(X,y,rcond=None)[0]
G={}; info={}
for d,f in DIM:
    for r in json.load(open(f'{RES}/gpt4_{f}_detailed.json')):
        v=[float(num.search(x).group()) for x in r.get('all_responses',[]) if num.search(x)]
        k=(r['doc_id'],r['system_id']); a=ann.get(k)
        if not v or a is None: continue
        G.setdefault(k,{})[d]=np.mean(v); info[k]=(a,r['source'])
keys=sorted(k for k in G if len(G[k])==4)
docs=sorted({k[0] for k in keys}); idx={dd:np.array([i for i,k in enumerate(keys) if k[0]==dd]) for dd in docs}
g={d:np.array([G[k][d] for k in keys]) for d,_ in DIM}
e={d:np.array([np.mean([x[d] for x in info[k][0]['expert_annotations']]) for k in keys]) for d,_ in DIM}
c={d:np.array([np.mean([x[d] for x in info[k][0]['turker_annotations']]) for k in keys]) for d,_ in DIM}
F=np.array([feats(info[k][0]['decoded'],info[k][1]) for k in keys]); sy=sorted({k[1] for k in keys}); D=np.array([[k[1]==s for s in sy[1:]] for k in keys],float)
pick=[np.concatenate([idx[docs[p]] for p in rng.choice(len(docs),len(docs))]) for _ in range(B)]
print(f'n={len(keys)} summaries, docs={len(docs)}, systems={len(sy)}, B={B}',flush=True)
for d,_ in DIM:
    oth=[x for x,_ in DIM if x!=d]
    H=lambda s:np.column_stack([s[x] for x in oth])
    sets=[('none',None,None,None),('length',)+(F[:,:1],)*3,('length+extractive+wordlen',F,F,F),('system',D,D,D),('all traits',)+(np.column_stack([F,D]),)*3,
          ('other 3 dimensions (halo)',H(g),H(e),H(c)),('traits + halo',np.column_stack([F,D,H(g)]),np.column_stack([F,D,H(e)]),np.column_stack([F,D,H(c)]))]
    print(f'--- {d}',flush=True)
    for nm,Xg,Xe,Xc in sets:
        gr,er,cr=resid(g[d],Xg),resid(e[d],Xe),resid(c[d],Xc)
        rge=np.corrcoef(gr,er)[0,1]; rgc=np.corrcoef(gr,cr)[0,1]
        bs=np.array([[np.corrcoef(gr[i],er[i])[0,1],np.corrcoef(gr[i],cr[i])[0,1]] for i in pick])
        gap=bs[:,0]-bs[:,1]
        print(f'  removed {nm:27s} r(G-Eval,experts)={rge:.3f} [{np.percentile(bs[:,0],5):.3f},{np.percentile(bs[:,0],95):.3f}]  r(G-Eval,crowd)={rgc:.3f} [{np.percentile(bs[:,1],5):.3f},{np.percentile(bs[:,1],95):.3f}]  gap={rge-rgc:.3f} [{np.percentile(gap,5):.3f},{np.percentile(gap,95):.3f}]',flush=True)
