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

# --- CARE-style analysis with exactly 3 "judges": G-Eval (one LLM judge), the expert mean, the crowd mean. Per dimension.
import sys, ast, scipy as sp, cvxpy as cp, pandas as pd
src=open('/tmp/care_repo/src/src/pgm_tools.py').read(); ns={'cp':cp,'np':np,'sp':sp,'pd':pd}
for nd in ast.parse(src).body:
    if isinstance(nd,ast.FunctionDef) and nd.name in ('learn_structure','get_weights'): exec(compile(ast.Module([nd],[]),'pgm','exec'),ns)
rng=np.random.default_rng(0); B=10000
def onefactor(R):
    # R = 3x3 correlation matrix of (G-Eval, experts, crowd). lambda_i^2 = r_ij r_ik / r_jk
    gj,ge,gc=R[0,1],R[0,2],R[1,2]
    with np.errstate(all='ignore'):
        return np.array([gj*ge/gc, gj*gc/ge, ge*gc/gj])
print(f'n={len(keys)} summaries, docs={len(docs)}; B={B} document-bootstrap draws, seed 0, 90% intervals',flush=True)
for d,_ in DIM:
    X=np.column_stack([g[d],e[d],c[d]]); R=np.corrcoef(X.T); l2=onefactor(R)
    boot=np.array([onefactor(np.corrcoef(X[np.concatenate([idx[docs[p]] for p in rng.choice(len(docs),len(docs))])].T)) for _ in range(B)])
    ci=np.nanpercentile(boot,[5,95],axis=0); frac=np.mean(boot>1,axis=0)
    print(f'--- {d}: r(G,E)={R[0,1]:.3f} r(G,C)={R[0,2]:.3f} r(E,C)={R[1,2]:.3f}')
    for i,nm in enumerate(['G-Eval','experts','crowd']):
        print(f'    one-factor loading^2 {nm}: {l2[i]:.3f}  [{ci[0,i]:.3f}, {ci[1,i]:.3f}]   share of draws with loading^2 > 1: {frac[i]:.2f}')
    Sx,Lx=ns['learn_structure'](R,gamma=1,solver=cp.SCS,verbose=False,max_iters=10000)
    ev=np.sort(np.linalg.eigvalsh(0.5*(Lx+Lx.T)))[::-1]
    print(f'    CARE authors\' learn_structure on this 3x3, gamma=1: eigenvalues of the low-rank part = {np.round(ev,4)}',flush=True)
