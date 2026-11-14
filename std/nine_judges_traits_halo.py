# Nine LLM judges x 4 dimensions (SummEval, 200 summaries per dimension): does the expert-vs-crowd gap survive when surface traits or the other three dimensions (halo) are removed?
# Judge score = share of positive answers (as in the nine-judge table). Seed 0, B = 10,000 item-bootstrap draws, 90% percentile interval of the gap r(judge,experts) - r(judge,crowd).
import json,csv,re,numpy as np
csv.field_size_limit(10**9); rng=np.random.default_rng(0); B=10000; D='./data/summeval'
dims=['coherence','consistency','fluency','relevance']
judges=['anthropic-claude-3-5-sonnet-20241022','anthropic-claude-3-haiku-20240307','deepseek-deepseek-chat','llama-Llama-3.3-70B-Instruct','mistral-mistral-large-latest','mistral-mistral-small-latest','openai-gpt-3.5-turbo','openai-gpt-4o-mini-2024-07-18','openai-o3-mini']
norm=lambda s:re.sub(r'\s+',' ',s.strip().lower()); tok=lambda s:re.findall(r"[a-z0-9']+",s.lower())
ann={}
for l in open(f'{D}/model_annotations.aligned.jsonl'):
    r=json.loads(l); ann.setdefault((r['id'],norm(r['decoded'])),r)
rows={d:list(csv.DictReader(open(f'{D}/{d}/ratings.csv'))) for d in dims}
keys={d:[(r['id'],norm(r['summary'])) for r in rows[d]] for d in dims}
common=[k for k in keys['coherence'] if all(k in set(keys[d]) for d in dims)]
print('items per dimension:',{d:len(set(keys[d])) for d in dims},'common to all 4:',len(common),flush=True)
art={}
for d in dims:
    for r in rows[d]: art[(r['id'],norm(r['summary']))]=(r['article'],r['summary'])
def feats(k):
    a,s=art[k]; w=tok(s); sa=tok(a); st=set(sa); sb=set(zip(sa,sa[1:])); bg=list(zip(w,w[1:])); n=max(len(w),1)
    return [np.log(n),np.mean([x in st for x in w]) if w else 0,np.mean([b in sb for b in bg]) if bg else 0,np.mean([len(x) for x in w]) if w else 0]
F=np.array([feats(k) for k in common])
def jscore(jn,d):
    rt=np.array(json.load(open(f'{D}/{d}/{jn}.json'))['resp_table'],float); pos=rt[:,0].sum((1,2)); n=rt.sum((1,2,3))
    s=np.where(n>0,pos/np.where(n>0,n,1),np.nan); return {k:s[i] for i,k in enumerate(keys[d])}
E={d:np.array([np.mean([a[d] for a in ann[k]['expert_annotations']]) for k in common]) for d in dims}
C={d:np.array([np.mean([a[d] for a in ann[k]['turker_annotations']]) for k in common]) for d in dims}
def resid(y,X):
    if X is None: return y
    X=np.column_stack([np.ones(len(y)),X]); return y-X@np.linalg.lstsq(X,y,rcond=None)[0]
res={}
def cc(a,b,pk):
    A=a[pk]-a[pk].mean(1,keepdims=True); Bb=b[pk]-b[pk].mean(1,keepdims=True); return (A*Bb).sum(1)/np.sqrt((A*A).sum(1)*(Bb*Bb).sum(1))
for jn in judges:
    S={d:jscore(jn,d) for d in dims}
    J={d:np.array([S[d].get(k,np.nan) for k in common]) for d in dims}
    ok=np.all([~np.isnan(J[x]) for x in dims],axis=0); pk=rng.integers(0,ok.sum(),size=(B,ok.sum()))
    for d in dims:
        oth=[x for x in dims if x!=d]
        H=lambda s:np.column_stack([s[x][ok] for x in oth])
        sets=[('none',None,None,None),('traits',F[ok],F[ok],F[ok]),('halo',H(J),H(E),H(C)),('traits+halo',np.column_stack([F[ok],H(J)]),np.column_stack([F[ok],H(E)]),np.column_stack([F[ok],H(C)]))]
        for nm,Xj,Xe,Xc in sets:
            j_,e_,c_=resid(J[d][ok],Xj),resid(E[d][ok],Xe),resid(C[d][ok],Xc)
            rge=np.corrcoef(j_,e_)[0,1]; rgc=np.corrcoef(j_,c_)[0,1]
            gap=cc(j_,e_,pk)-cc(j_,c_,pk); lo,hi=np.percentile(gap,[5,95])
            res[(jn,d,nm)]=(rge,rgc,rge-rgc,lo,hi,int(ok.sum()))
            print(f'{d:11s} {jn:38s} {nm:12s} n={ok.sum():3d} r(experts)={rge:5.2f} r(crowd)={rgc:5.2f} gap={rge-rgc:5.2f} [{lo:5.2f},{hi:5.2f}]'+('' if lo>0 else '  (interval reaches 0)'),flush=True)
print('\nSUMMARY over 36 judge-dimension pairs')
for nm in ['none','traits','halo','traits+halo']:
    v=[res[(j,d,nm)] for j in judges for d in dims]
    print(f'{nm:12s} median gap {np.median([x[2] for x in v]):5.2f}  median r(experts) {np.median([x[0] for x in v]):5.2f}  median r(crowd) {np.median([x[1] for x in v]):5.2f}  pairs with gap interval above 0: {sum(x[3]>0 for x in v)}/36')
