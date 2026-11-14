# Per-panel tau^2 test (Eq. 3.8, two panels, item bootstrap, one-sided 5%) with the nine LLM judges treated as one panel. Dimension: argv[1]. Seed 0, B = 10,000.
# Panels: E = experts (3 raters), C = crowd (5 raters), J = nine judges (share of positive answers, 9 'raters'). Each panel mean is standardised over items (variance 1), noise scaled the same way, so a scale difference does not enter.
# tau_hat_p^2 = Var(ybar_p) - noise_p - Cov(ybar_p, ybar_q); noise_p = mean over items of (within-item variance of the raters / number of raters).
import json,csv,re,numpy as np
csv.field_size_limit(10**9); import sys
rng=np.random.default_rng(0); B=10000; D='./data/summeval'; d=sys.argv[1]
judges=['anthropic-claude-3-5-sonnet-20241022','anthropic-claude-3-haiku-20240307','deepseek-deepseek-chat','llama-Llama-3.3-70B-Instruct','mistral-mistral-large-latest','mistral-mistral-small-latest','openai-gpt-3.5-turbo','openai-gpt-4o-mini-2024-07-18','openai-o3-mini']
norm=lambda s:re.sub(r'\s+',' ',s.strip().lower())
ann={}
for l in open(f'{D}/model_annotations.aligned.jsonl'):
    r=json.loads(l); ann.setdefault((r['id'],norm(r['decoded'])),r)
rows=list(csv.DictReader(open(f'{D}/{d}/ratings.csv'))); keys=[(r['id'],norm(r['summary'])) for r in rows]
S=[]
for jn in judges:
    rt=np.array(json.load(open(f'{D}/{d}/{jn}.json'))['resp_table'],float); pos=rt[:,0].sum((1,2)); n=rt.sum((1,2,3)); S.append(np.where(n>0,pos/np.where(n>0,n,1),np.nan))
S=np.array(S).T; ok=~np.isnan(S).any(1); S=S[ok]; kk=[k for k,o in zip(keys,ok) if o]
Er=np.array([[a[d] for a in ann[k]['expert_annotations']] for k in kk],float); Cr=np.array([[a[d] for a in ann[k]['turker_annotations']] for k in kk],float)
panels={'E':Er,'C':Cr,'J':S}; n=len(kk); print(f'dimension {d}: n={n} items, raters E={Er.shape[1]} C={Cr.shape[1]} J={S.shape[1]}',flush=True)
mean={p:x.mean(1) for p,x in panels.items()}; noise={p:x.var(1,ddof=1).mean()/x.shape[1] for p,x in panels.items()}
sd={p:mean[p].std(ddof=1) for p in panels}; m={p:mean[p]/sd[p] for p in panels}; nz={p:noise[p]/sd[p]**2 for p in panels}
wv={p:panels[p].var(1,ddof=1)/panels[p].shape[1]/sd[p]**2 for p in panels}
def tau(p,q,idx):
    a=m[p][idx]; b=m[q][idx]; a=a-a.mean(-1,keepdims=True); b=b-b.mean(-1,keepdims=True)
    N=a.shape[-1]; return (a*a).sum(-1)/(N-1)-wv[p][idx].mean(-1)-(a*b).sum(-1)/(N-1)
pick=rng.integers(0,n,size=(B,n))
for p,q in [('E','C'),('J','E'),('J','C'),('E','J'),('C','J')]:
    t=tau(p,q,np.arange(n)); tb=tau(p,q,pick); lo=np.percentile(tb,5)
    print(f'panel {p} tested against {q}: tau_hat^2 = {t:+.3f}  5th percentile of the bootstrap = {lo:+.3f}  flag = {lo>0}   (noise share of Var: {nz[p]:.3f}, Cov({p},{q}) = {np.cov(m[p],m[q])[0,1]:+.3f})',flush=True)
