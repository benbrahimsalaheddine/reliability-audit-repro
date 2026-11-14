import json,re,numpy as np
from scipy import stats
A=[json.loads(l) for l in open('./data/summeval/model_annotations.aligned.jsonl')]
ann={(a['id'],a['model_id']):a for a in A}
num=re.compile(r'-?\d+(?:\.\d+)?')
rng=np.random.default_rng(0); B=10000
for d,f in [('coherence','coh'),('consistency','con'),('fluency','flu'),('relevance','rel')]:
    R=json.load(open('ge/results/gpt4_%s_detailed.json'%f))
    rows=[]
    for r in R:
        v=[float(num.search(x).group()) for x in r['all_responses'] if num.search(x)] if 'all_responses' in r else []
        if not v: continue
        a=ann.get((r['doc_id'],r['system_id']))
        if a is None: continue
        rows.append((r['doc_id'],np.mean(v),
                     np.mean([x[d] for x in a['expert_annotations']]),
                     np.mean([x[d] for x in a['turker_annotations']])))
    docs={}
    for x in rows: docs.setdefault(x[0],[]).append(x[1:])
    keys=list(docs.keys())
    sE={};sC={}
    for k in keys:
        v=np.array(docs[k])
        sE[k]=np.nan if np.ptp(v[:,0])==0 or np.ptp(v[:,1])==0 else stats.spearmanr(v[:,0],v[:,1])[0]
        sC[k]=np.nan if np.ptp(v[:,0])==0 or np.ptp(v[:,2])==0 else stats.spearmanr(v[:,0],v[:,2])[0]
    aE=np.array([sE[k] for k in keys]); aC=np.array([sC[k] for k in keys])
    ptE=np.nanmean(aE); ptC=np.nanmean(aC)
    n=len(keys); gaps=np.empty(B)
    for b in range(B):
        idx=rng.integers(0,n,n)
        gaps[b]=np.nanmean(aE[idx])-np.nanmean(aC[idx])
    q=np.percentile(gaps,[5,95])
    print('%-11s rows=%d docs=%d  E=%.3f C=%.3f  gap=%.3f  CI[%.2f,%.2f]'%(
        d,len(rows),n,ptE,ptC,ptE-ptC,q[0],q[1]))
