import json,re,numpy as np
from scipy import stats
A=json.load(open('/dev/null')) if False else [json.loads(l) for l in open('./data/summeval/model_annotations.aligned.jsonl')]
ann={(a['id'],a['model_id']):a for a in A}
num=re.compile(r'-?\d+(?:\.\d+)?')
for d,f in [('coherence','coh'),('consistency','con'),('fluency','flu'),('relevance','rel')]:
    R=json.load(open(f'ge/results/gpt4_{f}_detailed.json'))
    rows=[]
    for r in R:
        v=[float(num.search(x).group()) for x in r['all_responses'] if num.search(x)] if 'all_responses' in r else []
        if not v: continue
        a=ann.get((r['doc_id'],r['system_id']))
        if a is None: continue
        rows.append((r['doc_id'],np.mean(v),np.mean([x[d] for x in a['expert_annotations']]),np.mean([x[d] for x in a['turker_annotations']])))
    docs={}
    for x in rows: docs.setdefault(x[0],[]).append(x[1:])
    def per(col):
        o=[]
        for k,v in docs.items():
            v=np.array(v)
            if np.ptp(v[:,0])==0 or np.ptp(v[:,col])==0: continue
            o.append(stats.spearmanr(v[:,0],v[:,col])[0])
        return np.mean(o),len(o)
    print(d,len(rows),'expert %.3f (docs %d)  crowd %.3f'%(per(1)[0],per(1)[1],per(2)[0]))
