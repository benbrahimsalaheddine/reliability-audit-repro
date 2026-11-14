# S-1.6: for each dataset, items in the release vs items used by the >=2-ratings-per-panel rule. Counting only; no resampling.
import json,pandas as pd,numpy as np
out=[]
def rep(name,total,kept,note=''): out.append(f"{name:28s} total {total:>7,}  used {kept:>7,}  dropped {total-kept:>6,} ({100*(total-kept)/total:.1f}%)  {note}")
# SummEval
n=0;ok=0
for l in open('./data/summeval/model_annotations.aligned.jsonl'):
    r=json.loads(l); n+=1
    ok+= (len(r['expert_annotations'])>=2 and len(r['turker_annotations'])>=2)
rep('SummEval',n,ok,'experts>=2 and crowd>=2')
# DICES-990 (Q_overall)
d=pd.read_csv('/tmp/dl/dices/990/diverse_safety_adversarial_dialog_990.csv'); tot=d.item_id.nunique()
q=d.dropna(subset=['Q_overall']); c=q.groupby(['item_id','rater_locale']).size().unstack()
loc=[x for x in c.columns][:2]; 
rep('DICES-990',tot,int((c.notna().all(axis=1)&(c>=2).all(axis=1)).sum()),'locales '+str(list(c.columns)))
# D3code pairs
R=pd.read_csv('d3repo/dataset/d3-ratings.csv'); H=pd.read_csv('d3repo/dataset/d3-raters.csv')
R=R.merge(H[['rater_id','Region']],on='rater_id'); tot=R.item_id.nunique(); R=R[R.rating_raw>=0]
k=R.groupby(['item_id','Region']).size().unstack()
import itertools
cnts=[int(((k[a]>=2)&(k[b]>=2)).sum()) for a,b in itertools.combinations(k.columns,2)]
rep('D3code (28 pairs: min..max)',tot,min(cnts),f'pair sizes {min(cnts)}..{max(cnts)}, median {int(np.median(cnts))}')
rep('D3code (rated in all 8)',tot,int((k.fillna(0)>=1).all(axis=1).sum()),'any rating in all regions')
# IRep
d=pd.read_csv('/tmp/irep/replication-dataset/IREP_data_release.csv'); d.columns=[c.strip('"') for c in d.columns]
pools=['Budapest','Mexico','KualaLumpur']; s=d[d.platform.isin(pools)]
cnt=s.groupby(['item_id','platform']).size().unstack()
rep('IRep (3 city pools)',s.item_id.nunique(),int((cnt[pools]>=2).all(axis=1).sum()),'>=2 raters in each pool')
# Jigsaw
j=pd.read_csv('/tmp/jig/specialized_rater_pools_data.csv',usecols=['id','toxic_score','rater_group']); tot=j.id.nunique()
j=j.dropna(subset=['toxic_score']); c=j.groupby(['id','rater_group']).size().unstack()
rep('Jigsaw SRP',tot,int((c.notna().all(axis=1)&(c>=2).all(axis=1)).sum()),'>=2 non-null in all 3 pools')
print('\n'.join(out))
