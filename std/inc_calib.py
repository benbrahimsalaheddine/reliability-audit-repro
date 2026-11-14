# Resumable base-case calibration. Chunks of 250 replications; each chunk saved at once. Seed of chunk c, cell t: 1000*t + c (all derived from the fixed seed 0 family). Total N = 5000, B = 10000.
import sys, json, os; sys.path.insert(0,'.'); sys.path.insert(0,'./std')
from repro_config import SEED,N,B
from sim_general_case_v113 import run
CH=250; tau2=float(sys.argv[1]); cell=0 if tau2==0 else 1
f=f'./std/ckpt/calib_{tau2}.jsonl'
done={json.loads(l)['c'] for l in open(f)} if os.path.exists(f) else set()
for c in range(N//CH):
    if c in done: continue
    rate=run((1,1),tau2,0,B=B,R=CH,seed=SEED+1000*cell+c,mode='pair')
    open(f,'a').write(json.dumps({'c':c,'flags':round(rate*CH)})+'\n')
rows=[json.loads(l) for l in open(f)]; fl=sum(r['flags'] for r in rows); n=CH*len({r['c'] for r in rows})
print('tau2',tau2,'chunks',len(rows),'N',n,'B',B,'rate %.4f'%(fl/n))
