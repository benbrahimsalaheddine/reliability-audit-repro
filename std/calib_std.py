# Base-case calibration (2 panels, equal loading 1, n=200, k=4, sigma2=1) with the common constants: seed 0, N=1000 replications, B=10000 draws.
import sys; sys.path.insert(0,'.'); sys.path.insert(0,'./std')
from repro_config import SEED,N,B
from sim_general_case_v113 import run
tau2=float(sys.argv[1])
print('tau2',tau2,'N',N,'B',B,'rate',run((1,1),tau2,0,B=B,R=N,seed=SEED,mode='pair'),flush=True)
