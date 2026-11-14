# Seed study of the base-case calibration (2 panels, equal loading 1, n=200, k=4, sigma2=1, B=250, R=10000), power at tau2=0.3 and FPR at tau2=0.
# Same simulator as sim_general_case_v113.run. Usage: python3 seeds_calib_v121.py <seed> <tau2>
import sys; sys.path.insert(0,'.')
from sim_general_case_v113 import run
seed=int(sys.argv[1]); tau2=float(sys.argv[2])
print('seed',seed,'tau2',tau2,'R=10000 rate',run((1,1),tau2,0,R=10000,seed=seed,mode='pair'),flush=True)
