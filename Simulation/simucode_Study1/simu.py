import os # OMEGA version!

NUM_CPU = len(os.sched_getaffinity(0))
print(f'Number of CPUs: {NUM_CPU}')
NUM_THREADS = 1
os.environ["MKL_NUM_THREADS"]     = str(NUM_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(NUM_THREADS)
os.environ["OMP_NUM_THREADS"]     = str(NUM_THREADS)

NUM_PROCESS = int(NUM_CPU // NUM_THREADS) # 可用在mp.Pool(NUM_PROCESS)中
print(f'Number of PROCESSs: {NUM_PROCESS}')

from estimation import *
from generation import *
import numpy as np
import time

M = 1000
N = 100
p = 20

np.random.seed(0)
sigma2List = [0.5, 1, 4, 9]
import pickle
with open('/mnt/GMM-MIL_REVISION/Simulation/para_est_KmeansInsInit.pkl', 'rb') as file:
    data = pickle.load(file)
mu1 = data['ins']['mu1'][:p]
mu0 = data['ins']['mu0'][:p]
Sigma0 = data['ins']['Sigma'][:p,:p]
pi0 = data['ins']['pi']
lambda0 = data['ins']['alpha']
Omega0 = np.linalg.inv(Sigma0)

rep = 500

def simulation(r):
    # if only focus on mu.1
    ErrMu1_ins = []; ErrMu1_bag = []
    ErrMu0_ins = []; ErrMu0_bag = []
    ErrSigma_ins = []; ErrSigma_bag = []
    ErrPi_ins = []; ErrPi_bag = []
    
    for sigma2 in sigma2List:
        Sigma = Sigma0 * sigma2
        Omega = Omega0 / sigma2
        data = DataGen(mu1, mu0, Sigma, N, M, lambda0, pi0, seed=r+10)
        X, Y, A = data['X'], data['Y'], data['A']
        
        ins = InsMLE(X, A, Y)
        bag = BagMLE(X, Y, mu1, mu0, Sigma, pi_ground=pi0, iter=1000, tol=1e-6)
        
        mu1_true = np.array(mu1.copy()); mu0_true = np.array(mu0.copy())
        
        ErrMu1_ins.append(L2norm(mu1_true, ins["mu1"]) / p)
        ErrMu1_bag.append(L2norm(mu1_true, np.array(bag["mu1"])) / p)
        ErrMu0_ins.append(L2norm(mu0_true, ins["mu0"]) / p)
        ErrMu0_bag.append(L2norm(mu0_true, np.array(bag["mu0"])) / p)
        ErrSigma_ins.append(np.linalg.norm(Omega - np.linalg.inv(ins["Sigma"]),ord ='fro')**2 / p**2)
        ErrSigma_bag.append(np.linalg.norm(Omega - np.linalg.inv(bag["Sigma"]),ord ='fro')**2 / p**2)
        ErrPi_ins.append(L2norm(pi0, ins["pi"]))
        ErrPi_bag.append(L2norm(pi0, np.array(bag["pi"])))

    results = (ErrMu1_ins,ErrMu1_bag,
              ErrMu0_ins,ErrMu0_bag,
              ErrSigma_ins,ErrSigma_bag,
              ErrPi_ins,ErrPi_bag)
    print(r)
    return results

import multiprocessing as mp
s1=time.time()
with mp.Pool(NUM_PROCESS) as pool:
    B = pool.map(simulation, range(rep))
s2=time.time()
print(s2-s1)

BB = np.array(B)

save_dir = f"/mnt/PSMIL_REVISION/Simulation/simucode_Study1/res/"
os.makedirs(save_dir, exist_ok=True)

np.savetxt(save_dir + "ErrMu1_ins.csv", BB[:,0,:], delimiter=',')
np.savetxt(save_dir + "ErrMu1_bag.csv", BB[:,1,:], delimiter=',')
np.savetxt(save_dir + "ErrMu0_ins.csv", BB[:,2,:], delimiter=',')
np.savetxt(save_dir + "ErrMu0_bag.csv", BB[:,3,:], delimiter=',')
np.savetxt(save_dir + "ErrSigma_ins.csv", BB[:,4,:], delimiter=',')
np.savetxt(save_dir + "ErrSigma_bag.csv", BB[:,5,:], delimiter=',')
np.savetxt(save_dir + "ErrPi_ins.csv", BB[:,6,:], delimiter=',')
np.savetxt(save_dir + "ErrPi_bag.csv", BB[:,7,:], delimiter=',')

