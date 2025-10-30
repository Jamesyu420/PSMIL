import os 

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

rep = 500
pilotList = [0,0.1,0.25,0.5,1,100]
alphaN = 0

M = 500
N = 1000
p = 50

np.random.seed(0)
import pickle
with open('/mnt/GMM-MIL_REVISION/Simulation/para_est_KmeansInsInit.pkl', 'rb') as file:
    data = pickle.load(file)
mu1 = data['ins']['mu1'][:50]
mu0 = data['ins']['mu0'][:50]
Sigma = data['ins']['Sigma'][:50,:50]
pi0 = data['ins']['pi']
lambda0 = data['ins']['alpha']
Omega = np.linalg.inv(Sigma)

def simulation(r):
    
    data = DataGen(mu1, mu0, Sigma, N, M, lambda0, pi0, seed=r+10)
    X, Y, A = data['X'], data['Y'], data['A']
    ins = InsMLE(X, A, Y)
    bag = BagMLE(X, Y, mu1, mu0, Sigma, pi_ground=pi0, iter=1000, tol=1e-6)
    
    mu1_true = np.array(mu1.copy()); mu0_true = np.array(mu0.copy())
    bag_true = {
            'mu1': mu1_true,
            'mu0': mu0_true,
            'Sigma': Sigma,
            'pi': pi0
    }
    
    ErrMu1_ins = []; ErrMu1_bag = []; ErrMu1_sub_small = []; ErrMu1_sub = []; ErrMu1_sub_flat = []; ErrMu1_sub_true = []
    ErrMu0_ins = []; ErrMu0_bag = []; ErrMu0_sub_small = []; ErrMu0_sub = []; ErrMu0_sub_flat = []; ErrMu0_sub_true = []
    ErrSigma_ins = []; ErrSigma_bag = []; ErrSigma_sub_small = [];ErrSigma_sub = [];ErrSigma_sub_flat = [];ErrSigma_sub_true = []
    ErrPi_ins = []; ErrPi_bag = []; ErrPi_sub_small = []; ErrPi_sub = []; ErrPi_sub_flat = []; ErrPi_sub_true = []
    
    k = -1
    for pilotprop in pilotList:
        k += 1
        
        if pilotprop > 1:
            alpha_n = sub_alphan(X, Y, est=bag_true, gamma_target=0.1)
            sub = SubMLE(X, Y, A, mu1, mu0, Sigma, pi0, bag_true, 
                              alpha_n=alpha_n, iter=1000, tol=1e-6, seed=0)
        elif pilotprop == 0:
            sub = SubMLEpure(X, Y, A, mu1, mu0, Sigma, pi0, bag, 0.1,
                             iter=1000, tol=1e-6, seed=0)
        else:
            np.random.seed(r*100 + k)
            select = np.random.rand(N) < pilotprop
            bag_small = BagMLE(X[select], Y[select], mu1, mu0, Sigma, pi_ground=pi0, iter=1000, tol=1e-6)
            alpha_n = sub_alphan(X, Y, est=bag_small, gamma_target=0.1)
            sub = SubMLE(X, Y, A, mu1, mu0, Sigma, pi0, bag_small, 
                               alpha_n=alpha_n, iter=1000, tol=1e-6, seed=0)

        ErrMu1_sub.append(L2norm(mu1_true,np.array(sub["mu1"])) / p)
        ErrMu0_sub.append(L2norm(mu0_true,np.array(sub["mu0"])) / p)
        ErrSigma_sub.append(np.linalg.norm(Omega - np.linalg.inv(sub["Sigma"]),ord ='fro')**2 / p**2)
        ErrPi_sub.append(L2norm(pi0, sub["pi"]))        

    results = (ErrMu1_sub, ErrMu0_sub, ErrSigma_sub, ErrPi_sub)
    return results

import multiprocessing as mp
K = 500
s1=time.time()
with mp.Pool(NUM_PROCESS) as pool:
    B = pool.map(simulation, range(K))
s2=time.time()
print(s2-s1)

BB = np.array(B)

save_dir = f"/mnt/PSMIL_REVISION/Simulation/simucode_Study4/res/"
os.makedirs(save_dir, exist_ok=True)

np.savetxt(save_dir + "mu1_sub.csv", BB[:,0,:], delimiter=',')
np.savetxt(save_dir + "mu0_sub.csv", BB[:,1,:], delimiter=',')
np.savetxt(save_dir + "Sigma_sub.csv", BB[:,2,:], delimiter=',')
np.savetxt(save_dir + "pi_sub.csv", BB[:,3,:], delimiter=',')
