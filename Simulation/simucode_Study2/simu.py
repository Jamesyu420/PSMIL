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
import numpy as np
import scipy.stats as stats
import time

def DataGen(mu1, mu0, Sigma, N, M, pis, lambda0, seed=0):
    p = len(mu1)
    np.random.seed(seed)
    Y = 1*(np.random.rand(N)<lambda0) # 伯努利分布
    A = 1*(np.random.rand(N,M)<pis.reshape([-1,1]))
    A[Y == 0] =0 
    
    np.random.seed(seed)
    X1 = np.random.multivariate_normal(mu1, Sigma, np.sum(A)) # reshape对吗
    X0 = np.random.multivariate_normal(mu0, Sigma, np.sum(1-A))
    X = np.zeros([N*M, p]); 
    X[A.reshape(-1) == 1] = X1; 
    X[A.reshape(-1) == 0] = X0
    X = X.reshape([N,M,p])

    return {
        'pi': pis,
        'mu1': mu1,
        'mu0': mu0,
        'Sigma': Sigma,
        'X': X,
        'A': A,
        'Y': Y
    }

N = 100
M = 1000
p = 20

import pickle
with open('/mnt/GMM-MIL_REVISION/Simulation/para_est_KmeansInsInit.pkl', 'rb') as file:
    data = pickle.load(file)
mu1 = data['ins']['mu1'][:p]
mu0 = data['ins']['mu0'][:p]
Sigma = data['ins']['Sigma'][:p,:p]
pi0 = data['ins']['pi']
lambda0 = data['ins']['alpha']
Omega = np.linalg.inv(Sigma)


rep = 500
alphalist = np.array([-10, -6, -4, -2, -1, 0, 1, 4, 6, 10])
Nlist = alphalist



np.random.seed(0)

def simulation(r):
    data = DataGen(mu1, mu0, Sigma, N, M, pi0, lambda0, seed=r+10)
    X, Y, A = data['X'], data['Y'], data['A']
    ins = InsMLE(X, A, Y)
    bag = BagMLE(X, Y, mu1, mu0, Sigma, pi_ground=pi0, iter=1000, tol=1e-6)
    
    mu1_true = np.array(mu1.copy()); mu0_true = np.array(mu0.copy())
    prop = []
    ErrMu1_ins = []; ErrMu1_bag = []; ErrMu1_sub = []; ErrMu1_com = []; ErrMu1_com_flat = []
    ErrMu0_ins = []; ErrMu0_bag = []; ErrMu0_sub = []; ErrMu0_com = []; ErrMu0_com_flat = []
    ErrSigma_ins = []; ErrSigma_bag = []; ErrSigma_sub = []; ErrSigma_com = []; ErrSigma_com_flat = []
    ErrPi_ins = []; ErrPi_bag = []; ErrPi_sub = []; ErrPi_com = []; ErrPi_com_flat = []
    
    for alphaN in alphalist:
        com = SubMLE(X, Y, A, mu1, mu0, Sigma, pi0, bag, 
            alpha_n=alphaN, iter=1000, tol=1e-6, seed=0)

        ErrMu1_ins.append(L2norm(mu1_true, ins["mu1"]) / p)
        ErrMu1_bag.append(L2norm(mu1_true, np.array(bag["mu1"])) / p)
        ErrMu1_com.append(L2norm(mu1_true,np.array(com["mu1"])) / p)
        
        ErrMu0_ins.append(L2norm(mu0_true, ins["mu0"]) / p)
        ErrMu0_bag.append(L2norm(mu0_true, np.array(bag["mu0"])) / p)
        ErrMu0_com.append(L2norm(mu0_true,np.array(com["mu0"])) / p)
        
        ErrSigma_ins.append(np.linalg.norm(Omega - np.linalg.inv(ins["Sigma"]),ord ='fro')**2 / p**2)
        ErrSigma_bag.append(np.linalg.norm(Omega - np.linalg.inv(bag["Sigma"]),ord ='fro')**2 / p**2)
        ErrSigma_com.append(np.linalg.norm(Omega - np.linalg.inv(com["Sigma"]),ord ='fro')**2 / p**2)
        
        ErrPi_ins.append(L2norm(pi0, ins["pi"]))
        ErrPi_bag.append(L2norm(pi0, np.array(bag["pi"])))
        ErrPi_com.append(L2norm(pi0, com["pi"]))
        
        prop.append(com["prop"])

    results = (prop,ErrMu1_ins,ErrMu1_bag,ErrMu1_com,
              ErrMu0_ins,ErrMu0_bag,ErrMu0_com,
              ErrSigma_ins,ErrSigma_bag,ErrSigma_com,
              ErrPi_ins,ErrPi_bag,ErrPi_com)
    print(r)
    return results

import multiprocessing as mp

s1=time.time()
with mp.Pool(NUM_PROCESS) as pool:
    B = pool.map(simulation, range(rep))
s2=time.time()
print(s2-s1)

BB = np.array(B)
save_dir = f"/mnt/PSMIL_REVISION/Simulation/simucode_Study2/res/"
os.makedirs(save_dir, exist_ok=True)

np.savetxt(save_dir + "prop.csv", BB[:,0,:], delimiter=',')

np.savetxt(save_dir + "mu1_ins.csv", BB[:,1,:], delimiter=',')
np.savetxt(save_dir + "mu1_bag.csv", BB[:,2,:], delimiter=',')
np.savetxt(save_dir + "mu1_com.csv", BB[:,3,:], delimiter=',')

np.savetxt(save_dir + "mu0_ins.csv", BB[:,4,:], delimiter=',')
np.savetxt(save_dir + "mu0_bag.csv", BB[:,5,:], delimiter=',')
np.savetxt(save_dir + "mu0_com.csv", BB[:,6,:], delimiter=',')

np.savetxt(save_dir + "Sigma_ins.csv", BB[:,7,:], delimiter=',')
np.savetxt(save_dir + "Sigma_bag.csv", BB[:,8,:], delimiter=',')
np.savetxt(save_dir + "Sigma_com.csv", BB[:,9,:], delimiter=',')

np.savetxt(save_dir + "pi_ins.csv", BB[:,10,:], delimiter=',')
np.savetxt(save_dir + "pi_bag.csv", BB[:,11,:], delimiter=',')
np.savetxt(save_dir + "pi_com.csv", BB[:,12,:], delimiter=',')
