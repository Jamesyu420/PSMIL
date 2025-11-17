from xml.etree import ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import os
import cv2
import json
import time 
import pickle
import logging
from sklearn.metrics import roc_curve, roc_auc_score
import PSMIL
from estimation import *

logging.basicConfig(filename='estKMins.log',
                    level=logging.INFO)

with open('/workplace/baichen/WSIresult_1727165526/RealDataResult/512_75/train_XAY.pkl', 'rb') as file:
    data = pickle.load(file)
X = data['X']; Y = data['Y']; A = data['A']

logging.info('Data loaded')

l = data['problem_index']
print("Problem:", len(l))

X = np.delete(X, l, axis=0)
A = np.delete(A, l, axis=0)
Y = np.delete(Y, l, axis=0)

print(X.shape, A.shape, Y.shape)

start = time.time()
ins = InsMLE(X, A, Y)
end = time.time()
logging.info("IMLE-time:" + str(end-start))

N = X.shape[0]; M = X.shape[1]; p = X.shape[2]

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, n_init='auto')
kmeans.fit(X.reshape(N*M,p))
ins_pred = kmeans.predict(X.reshape(N*M,p))
ins_pred = ins_pred.reshape([N,M])
ins_pred[(1-Y).astype(bool)] = 0
logging.info("KMeans-INS AUC:" + str(np.round(roc_auc_score(A[Y.astype(bool)].reshape(-1),
                                                  ins_pred[Y.astype(bool)].reshape(-1)),3)))
ins_pred_flat = ins_pred.flatten()
indices = np.arange(ins_pred.size)
sorted_indices = indices[np.argsort(ins_pred_flat)[::-1]]
top_indices = sorted_indices[:int(np.sum(A))]
ins_pred = np.zeros(ins_pred.shape)
ins_pred.ravel()[top_indices] = 1
logging.info("KMeans-INS AUC:" + str(np.round(roc_auc_score(A[Y.astype(bool)].reshape(-1),
                                                           ins_pred[Y.astype(bool)].reshape(-1)),3)))
init = InsMLE(X, ins_pred, Y)

mu0_init = init['mu0']
mu1_init = init['mu1']
Sigma_init = init['Sigma']

start = time.time()
bag = BagMLE(X, Y, mu1_init, mu0_init, Sigma_init, ins['pi'], iter=100, tol=1e-10)
end = time.time()
logging.info("BMLE-time:" + str(end-start))
logging.info("BMLE-iter:" + str(bag['iter']))
logging.info("BMLE-pi:" + str(bag['pi']))

start = time.time()
alpha_1 = sub_alphan(X, Y, bag, 0.01, iter=1000)
com1 = SubMLE(X, Y, A, bag["mu1"], bag["mu0"], bag["Sigma"], bag["pi"], bag, 
             alpha_n=alpha_1, iter=100, tol=1e-20, seed=0)
end = time.time()
logging.info("SMLE-1-time:" + str(end-start))
logging.info("SMLE-1-iter:" + str(com1['iter']))
logging.info("SMLE-1-pi:" + str(com1['pi']))
logging.info("SMLE-1-prop:" + str(com1['prop']))

start = time.time()
alpha_5 = sub_alphan(X, Y, bag, 0.05, iter=1000)
com5 = SubMLE(X, Y, A, bag["mu1"], bag["mu0"], bag["Sigma"], bag["pi"], bag, 
             alpha_n=alpha_5, iter=100, tol=1e-20, seed=0)
end = time.time()
logging.info("SMLE-5-time:" + str(end-start))
logging.info("SMLE-5-iter:" + str(com5['iter']))
logging.info("SMLE-5-pi:" + str(com5['pi']))
logging.info("SMLE-5-prop:" + str(com5['prop']))

start = time.time()
alpha_10 = sub_alphan(X, Y, bag, 0.1, iter=1000)
com10 = SubMLE(X, Y, A, bag["mu1"], bag["mu0"], bag["Sigma"], bag["pi"], bag, 
             alpha_n=alpha_10, iter=100, tol=1e-20, seed=0)
end = time.time()
logging.info("SMLE-10-time:" + str(end-start))
logging.info("SMLE-10-iter:" + str(com10['iter']))
logging.info("SMLE-10-pi:" + str(com10['pi']))
logging.info("SMLE-10-prop:" + str(com10['prop']))

start = time.time()
alpha_20 = sub_alphan(X, Y, bag, 0.2, iter=1000)
com20 = SubMLE(X, Y, A, bag["mu1"], bag["mu0"], bag["Sigma"], bag["pi"], bag, 
             alpha_n=alpha_20, iter=100, tol=1e-20, seed=0)
end = time.time()
logging.info("SMLE-20-time:" + str(end-start))
logging.info("SMLE-20-iter:" + str(com20['iter']))
logging.info("SMLE-20-pi:" + str(com20['pi']))
logging.info("SMLE-20-prop:" + str(com20['prop']))

start = time.time()
alpha_50 = sub_alphan(X, Y, bag, 0.5, iter=1000)
com50 = SubMLE(X, Y, A, bag["mu1"], bag["mu0"], bag["Sigma"], bag["pi"], bag, 
             alpha_n=alpha_50, iter=100, tol=1e-20, seed=0)
end = time.time()
logging.info("SMLE-50-time:" + str(end-start))
logging.info("SMLE-50-iter:" + str(com50['iter']))
logging.info("SMLE-50-pi:" + str(com50['pi']))
logging.info("SMLE-50-prop:" + str(com50['prop']))

with open('/workplace/baichen/PSMIL_REVISION/Ours/RealData/para_est_KmeansInsInit_1002.pkl', 'wb') as file:
    pickle.dump({'ins':ins, 'bag': bag, 'com1': com1, 'com5': com5, 'com10': com10, 'com20': com20, 'com50': com50}, file)
