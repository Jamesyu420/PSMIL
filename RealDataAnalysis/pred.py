from utils import *
from xml.etree import ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import openslide
from PIL import Image
import imageio
import os
import cv2
import json
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from sklearn.metrics import roc_curve, roc_auc_score
import pickle
import numpy as np
import pandas as pd
from sklearn import metrics
import logging

logging.basicConfig(filename='/workplace/baichen/PSMIL_REVISION/Ours/RealData/pred_res/pred_1002.log', level=logging.INFO)

EVAL_KEYS = ['BMLE', 'IMLE', 'SMLE1', 'SMLE5', 'SMLE10', 'SMLE20', 'SMLE50']
#EVAL_KEYS = ['BMLE', 'IMLE', 'SMLE']
eval_cont = {
    k: {
        "Y_test": [],        # bag labels
        "bag_prob": [],      # bag probs
        "all_ins_true": [],  # overall instance labels
        "all_ins_pred": [],  # overall instance probs
        "ins_AUC": [], "ins_AUPRC": [],
        "ins_Prec": [], "ins_Recall": [], "ins_F1": []
    } for k in EVAL_KEYS
}

with open('/workplace/baichen/PSMIL_REVISION/Ours/RealData/para_est_KmeansInsInit.pkl', 'rb') as file:
    data = pickle.load(file)


bag = data['bag']
ins = data['ins']
com1 = data['com1']
com5 = data['com5']
com10 = data['com10']
com20 = data['com20']
com50 = data['com50']

mapp = {'BMLE': bag, 'IMLE': ins, 'SMLE1': com1, 'SMLE5': com5, 'SMLE10': com10, 'SMLE20': com20, 'SMLE50': com50}

testdata = pd.read_csv('/mnt/GMM_for_MIL_codes/DataAnalysis_3/data/test/reference.csv', header=None)
testdata.columns = ["filename", "class", "type", "size"]

normal_filename = testdata[testdata['class']=="Normal"]['filename'].tolist()
remove_list = ['test_114']
tumor_filename = testdata[testdata['class']=="Tumor"]['filename'].tolist()
tumor_filename = [item for item in tumor_filename if item not in remove_list]
filenames = tumor_filename + normal_filename

filelist = []
Ytest = []

bagpred = []; inspred = []; subpred = []
ths_list = [0.000001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9999, 0.99999, 0.999999]

for filename in filenames:
    try:
        with open('/workplace/baichen/PSMIL_REVISION/Ours/RealData/testData/' + filename +'.pkl', 'rb') as file:
            data = pickle.load(file)
        filelist.append(filename)
        X = data['X']
        A = data['A']
        Y = np.sum(A) > 1
        Ytest.append(Y)
    except:
        continue

    wsi_path = '/database/datasets/CAMELYON16/testing/images/' + filename + '.tif'
    json_path = '/mnt/GMM_for_MIL_codes/DataAnalysis_3/data/test/json_annotations/' + filename + '.json'
    
    level = 7 # at which WSI level to obtain the mask
    
    slide = openslide.OpenSlide(wsi_path)
    w, h = slide.level_dimensions[level]
    factor = slide.level_downsamples[level]# get the factor of level * e.g. level 6 is 2^6
    mask_tumor = Masktumor(factor, json_path, Y, w, h)

    img_RGB = np.array(slide.read_region((0, 0),level,
                                         slide.level_dimensions[level]))[:,:,:3]
    mask_tissue = MaskTissue(img_RGB, RGB_min = 50)
    mask_tumor = np.transpose(mask_tumor)
    mask_tissue = np.transpose(mask_tissue)

    N = X.shape[0]; M = X.shape[1]
    w, h = mask_tumor.shape[0]//4, mask_tumor.shape[1]//4

    block_tissue = mask_tissue.reshape((mask_tissue.shape[0]//4, 4, mask_tissue.shape[1]//4, 4)).sum(axis=(1, 3))
    mask_tissue_new = np.zeros((mask_tissue.shape[0]//4, mask_tissue.shape[1]//4), dtype = int)
    mask_tissue_new[block_tissue > 0.75 * 4**2] = 1

    block_tumor = mask_tumor.reshape((mask_tumor.shape[0]//4, 4, mask_tumor.shape[1]//4, 4)).sum(axis=(1, 3))
    mask_tumor_new = np.zeros((mask_tumor.shape[0]//4, mask_tumor.shape[1]//4), dtype = int)
    mask_tumor_new[block_tumor > 0.75 * 4**2] = 1

    mask_tumor_new = A
    A = mask_tumor_new.reshape(1, w*h)
    
    for method in EVAL_KEYS:    
        est = mapp[method]
        an = (np.dot(est['mu0'], np.linalg.solve(est['Sigma'], est['mu0'])) - np.dot(est['mu1'], np.linalg.solve(est['Sigma'], est['mu1'])))/2 + np.log(est['pi']/(1-est['pi']))
        betahat = np.linalg.solve(est['Sigma'], est['mu1']-est['mu0'])
        inexp = np.dot(X, betahat) + an; inexp[inexp>100] = 100
        probest = np.exp(inexp) / (1 + np.exp(inexp))
        probest_new = probest.reshape([w,h]).copy()
        probest_new[probest_new>1-np.finfo(np.float64).eps] = 1-np.finfo(np.float64).eps
        probest_new[mask_tissue_new==0] = 0
        result = probest_new
        eval_cont[method]['Y_test'].append(Y)
        eval_cont[method]['bag_prob'].append(1 - np.exp(np.sum(np.log(1 - result))))
        
        ins_pred = probest_new.reshape((w,h))[mask_tissue_new.astype(bool)]
        ins_true = A.reshape((w,h))[mask_tissue_new.astype(bool)]
        
        eval_cont[method]['all_ins_true'].append(ins_true)
        eval_cont[method]['all_ins_pred'].append(ins_pred)
        
        if Y == 1:
            if A.reshape((w,h))[mask_tissue_new.astype(bool)].sum() > 0:
                eval_cont[method]['ins_AUC'].append(roc_auc_score(ins_true, ins_pred))
                eval_cont[method]['ins_AUPRC'].append(average_precision_score(ins_true, ins_pred))
                eval_cont[method]['ins_Prec'].append(precision_score(ins_true, ins_pred > np.mean(ins_pred)))
                eval_cont[method]['ins_Recall'].append(recall_score(ins_true, ins_pred > np.mean(ins_pred)))
                eval_cont[method]['ins_F1'].append(f1_score(ins_true, ins_pred > np.mean(ins_pred)))

for key in EVAL_KEYS:
    Y_test = np.asarray(eval_cont[key]["Y_test"]).astype(int)
    bag_prob = np.asarray(eval_cont[key]["bag_prob"], dtype=float)

    all_ins_true = np.concatenate(eval_cont[key]["all_ins_true"])
    all_ins_pred = np.concatenate(eval_cont[key]["all_ins_pred"])

    bag_auc = safe_roc_auc(Y_test, bag_prob)
    bag_ap  = safe_ap(Y_test, bag_prob)
    bag_p, bag_r, bag_f = safe_prf(Y_test, bag_prob, thresh=0.9999999)
    
    ins_auc_med  = safe_stat(eval_cont[key]["ins_AUC"],   np.median)
    ins_auc_mean = safe_stat(eval_cont[key]["ins_AUC"],   np.mean)
    ins_ap_med   = safe_stat(eval_cont[key]["ins_AUPRC"], np.median)
    ins_ap_mean  = safe_stat(eval_cont[key]["ins_AUPRC"], np.mean)
    ins_p_med    = safe_stat(eval_cont[key]["ins_Prec"],  np.median)
    ins_p_mean   = safe_stat(eval_cont[key]["ins_Prec"],  np.mean)
    ins_r_med    = safe_stat(eval_cont[key]["ins_Recall"],np.median)
    ins_r_mean   = safe_stat(eval_cont[key]["ins_Recall"],np.mean)
    ins_f_med    = safe_stat(eval_cont[key]["ins_F1"],    np.median)
    ins_f_mean   = safe_stat(eval_cont[key]["ins_F1"],    np.mean)

    # overall instance
    ov_auc = safe_roc_auc(all_ins_true, all_ins_pred)
    ov_ap  = safe_ap(all_ins_true, all_ins_pred)
    ov_p, ov_r, ov_f = safe_prf(all_ins_true, all_ins_pred,
                                thresh=float(all_ins_true.mean()) if all_ins_true.size>0 else 0.5)

    msg = [
        f"[{key}] Bag AUC={np.nan if np.isnan(bag_auc) else bag_auc:.4f}  AUPRC={np.nan if np.isnan(bag_ap) else bag_ap:.4f}  "
        f"P={bag_p:.4f} R={bag_r:.4f} F1={bag_f:.4f}",
        f"[{key}] PerBag-Ins AUC: median={ins_auc_med:.4f} mean={ins_auc_mean:.4f}  "
        f"AUPRC: median={ins_ap_med:.4f} mean={ins_ap_mean:.4f}",
        f"[{key}] PerBag-Ins P/R/F1: P_med={ins_p_med:.4f} P_mean={ins_p_mean:.4f}  "
        f"R_med={ins_r_med:.4f} R_mean={ins_r_mean:.4f}  F1_med={ins_f_med:.4f} F1_mean={ins_f_mean:.4f}",
        f"[{key}] Overall-Ins AUC={np.nan if np.isnan(ov_auc) else ov_auc:.4f}  AUPRC={np.nan if np.isnan(ov_ap) else ov_ap:.4f}  "
        f"P={ov_p:.4f} R={ov_r:.4f} F1={ov_f:.4f}"
    ]
    for line in msg:
        print(line)
    print()