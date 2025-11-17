import os
import numpy as np
import json
import xml.etree.ElementTree as ET
import gc
import cv2
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from scipy.ndimage import label
import torch
import random

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU

class MemCache:
    @staticmethod
    def byte2MB(bt):
        return round(bt / (1024 ** 2), 3)

    def __init__(self):
        self.dctn = {}
        self.max_reserved = 0
        self.max_allocate = 0

    def mclean(self):
        r0 = torch.cuda.memory_reserved(0)
        a0 = torch.cuda.memory_allocated(0)
        f0 = r0 - a0

        for key in list(self.dctn.keys()):
            del self.dctn[key]
        gc.collect()
        torch.cuda.empty_cache()

        r1 = torch.cuda.memory_reserved(0)
        a1 = torch.cuda.memory_allocated(0)
        f1 = r1 - a1

        print('Mem Free')
        print(f'Reserved  \t {MemCache.byte2MB(r1 - r0)}MB')
        print(f'Allocated \t {MemCache.byte2MB(a1 - a0)}MB')
        print(f'Free      \t {MemCache.byte2MB(f1 - f0)}MB')

    def __setitem__(self, key, value):
        self.dctn[key] = value
        self.max_reserved = max(self.max_reserved, torch.cuda.memory_reserved(0))
        self.max_allocate = max(self.max_allocate, torch.cuda.memory_allocated(0))

    def __getitem__(self, item):
        return self.dctn[item]

    def __delitem__(self, *keys):
        r0 = torch.cuda.memory_reserved(0)
        a0 = torch.cuda.memory_allocated(0)
        f0 = r0 - a0

        for key in keys:
            del self.dctn[key]

        r1 = torch.cuda.memory_reserved(0)
        a1 = torch.cuda.memory_allocated(0)
        f1 = r1 - a1

        print('Cuda Free')
        print(f'Reserved  \t {MemCache.byte2MB(r1 - r0)}MB')
        print(f'Allocated \t {MemCache.byte2MB(a1 - a0)}MB')
        print(f'Free      \t {MemCache.byte2MB(f1 - f0)}MB')

    def show_cuda_info(self):
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        f = r - a

        print('Cuda Info')
        print(f'Total     \t{MemCache.byte2MB(t)} MB')
        print(f'Reserved  \t{MemCache.byte2MB(r)} [{MemCache.byte2MB(self.max_reserved)}] MB')
        print(f'Allocated \t{MemCache.byte2MB(a)} [{MemCache.byte2MB(self.max_allocate)}] MB')
        print(f'Free      \t{MemCache.byte2MB(f)} MB')

def preserve_single_true_points(arr, num):
    labeled_array, num_features = label(arr)
    unique_labels, label_counts = np.unique(labeled_array, return_counts=True)

    for label_index, count in zip(unique_labels[1:], label_counts[1:]):
        if count <= num:
            arr[labeled_array == label_index] = 0

    return arr

def maxIOU(pred, ths_list, mask_tumor, mask_tissue):
    IOU = []
    w, h = mask_tumor.shape
    for ths in ths_list:
        pred_tumor = (pred>ths).reshape([w,h]) & mask_tissue
        IOU.append(np.sum(pred_tumor & mask_tumor) / np.sum(pred_tumor | mask_tumor))
    ths = IOU.index(max(IOU))
    return (pred>ths_list[ths]).reshape([w,h]) & mask_tissue, IOU[ths]

def Masktumor(factor, json_path, YY, w, h):
    mask_tumor = np.zeros((h, w)) # the init mask, and all the value is 0

    if YY==1:
        # mask tumor
        with open(json_path) as f:
            dicts = json.load(f)
        tumor_polygons = dicts['positive']
        for tumor_polygon in tumor_polygons:
            # plot a polygon
            vertices = np.array(tumor_polygon["vertices"]) / factor
            vertices = vertices.astype(np.int32)
            cv2.fillPoly(mask_tumor, [vertices], (255))
        mask_tumor = mask_tumor[:] > 127
    mask_tumor = mask_tumor.astype(bool)
    return mask_tumor

def MaskTissue(img_RGB, RGB_min = 50): # min value for RGB channel
    img_HSV = rgb2hsv(img_RGB)

    background_R = img_RGB[:, :, 0] > threshold_otsu(img_RGB[:, :, 0])
    background_G = img_RGB[:, :, 1] > threshold_otsu(img_RGB[:, :, 1])
    background_B = img_RGB[:, :, 2] > threshold_otsu(img_RGB[:, :, 2])
    tissue_RGB = np.logical_not(background_R & background_G & background_B)
    tissue_S = img_HSV[:, :, 1] > threshold_otsu(img_HSV[:, :, 1])
    min_R = img_RGB[:, :, 0] > RGB_min
    min_G = img_RGB[:, :, 1] > RGB_min
    min_B = img_RGB[:, :, 2] > RGB_min
    mask_tissue = tissue_S & tissue_RGB & min_R & min_G & min_B
    return mask_tissue

# Reset Keras Session
def reset_keras(GPU = "0"):
    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()

    try:
        del classifier # this is from global space - change this as you need
    except:
        pass

    print(gc.collect()) # if it's done something you should see a number being outputted

    # use the same config as you used to create the session
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = GPU
    set_session(tf.compat.v1.Session(config=config))

def camelyon16xml2json(inxml, outjson):
        """
        Convert an annotation of camelyon16 xml format into a json format.
        Arguments:
            inxml: string, path to the input camelyon16 xml format
            outjson: string, path to the output json format
        """
        root = ET.parse(inxml).getroot()
        annotations_tumor = \
            root.findall('./Annotations/Annotation[@PartOfGroup="Tumor"]')
        annotations_0 = \
            root.findall('./Annotations/Annotation[@PartOfGroup="_0"]')
        annotations_1 = \
            root.findall('./Annotations/Annotation[@PartOfGroup="_1"]')
        annotations_2 = \
            root.findall('./Annotations/Annotation[@PartOfGroup="_2"]')
        annotations_positive = \
            annotations_tumor + annotations_0 + annotations_1
        annotations_negative = annotations_2

        json_dict = {}
        json_dict['positive'] = []
        json_dict['negative'] = []

        for annotation in annotations_positive:
            X = list(map(lambda x: float(x.get('X')),
                     annotation.findall('./Coordinates/Coordinate')))
            Y = list(map(lambda x: float(x.get('Y')),
                     annotation.findall('./Coordinates/Coordinate')))
            vertices = np.round([X, Y]).astype(int).transpose().tolist()
            name = annotation.attrib['Name']
            json_dict['positive'].append({'name': name, 'vertices': vertices})

        for annotation in annotations_negative:
            X = list(map(lambda x: float(x.get('X')),
                     annotation.findall('./Coordinates/Coordinate')))
            Y = list(map(lambda x: float(x.get('Y')),
                     annotation.findall('./Coordinates/Coordinate')))
            vertices = np.round([X, Y]).astype(int).transpose().tolist()
            name = annotation.attrib['Name']
            json_dict['negative'].append({'name': name, 'vertices': vertices})

        with open(outjson, 'w') as f:
            json.dump(json_dict, f, indent=1)

def overlapping_area(rect1, rect2):
    """
    Calculate the overlapping area of two rectangles.
    Rectangles are defined by tuples (x1, y1, x2, y2), where (x1, y1) is the bottom left corner and (x2, y2) is the top right corner.
    """
    # Unpack the coordinates
    x1, y1, x2, y2 = rect1
    x3, y3, x4, y4 = rect2

    # Calculate the coordinates of the overlapping rectangle
    overlap_x1 = max(x1, x3)
    overlap_y1 = max(y1, y3)
    overlap_x2 = min(x2, x4)
    overlap_y2 = min(y2, y4)

    # Check if there is an overlap
    if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
        # Calculate the area of the overlapping rectangle
        return (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
    else:
        # No overlap
        return 0

def sample_points_in_rectangle_band(outer_rect, inner_rect, num_samples):
    """
    Optimized sampling of random points in the band between two concentric rectangles.
    """
    ox1, oy1, ox2, oy2 = outer_rect
    ix1, iy1, ix2, iy2 = inner_rect

    # Define the four bands (top, bottom, left, right)
    bands = [
        (ox1, iy2, ox2, oy2),  # Top band
        (ox1, oy1, ox2, iy1),  # Bottom band
        (ox1, iy1, ix1, iy2),  # Left band
        (ix2, iy1, ox2, iy2)   # Right band
    ]

    points = []
    for _ in range(num_samples):
        chosen_band = bands[np.random.randint(0, 4)]
        x = np.random.uniform(chosen_band[0], chosen_band[2])
        y = np.random.uniform(chosen_band[1], chosen_band[3])
        points.append((x, y))

    return points

def calrect(c, length):
    return (c[0] - length[0], c[1] - length[1],
            c[0] + length[0], c[1] + length[1])


import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score

def safe_roc_auc(y_true, y_score):
    y_true = np.asarray(y_true)
    if y_true.max() == y_true.min():
        return np.nan
    try:
        return roc_auc_score(y_true, y_score)
    except Exception:
        return np.nan

def safe_ap(y_true, y_score):
    y_true = np.asarray(y_true)
    if y_true.sum() == 0: 
        return np.nan
    try:
        return average_precision_score(y_true, y_score)
    except Exception:
        return np.nan

def safe_prf(y_true, y_score, thresh=None):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if thresh is None:
        thresh = float(y_true.mean()) 
    y_pred = (y_score > thresh).astype(int)
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f = f1_score(y_true, y_pred, zero_division=0)
    return p, r, f

def safe_stat(x, fn):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan
    return fn(x)

def bag_prob_from_instance(p_inst_masked):
    if p_inst_masked.size == 0:
        return 0.0
    p = np.clip(p_inst_masked, 0.0, 1.0)
    s = np.sum(np.log1p(-p))  
    val = 1.0 - np.exp(s)
    if val < 0:  
        val = 0.0
    if val > 1:
        val = 1.0
    return float(val)
