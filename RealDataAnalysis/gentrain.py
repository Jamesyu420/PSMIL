import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6"  # Set visible GPUs (7 A30 GPUs)

import torch
import torch.nn as nn
import numpy as np
import pickle
from utils import *

# Import relevant packages for the UNI model
from uni import get_encoder
from uni.downstream.extract_patch_features import extract_patch_features_from_dataloader
from uni.downstream.utils import concat_images

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   # Set GPU
device_ids = list(range(torch.cuda.device_count()))                       # Get all available GPU device IDs
print(device, device_ids)                                                 # Print current GPU and all GPUs

# Initialize memory cache tool and display CUDA info
mc = MemCache()
mc.mclean()
mc.show_cuda_info()
torch.cuda.empty_cache()                                                  # Clear the GPU cache

# Import relevant packages for loading the model
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision import transforms

local_dir = "../"
model = timm.create_model(
    "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
)
model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location=torch.device('cpu')), strict=True)

class VisionTransformerWithAvgPool(nn.Module):
    def __init__(self, original_model):
        super(VisionTransformerWithAvgPool, self).__init__()
        self.original_model = original_model
        self.avg_pool = nn.AdaptiveAvgPool1d(512)
    
    def forward(self, x):
        x = self.original_model(x)
        x = self.avg_pool(x)
        return x

# Create the custom model
model = VisionTransformerWithAvgPool(model)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
model = model.to(device)
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)
model.eval()

import openslide # Load WSI
print("IMPORT DONE")

# List tumor and normal files from the CAMELYON16 training dataset
tumor_file_list = os.listdir("/database/datasets/CAMELYON16/training/tumor")
no_tumor_file_list = os.listdir("/database/datasets/CAMELYON16/training/normal")
file_list = tumor_file_list + no_tumor_file_list  # Combine tumor and normal file lists
# Remove specific problematic files
remove_list = ['tumor_010.tif','tumor_015.tif','tumor_018.tif','tumor_020.tif','tumor_025.tif',
               'tumor_029.tif','tumor_033.tif','tumor_034.tif','tumor_044.tif','tumor_046.tif',
               'tumor_051.tif','tumor_054.tif','tumor_055.tif','tumor_056.tif','tumor_067.tif',
               'tumor_079.tif','tumor_085.tif','tumor_092.tif','tumor_095.tif','tumor_110.tif',
               'normal_86.tif','tumor_076.tif','tumor_078.tif','tumor_082.tif','normal_101.tif']
file_list = [item for item in file_list if item not in remove_list]

# Get the total number of files and create a label array: tumor samples = 1, normal samples = 0
N = len(file_list)           # Number of bags
Mplus = 100000               # Number of random sampling instances
M = 10000                    # Number of final instances 
p = 512                      # Dimensionality of embedded features
X = np.zeros([N,M,p])        # X: feature 
level = 7                    # WSI zoom level for extracting masks
patch_size = 2 ** (level+2)  # Patch size for extracted images
Y = np.zeros([N])            # Y: bag-level label
# Set tumor labels to 1 (files starting with 't' are tumor files)
Y[0:next((i for i, filename in enumerate(file_list) if not filename.startswith('t')), None)] = 1
thres = 75                   # How to define an instance - thresholding percentage

# Initialize lists for handling problematic files
problem_file = []
problem_index = []
sample_loc = np.zeros([N,M,2]) # Record the random sampling location
A = np.zeros([N,M])          # A: instance-level label

i = -1  # Initialize file index
import time
for file in file_list:
    start = time.time()  # Record the start time of file processing
    i += 1               # Increment file index
    set_seed(i)          # Random seed
    
    # Determine file paths based on the sample label (tumor or normal)
    if Y[i]==1:          # Tumor
        wsi_path = '/database/datasets/CAMELYON16/training/tumor/' + file
        json_path = '/mnt/GMM_for_MIL_codes/DataAnalysis_3/data/train/json_annotations/' + file[:-4] + '.json'
    else:                # Non-tumor
        wsi_path = '/database/datasets/CAMELYON16/training/normal/' + file

    
    slide = openslide.OpenSlide(wsi_path)    # Open the WSI file using OpenSlide
    w, h = slide.level_dimensions[level]     # Get width and height of the WSI at the specified level
    factor = slide.level_downsamples[level]  # Get the downsampling factor for the level
    
    mask_tumor = Masktumor(factor, json_path, Y[i], w, h) # Generate tumor mask
    
    # Generate the tissue mask
    img_RGB = np.array(slide.read_region((0, 0), level,
                                         slide.level_dimensions[level]))[:, :, :3]
    mask_tissue = MaskTissue(img_RGB, RGB_min=50) | mask_tumor 
    
    X_idcs, Y_idcs = np.where(mask_tissue)                                                     # Get coordinates of all tissue pixels
    centre_points = np.stack(np.vstack((X_idcs.T, Y_idcs.T)), axis=1)                          # Calculate center points
    sampled_points = centre_points[np.random.randint(centre_points.shape[0], size=Mplus), :]   # Randomly sample in tissue regions
    
    # Calculate the tumor area and tissue area for each sampled point
    if Y[i] == 1:
        mask_tumor_area = np.array([np.sum(mask_tumor[x-2:x+2, y-2:y+2]) / 16 for x, y in sampled_points])
    elif Y[i] == 0:
        mask_tumor_area = np.zeros([Mplus])
    mask_tissue_area = np.array([np.sum(mask_tissue[x-2:x+2, y-2:y+2]) / 16 for x, y in sampled_points])

    # Define instance labels
    inslbl = np.full(Mplus, -1)                                               # Set all labels to -1 by default - not considered
    inslbl[mask_tumor_area >= thres / 100] = 1                                # Set to be tumor if tumor area > thres
    inslbl[(mask_tumor_area < thres / 100) & (mask_tissue_area >= 0.75)] = 0  # Set to be tissue if tissue area > 75%

    # Scale the sampled points and convert to integer type
    sampled_points = (sampled_points * 2 ** level).astype(np.int32)
    center_points = sampled_points

    valid_indices = np.zeros([Mplus]).astype(bool)
    try:
        chosen_zeros = np.random.choice(np.where(inslbl >= 0)[0], M, replace=False)
    except:
        problem_file.append(file)
        problem_index.append(i)
        continue
    valid_indices[chosen_zeros] = True   # Sample those valid center points
    
    
    inslbl = inslbl[valid_indices]       # Define the instance label
    insimg = np.zeros([M,patch_size,patch_size,3])

    # Collect the (x,y) coordinates of the valid center points
    j = -1; k = -1
    valid_centers = [center for j, center in enumerate(center_points) if valid_indices[j]]
    x_coords = np.array([center[0] - patch_size / 2 for center in valid_centers], dtype=int)
    y_coords = np.array([center[1] - patch_size / 2 for center in valid_centers], dtype=int)

    sample_loc[i] = valid_centers        # Collect the sampled locations

    # Iterate over valid centers and corresponding coordinates
    for k, (x, y) in enumerate(zip(x_coords, y_coords)):
        img = slide.read_region((y, x), 0, (patch_size, patch_size))  # (width, height)
        insimg[k, :, :, :] = np.array(img)[:, :, :3] / 255.0          # Obtain the sub-images
    
    A[i,:] = inslbl
    insimg = np.transpose(insimg, (0, 3, 1, 2))
    
    mid = time.time()
    
    # Use mini-batch technique to obtain the embedded features

    insimg_tensor1 = torch.from_numpy(insimg[:M//8,:,:,:]).type(torch.FloatTensor).to(device)
    insimg_tensor1 = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(insimg_tensor1)
    with torch.inference_mode():
        feature_emb1 = model(insimg_tensor1) # Extracted features (torch.Tensor) 
    X[i,:M//8,:] = feature_emb1.cpu().numpy()
    del insimg_tensor1
    del feature_emb1
    
    insimg_tensor2 = torch.from_numpy(insimg[M//8:2*M//8,:,:,:]).type(torch.FloatTensor).to(device)
    insimg_tensor2 = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(insimg_tensor2)
    with torch.inference_mode():
        feature_emb2 = model(insimg_tensor2) # Extracted features (torch.Tensor) 
    X[i,M//8:2*M//8,:] = feature_emb2.cpu().numpy()
    del insimg_tensor2
    del feature_emb2
    
    insimg_tensor3 = torch.from_numpy(insimg[2*M//8:3*M//8,:,:,:]).type(torch.FloatTensor).to(device)
    insimg_tensor3 = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(insimg_tensor3)
    with torch.inference_mode():
        feature_emb3 = model(insimg_tensor3) # Extracted features (torch.Tensor) 
    X[i,2*M//8:3*M//8,:] = feature_emb3.cpu().numpy()
    del insimg_tensor3
    del feature_emb3
    
    insimg_tensor4 = torch.from_numpy(insimg[3*M//8:4*M//8,:,:,:]).type(torch.FloatTensor).to(device)
    insimg_tensor4 = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(insimg_tensor4)
    with torch.inference_mode():
        feature_emb4 = model(insimg_tensor4) # Extracted features (torch.Tensor) 
    X[i,3*M//8:4*M//8,:] = feature_emb4.cpu().numpy()
    del insimg_tensor4
    del feature_emb4
    
    insimg_tensor5 = torch.from_numpy(insimg[4*M//8:5*M//8,:,:,:]).type(torch.FloatTensor).to(device)
    insimg_tensor5 = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(insimg_tensor5)
    with torch.inference_mode():
        feature_emb5 = model(insimg_tensor5) # Extracted features (torch.Tensor) 
    X[i,4*M//8:5*M//8,:] = feature_emb5.cpu().numpy()
    del insimg_tensor5
    del feature_emb5
    
    insimg_tensor6 = torch.from_numpy(insimg[5*M//8:6*M//8,:,:,:]).type(torch.FloatTensor).to(device)
    insimg_tensor6 = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(insimg_tensor6)
    with torch.inference_mode():
        feature_emb6 = model(insimg_tensor6) # Extracted features (torch.Tensor) 
    X[i,5*M//8:6*M//8,:] = feature_emb6.cpu().numpy()
    del insimg_tensor6
    del feature_emb6
    
    insimg_tensor7 = torch.from_numpy(insimg[6*M//8:7*M//8,:,:,:]).type(torch.FloatTensor).to(device)
    insimg_tensor7 = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(insimg_tensor7)
    with torch.inference_mode():
        feature_emb7 = model(insimg_tensor7) # Extracted features (torch.Tensor) 
    X[i,6*M//8:7*M//8,:] = feature_emb7.cpu().numpy()
    del insimg_tensor7
    del feature_emb7
    
    insimg_tensor8 = torch.from_numpy(insimg[7*M//8:,:,:,:]).type(torch.FloatTensor).to(device)
    insimg_tensor8 = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(insimg_tensor8)
    with torch.inference_mode():
        feature_emb8 = model(insimg_tensor8) # Extracted features (torch.Tensor) 
    X[i,7*M//8:,:] = feature_emb8.cpu().numpy()
    del insimg_tensor8
    del feature_emb8

    end = time.time()
    print(i,mid-start, end-mid)
    
# Save to pkl

with open('/teams/WSIresult_1727165526/RealDataResult/' + str(patch_size) + '_' + str(thres) + '/train_XAY.pkl', 'wb') as file:
    pickle.dump({'X': X, 'A':A, 'Y':Y,
                 'file': file_list, 'sampleLoc': sample_loc,
                'problem_file': problem_file,
                'problem_index': problem_index}, file)
