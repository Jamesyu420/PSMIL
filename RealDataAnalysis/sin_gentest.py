import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-filename', type=str, default = 'test_001.tif')
    parser.add_argument('-num', type=int, default = 0)
    args = parser.parse_args()
    
    filename = args.filename
    i = args.num

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    import torch
    import torchvision
    import torch.nn as nn
    import torch.nn.functional as F

    from os.path import join as j_
    import pandas as pd
    import numpy as np

    # Import relevant packages for the UNI model
    from uni import get_encoder
    from uni.downstream.extract_patch_features import extract_patch_features_from_dataloader
    from uni.downstream.utils import concat_images

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   # Set GPU
    device_ids = list(range(torch.cuda.device_count()))                       # Get all available GPU device IDs
    print(device, device_ids)                                                 # Print current GPU and all GPUs

    from utils import *
    set_seed(i)                                                               # Set random seed
    
    # Initialize memory cache tool and display CUDA info
    mc = MemCache()
    mc.mclean()
    mc.show_cuda_info()
    torch.cuda.empty_cache()                                                  # Clear the GPU cache

    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    from torchvision import transforms

    local_dir = "../"

    import openslide

    print("IMPORT DONE")
    
    p = 512                                 # Dimensionality of embedded features
    level = 7                               # WSI zoom level for extracting masks
    patchsize = 2 ** (level+2)              # Patch size for extracted images
    thres = 75                              # How to define an instance - thresholding percentage

    output_path = '/teams/WSIresult_1727165526/RealDataResult/'+ str(patchsize) + '_' + str(thres) + '/testXAY/'
    
    block_size = (patchsize,patchsize)      # Size of block size

    import time
    import pickle

    # Load in the model
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
    
    json_list = os.listdir('/mnt/GMM_for_MIL_codes/DataAnalysis_3/data/test/json_annotations/')
    wsi_path = '/database/datasets/CAMELYON16/testing/images/' + filename
    slide = openslide.OpenSlide(wsi_path)    # Open the WSI file using OpenSlide
    w, h = slide.level_dimensions[level]     # Get width and height of the WSI at the specified level
    factor = slide.level_downsamples[level]  # Get the downsampling factor for the level

    json_path = filename[:-4] + '.json'
    if json_path in json_list:
        Y = 1  # Tumor bag
    else:
        Y = 0  # Non-tumor bag
    
    # Generate the tumor and tissue mask
    mask_tumor = Masktumor(factor, '/mnt/GMM_for_MIL_codes/DataAnalysis_3/data/test/json_annotations/' + json_path, Y, w, h)

    img_RGB = np.array(slide.read_region((0, 0),level,
                                         slide.level_dimensions[level]))[:,:,:3]
    mask_tissue = MaskTissue(img_RGB, RGB_min = 50) | mask_tumor
    mask_tumor = np.transpose(mask_tumor)
    mask_tissue = np.transpose(mask_tissue)

    img_size = slide.level_dimensions[0]                                     # WSI size
    ww, hh = (img_size[0]//patchsize, img_size[1]//patchsize)                # How many sub-images in height and width
    # Calculating the start and end indices to crop the center of the image
    start_x = (img_size[0] - patchsize * (img_size[0] // patchsize)) // 2
    start_y = (img_size[1] - patchsize * (img_size[1] // patchsize)) // 2
    end_x = start_x + patchsize * (img_size[0] // patchsize)
    end_y = start_y + patchsize * (img_size[1] // patchsize)

    print(i, "begin cutting")

    X = np.zeros([ww,hh,p])
    k = -1
    # Splitting the cropped image into blocks
    for x in range(start_x, end_x, patchsize):
        k += 1
        j = -1
        tmpimgs = np.zeros([hh,patchsize,patchsize,3])
        for y in range(start_y, end_y, patchsize):
            j += 1
            img = np.array(slide.read_region((x,y),0,block_size))[:,:,:3] / 255
            tmpimgs[j,:,:,:] = img

        tmpimgs = np.transpose(tmpimgs, (0, 3, 1, 2))
        insimg_tensor = torch.from_numpy(tmpimgs).type(torch.FloatTensor).to(device)
        insimg_tensor = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(insimg_tensor)

        with torch.inference_mode():
            feature_emb = model(insimg_tensor) # Extracted features (torch.Tensor)

        X[k,:,:] = feature_emb.cpu().numpy()   # Save to feature tensor
        del feature_emb

    print(i, "embedded")

    # Define the instance-level label
    A = np.zeros([ww,hh])
    k = -1
    for x in range(start_x, end_x, block_size[0]):
        k += 1
        j = -1
        for y in range(start_y, end_y, block_size[1]):
            j += 1
            rect_area = mask_tumor[int(x//factor-2):int(x//factor+2), int(y//factor-2):int(y//factor+2)]
            A[k,j] = np.sum(rect_area)

    N = 1; M = hh*ww
    
    A = (A > 4**2*thres/100).astype(np.int32)
    X = X.reshape([ww*hh,p])
    X = X.reshape(1,M,p)
    
    # Save to single pkl file
    with open(output_path + filename[:-4] + '.pkl', 'wb') as file:
        pickle.dump({'A': A, 'X': X}, file)

    del model
    del A
    del X
    del mask_tissue
    del mask_tumor