import pandas as pd
import numpy as np
import math
import os
import shutil
import pickle
import json,torch
import re
import cv2
from tqdm import tqdm
from glob import glob
import shutil
from scipy.ndimage.morphology import binary_fill_holes, binary_opening, binary_dilation


def get_mask(im):
    # use a intensity threshold to roughly find the mask of the body
    th = 32000  # an approximate background intensity value
    mask = im > th
    mask = binary_opening(mask, structure=np.ones((7, 7)))  # roughly remove bed
    # mask = binary_dilation(mask)
    # mask = binary_fill_holes(mask, structure=np.ones((11,11)))  # fill parts like lung

    if mask.sum() == 0:  # maybe atypical intensity
        mask = im * 0 + 1
    return mask.astype(dtype=np.int32)


def get_range(mask, margin=0):
    idx = np.nonzero(mask)
    u = max(0, idx[0].min() - margin)
    d = min(mask.shape[0] - 1, idx[0].max() + margin)
    l = max(0, idx[1].min() - margin)
    r = min(mask.shape[1] - 1, idx[1].max() + margin)
    return u, d, l, r

def load_multislice_img_16bit_png(data_dir, imname, slice_intv, num_slice):
    data_cache = {}
    def _load_data(imname, delta=0):
        imname1 = get_slice_name(data_dir, imname, delta)
        if imname1 not in data_cache.keys():
            data_cache[imname1] = cv2.imread(os.path.join(data_dir, imname1), -1)
            assert data_cache[imname1] is not None, 'file reading error: '+ data_dir + '/'+ imname1
            # if data_cache[imname1] is None:
            #     print('file reading error:', imname1)
        return data_cache[imname1]
    

    im_cur = _load_data(imname)
    mask = get_mask(im_cur)
    c = get_range(mask, margin=0)
    im_cur = im_cur[c[0]:c[1] + 1, c[2]:c[3] + 1]
    #ims = [im_cur] * num_slice  
    #run a check if multiplication is required

    ims = [im_cur]
    # find neighboring slices of im_cure
    rel_pos = float(2.0) / slice_intv
    a = rel_pos - np.floor(rel_pos)
    b = np.ceil(rel_pos) - rel_pos
    if a == 0:  # required SLICE_INTV is a divisible to the actual slice_intv, don't need interpolation
        for p in range(int((num_slice-1)/2)):
            im_prev = _load_data(imname, - rel_pos )
            im_prev = im_prev[c[0]:c[1] + 1, c[2]:c[3] + 1]
            im_next = _load_data(imname, rel_pos )
            im_next = im_next[c[0]:c[1] + 1, c[2]:c[3] + 1]
            ims = [im_prev] + ims + [im_next]
        
    else:
        for p in range(int((num_slice-1)/2)):
            intv1 = rel_pos
            slice1 = _load_data(imname, - np.ceil(intv1))
            slice1 = slice1[c[0]:c[1] + 1, c[2]:c[3] + 1]
            slice2 = _load_data(imname, - np.floor(intv1))
            slice2 = slice2[c[0]:c[1] + 1, c[2]:c[3] + 1]
            im_prev = a * slice1 + b * slice2  # linear interpolation

            slice1 = _load_data(imname, np.ceil(intv1))
            slice1 = slice1[c[0]:c[1] + 1, c[2]:c[3] + 1]
            slice2 = _load_data(imname, np.floor(intv1))
            slice2 = slice2[c[0]:c[1] + 1, c[2]:c[3] + 1]
            im_next = a * slice1 + b * slice2

            ims = [im_prev] + ims + [im_next]

    ims = [im.astype(float) for im in ims]
    im = cv2.merge(ims)
    im = im.astype(np.float32, copy=False) - 32768  # there is an offset in the 16-bit png files, intensity - 32768 = Hounsfield unit
    #print(imname , im.shape)
    return im


def get_slice_name(data_dir, imname, delta=0):
    """Infer slice name with an offset"""
    if delta == 0:
        return imname
    delta = int(delta)

    slice_idx = int(imname[:-4])
    imname1 =  '%03d.png' % ( slice_idx + delta)
    

    # if the slice is not in the dataset, use its neighboring slice
    while not os.path.exists(os.path.join(data_dir, imname1)):
        # print('file not found:', imname1)
        #print(data_dir,imname1, 'not exist so using neighbour')
        delta -= np.sign(delta)
        imname1 = '%03d.png' % ( slice_idx + delta)
        if delta == 0:
            #print('delta reduced to zero while finding neighbour')
            break
    #print('using img slice',data_dir,imname1 )
    return imname1


def windowing(im, win):
    """scale intensity from win[0]~win[1] to float numbers in 0~255"""
    im1 = im.astype(float)
    im1 -= win[0]
    im1 /= win[1] - win[0]
    im1[im1 > 1] = 1
    im1[im1 < 0] = 0
    im1 *= 255
    return im1


def load_prep_img(data_dir, imname, slice_intv,im_scale, num_slice):
    """load volume, windowing, interpolate multiple slices, clip black border, resize according to spacing"""
    ig = load_multislice_img_16bit_png(data_dir, imname, slice_intv, num_slice)
    ig = windowing(ig, [-1024,3072])
    #ig1 = windowing(ig, [-173,275])
    #ig2 = windowing(ig, [-1495,485])
    #ig3 = windowing(ig, [-534,1426])
    
    
    #ig = np.concatenate([ig1,ig2,ig3,ig4,ig5], axis = 2)
    ig = cv2.resize(ig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    #ig = ig.astype('uint8')
    #print('shape for ig1 ang ig',ig1.shape,ig.shape)
    return ig
	
#np.save ('np_ig.npy',ig)
#root_dir_path = '/media/abhishek/Seagate Backup Plus Drive/DeepLesionDataset/Images_png'
#file_name = dataset_dict["file_name"].split('/')
#data_dir = os.path.join(root_dir_path, file_name[0])
#ig = load_prep_img(data_dir, file_name[1], slice_intv,im_scale, num_slice=9)
