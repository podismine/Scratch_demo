#coding:utf8
import os
import pandas as pd
from torch.utils import data
import numpy as np
from sklearn.utils import shuffle
import nibabel as nib
import operator
import scipy.ndimage as nd
import random
from random import gauss
from transformations import rotation_matrix
from scipy.ndimage.interpolation import map_coordinates
from dp_model.dp_utils import crop_center,label_smooth,generate_label
from prefetch_generator import BackgroundGenerator


def resize3d(image,new_shape,order=3):
    real_resize_factor = tuple(map(operator.truediv, new_shape, image.shape))
    return nd.zoom(image, real_resize_factor, order=order)

def z_score(x, axis):
    x = np.array(x).astype(float)
    xr = np.rollaxis(x, axis=axis)
    xr -= np.mean(x, axis=axis)
    xr /= np.std(x, axis=axis)
    return x

def make_rand_vector(dims):
    vec = [gauss(0, 1) for i in range(dims)]
    mag = sum(x**2 for x in vec) ** .5
    return [x/mag for x in vec]

def flip_sagital(img,prob):
    if random.random() < prob:
        for batch in range(img.shape[2]):
            img[:,:,batch] = np.flipud(img[:,:,batch])
    return img

def coordinateTransformWrapper(X_T1,maxDeg=0,maxShift=7.5,mirror_prob = 0.5):
    #X_T1 = flip_sagital(X_T1, mirror_prob)
    randomAngle = np.radians(maxDeg*2*(random.random()-0.5))
    unitVec = tuple(make_rand_vector(3))
    shiftVec = [maxShift*2*(random.random()-0.5),
                maxShift*2*(random.random()-0.5),
                maxShift*2*(random.random()-0.5)]
    X_T1 = coordinateTransform(X_T1,randomAngle,unitVec,shiftVec)
    return X_T1

def coordinateTransform(vol,randomAngle,unitVec,shiftVec,order=1,mode='constant'):
    #from transformations import rotation_matrix
    ax = (list(vol.shape))
    ax = [ ax[i] for i in [1,0,2]]
    coords=np.meshgrid(np.arange(ax[0]),np.arange(ax[1]),np.arange(ax[2]))

    # stack the meshgrid to position vectors, center them around 0 by substracting dim/2
    xyz=np.vstack([coords[0].reshape(-1)-float(ax[0])/2,     # x coordinate, centered
               coords[1].reshape(-1)-float(ax[1])/2,     # y coordinate, centered
               coords[2].reshape(-1)-float(ax[2])/2,     # z coordinate, centered
               np.ones((ax[0],ax[1],ax[2])).reshape(-1)])    # 1 for homogeneous coordinates
    
    # create transformation matrix
    mat=rotation_matrix(randomAngle,unitVec)

    # apply transformation
    transformed_xyz=np.dot(mat, xyz)

    # extract coordinates, don't use transformed_xyz[3,:] that's the homogeneous coordinate, always 1
    x=transformed_xyz[0,:]+float(ax[0])/2+shiftVec[0]
    y=transformed_xyz[1,:]+float(ax[1])/2+shiftVec[1]
    z=transformed_xyz[2,:]+float(ax[2])/2+shiftVec[2]
    x=x.reshape((ax[1],ax[0],ax[2]))
    y=y.reshape((ax[1],ax[0],ax[2]))
    z=z.reshape((ax[1],ax[0],ax[2]))
    new_xyz=[y,x,z]
    new_vol=map_coordinates(vol,new_xyz, order=order,mode=mode)
    return new_vol

from dp_model.dp_utils import num2vect,generate_label,generate_biolabel
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

import h5py

class CombinedData(data.Dataset):

    def __init__(self,csv,train=True, test=False, val = False):
        import pandas as pd
        self.train = train

        if csv.endswith(".csv"):
            df = pd.read_csv(csv)
            self.imgs = list(df['t1'])
            self.lbls = [float(f) for f in list(df['age'])]
            self.sexs = [float(f) for f in list(df['sex'])]
            self.dataset_len = len(self.imgs)
        elif csv.endswith(".h5"):
            with h5py.File(csv, 'r') as file:
                self.dataset_len = len(file["data"])
            f = h5py.File(csv,'r')
            self.imgs = f['data']
            self.lbls = f['age']
            self.sexs = f['sex']
            self.preload = True
        
    def __getitem__(self,index):
        '''
        一次返回一张图片的数据
        '''
        img = self.imgs[index]
        lbl = self.lbls[index]

        lbl_y, lbl_bc = generate_label(lbl, sigma = 2)

        sex = self.sexs[index]
        sex_arr = np.zeros(2)
        sex_arr[int(sex)] = 1
        sex_arr = sex_arr[..., np.newaxis, np.newaxis, np.newaxis]

        img = img / np.mean(img)
        if self.train:
            img = coordinateTransformWrapper(img,maxDeg=10,maxShift=5, mirror_prob = 0)

        img = img[np.newaxis,...]
        
        return img, lbl, sex_arr, lbl_y, lbl_bc
    
    def __len__(self):
        return len(self.imgs)
