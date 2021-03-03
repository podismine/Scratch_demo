#coding:utf8
from config import opt
import os
import torch as t
#import models
from unet import unet,ResBlock,RecombinationBlock,SEBlock
from dataset import Mydata
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchnet import meter
from visualize import Visualizer
from tqdm import tqdm
from torch.nn import BCELoss
import warnings
from metric import *
import numpy as np
import math
import nibabel as nib
import logging
from resnet import *
import torch
from torch.utils.data.distributed import DistributedSampler
import os
import h5py
import pandas as pd
from sklearn.utils import shuffle

print("Let's use", torch.cuda.device_count(), "GPUs!")
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#opt.parse(None)
model = resnet34().to("cuda:1")

pretrained_model_obj = torch.load('./resnet_34.pth', map_location='cpu')
new_dict = {}
for key,value in pretrained_model_obj['state_dict'].items():
    new_key = key.replace('module.', '')
    new_dict[new_key] = value

new_model = resnet34()
new_model_dict = new_model.state_dict()
pretrained_dict = {key: value for key, value in new_dict.items() if key in new_model_dict.keys()}
new_model_dict.update(pretrained_dict)
new_model.load_state_dict(new_model_dict)

new_model = nn.DataParallel(model.cuda(), device_ids=[1,2, 3])
new_model.train()