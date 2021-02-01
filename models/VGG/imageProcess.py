import os

from PIL import Image

import cv2
import numpy as np
import shutil


def isGrayMap(img, threshold = 15):
    """
    入参：
    img：PIL读入的图像
    threshold：判断阈值，图片3个通道间差的方差均值小于阈值则判断为灰度图。
    阈值设置的越小，容忍出现彩色面积越小；设置的越大，那么就可以容忍出现一定面积的彩色，例如微博截图。
    如果阈值设置的过小，某些灰度图片会被漏检，这是因为某些黑白照片存在偏色，例如发黄的黑白老照片、
    噪声干扰导致灰度图不同通道间值出现偏差（理论上真正的灰度图是RGB三个通道的值完全相等或者只有一个通道，
    然而实际上各通道间像素值略微有偏差看起来仍是灰度图）
    出参：
    bool值
    """
    if len(img.getbands()) == 1:
        return True
    img1 = np.asarray(img.getchannel(channel=0), dtype=np.int16)
    img2 = np.asarray(img.getchannel(channel=1), dtype=np.int16)
    img3 = np.asarray(img.getchannel(channel=2), dtype=np.int16)
    diff1 = (img1 - img2).var()
    diff2 = (img2 - img3).var()
    diff3 = (img3 - img1).var()
    diff_sum = (diff1 + diff2 + diff3) / 3.0
    if diff_sum <= threshold:
        return True
    else:
        return False


train_path="/home/hitlab/PycharmProjects/CNN_VGG/deep-learning-for-image-processing-master/data_set/Age_data/val"
test_path="/home/hitlab/PycharmProjects/CNN_VGG/deep-learning-for-image-processing-master/pytorch_classification/Test3_vggnet/data/Validation"

i=0
objects=os.listdir(train_path)
heibai_path=((os.getcwd()+"/data/heibai_Vai"))
for obj in objects:

    test1=os.path.join(train_path,obj)
    next_obj=os.listdir(test1)
    for next_ob in next_obj:
        testt2=os.path.join(test1,next_ob)



        img1 = Image.open(testt2)
        flag=(isGrayMap(img1))

        if(flag==True):
            i=i+1
            shutil.move(testt2, heibai_path)
            print(obj)
            print(i)










