#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-4-22 下午5:37
# @Author  : Aries
# @Site    :
# @File    : change.py
# @Software: PyCharm

from django.http import JsonResponse
import time
from PIL import Image
from skimage.feature import hog
import numpy as np
import joblib

size = 128

model_path = 'core/model/'


# 变成灰度图片
def rgb2gray(im):
    gray = im[:, :, 0] * 0.2989 + im[:, :, 1] * 0.5870 + im[:, :, 2] * 0.1140
    return gray


# 提取特征
def extra_feat():
    get_feat_new()


# 只提取待测图片的特征
def get_feat_new(path):
        image = Image.open(path).resize((size, size), Image.ANTIALIAS)

        # 如果你的图片不是彩色的  可能需要把3改为-1
        image = np.reshape(image, (size, size, 3))
        gray = rgb2gray(image) / 255.0
        # 这句话根据你的尺寸改改
        fd = hog(gray, orientations=12, pixels_per_cell=[8, 8], cells_per_block=[4, 4], visualise=False,
                 transform_sqrt=True)
        return fd

# 训练和测试
def train_and_test(path):

    t0 = time.time()
    clf_type = 'LIN_SVM'
    num = 0
    total = 0
    if clf_type is 'LIN_SVM':
        # model名称自己修改一下
        clf = joblib.load(model_path + 'model')
        data_feat=get_feat_new(path)
        result = clf.predict(data_feat.reshape((1,-1)).astype(np.float64))
        t1 = time.time()
        print(result[0])
        print('耗时是 : %f' % (t1 - t0))
    return result[0]
def predict(request):
    label_map = {'1':'黄水仙',
                 '2':'春兰',
                 '3':'铃兰',
                 '4':'手参',
                 '5':'番红花',
                 '6':'德国鸢尾花',
                 '7':'毛百合',
                 '8': '川贝母',
                 '9': '向日葵',
                 '0': '葱莲'}
    name = request.GET['name']
    lable=''
    with open('test.txt') as f:
        for line in f.readlines():
            imagename=line.split(' ')[0]
            if(name == imagename):
                lable=line.split(' ')[1].strip('\n')
    path="static/test_image/"+name
    result=train_and_test(path).strip('\n')+''   # 训练并预测
    return JsonResponse({'data': label_map[result],'lable':label_map[lable]})

def all(request):
    data=[]
    with open('test.txt') as f:
        for line in f.readlines():
            imagename = line.split(' ')[0]
            data.append({"name":imagename})

    return JsonResponse({'data': data})