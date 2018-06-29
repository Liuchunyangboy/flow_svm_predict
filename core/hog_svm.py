#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-4-22 下午5:37
# @Author  : Aries
# @Site    :
# @File    : change.py
# @Software: PyCharm
import glob
import time
from PIL import Image
from skimage.feature import hog
import numpy as np
import os
import joblib
from sklearn import svm
import shutil
from django.http import JsonResponse
from django.shortcuts import render
label_map = {0:'0',
             1:'1',
             2:'2',
             3:'3',
             4:'4',
             5: '5',
             6: '6',
             7: '7',
             8: '8',
             9: '9'}
#训练集图片的位置
train_image_path = 'static/image/'
#测试集图片的位置
test_image_path = 'static/test_image/'
#训练集标签的位置
train_label_path = 'mydata.txt'
#测试集标签的位置
test_label_path = 'test.txt'

size = 128

train_feat_path = 'core/train/'
test_feat_path = 'core/test/'
model_path = 'core/model/'


#获得图片列表
def get_image_list(filePath,nameList):
    img_list = []
    for name in nameList:
        img_list.append(Image.open(filePath+name))
    return img_list

#提取特征并保存
def get_feat(image_list,name_list,label_list,savePath,size):
    i = 0
    for image in image_list:
        try:
            #如果是灰度图片  把3改为-1
            image = np.reshape(image, (size, size, 3))
        except:
            print(name_list[i])
            continue
        gray = rgb2gray(image)/255.0
        fd = hog(gray, orientations=12, pixels_per_cell=[8,8], cells_per_block=[4,4], visualise=False, transform_sqrt=True)
        fd = np.concatenate((fd, [label_list[i]]))
        fd_name = name_list[i]+'.feat'
        fd_path = os.path.join(savePath, fd_name)
        joblib.dump(fd, fd_path)
        i += 1
    print ("Test features are extracted and saved.")

#变成灰度图片
def rgb2gray(im):
    gray = im[:, :, 0]*0.2989+im[:, :, 1]*0.5870+im[:, :, 2]*0.1140
    return gray

#获得图片名称与对应的类别
def get_name_label(file_path):
    name_list = []
    label_list = []
    with open(file_path) as f:
        for line in f.readlines():
            name_list.append(line.split(' ')[0])
            label_list.append(line.split(' ')[1])
    return name_list,label_list


#提取特征
def extra_feat():
    train_name,train_label = get_name_label(train_label_path)
    test_name,test_label = get_name_label(test_label_path)

    train_image = get_image_list(train_image_path,train_name)
    test_image = get_image_list(test_image_path,test_name)
    get_feat(train_image,train_name,train_label,train_feat_path,size)
    get_feat(test_image,test_name,test_label,test_feat_path,size)


# 创建存放特征的文件夹
def mkdir():
    if not os.path.exists(train_feat_path):
        os.mkdir(train_feat_path)
    if not os.path.exists(test_feat_path):
        os.mkdir(test_feat_path)

#训练
def train(request):
    t0 = time.time()
    clf_type = 'LIN_SVM'
    fds = []
    labels = []
    need_extra_feat = 'y'
    if need_extra_feat == 'y':
        shutil.rmtree(train_feat_path)
        mkdir()  # 不存在文件夹就创建
        extra_feat()  # 获取特征并保存在文件夹

    if clf_type is 'LIN_SVM':
        for feat_path in glob.glob(os.path.join(train_feat_path, '*.feat')):
            data = joblib.load(feat_path)
            fds.append(data[:-1])
            labels.append(data[-1])
        print( "Training a Linear LinearSVM Classifier.")
        clf = svm.LinearSVC()
        clf.fit(fds, labels)

        # 下面的注释代码是保存模型的
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        joblib.dump(clf, model_path+'model')
        print("训练之后的模型存放在model文件夹中")
        t1 = time.time()-t0
        return JsonResponse({'data':t1 })

def test(request):
    t0 = time.time()
    num = 0
    total = 0
    clf = joblib.load(model_path+'model')
    result_list = []
    for feat_path in glob.glob(os.path.join(test_feat_path, '*.feat')):
        total += 1
        image_name = feat_path.split('/')[1].split('.feat')[0]
        data_test = joblib.load(feat_path)
        data_test_feat = data_test[:-1].reshape((1, -1)).astype(np.float64)
        result = clf.predict(data_test_feat)
        result_list.append(image_name + ' ' + label_map[int(result[0])] + '\n')

        if int(result[0]) == int(data_test[-1]):
            num += 1
    write_to_txt(result_list)
    rate = float(num) / total
    t1 = time.time()
    print('准确率是： %f' % rate)
    print('耗时是 : %f' % (t1 - t0))
    return JsonResponse({'data': rate})

def write_to_txt(list):
    with open('result.txt','w') as f:
        f.writelines(list)

def event(request):
    # return render(request,'core/index.html')
    return render(request,'view.html')
