#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-4-22 下午5:37
# @Author  : Aries
# @Site    : 
# @File    : change.py
# @Software: PyCharm
from PIL import Image
import numpy as np
import cv2
import os
#改变图片大小
def convertjpg( ):
    a = os.listdir('test_image')
    for img in a:
        img_path = 'test_image/'+ img
        save_path = 'test_image/' + img
        img=Image.open(img_path)
        try:
            new_img=img.resize((128,128),Image.BILINEAR)
            new_img.save(save_path)
        except Exception as e:
            print(e)

#前景提取
def cutimage(path,savepath):
    img = cv2.imread(path)
    mask = np.zeros(img.shape[:2],np.uint8)

    bgdModel = np.zeros((1,65),np.float64)

    fgdModel = np.zeros((1,65),np.float64)

    rect = (100,100,500,490)
#img：待分割的源图像，必须是8位3通道（CV_8UC3）图像，在处理的过程中不会被修改；
#mask：掩码图像，如果使用掩码进行初始化，那么mask保存初始化掩码信息；在执行分割的时候，
# 也可以将用户交互所设定的前景与背景保存到mask中，然后再传入,rabCut函数；在处理结束之后，mask中会保存结果
#rect：用于限定需要进行分割的图像范围，只有该矩形窗口内的图像才被认为可能是前景GCD_PR_FGD，矩形外的被认为是背景GCD_BGD。
# iterCount：迭代次数；
# mode：用于指示grabCut函数进行什么操作，可选的值有：
#                    GC_INIT_WITH_RECT（=0），用矩形窗初始化GrabCut；
#                    GC_INIT_WITH_MASK（=1），用掩码图像初始化GrabCut；
#                    GC_EVAL（=2），执行分割。
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')

    img2 = img*mask2[:,:,np.newaxis]
    cv2.imwrite(savepath, img2)
    cv2.waitKey(0)

def delimage():
    a = os.listdir('source')
    for list in range(0, len(a)):
        pth = os.listdir('source/' + a[list])
        for img in pth:
            img_path='source/' + a[list]+'/'+img
            save_path='image/' + a[list]+'/'+img
            cutimage(img_path,save_path)
delimage()