#!/usr/bin/env python
# encoding: utf-8
'''
@author: zxqyiyang
@contact: 632695399@qq.com
@file: evalua.py
@time: 2020/6/13 10:32
'''

"""
文件用于评估程序实现的融合效果与 ENVI 实现的融合效果
评估方法有： 图像熵值计算
"""

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

class calcultat_entropy():
    """
    定义计算图片熵的类
    """
    def __init__(self, img):
        self.img = img

    def rgb_to_gray(self):
        """
        定义函数：将图片转为灰度值
        :return:
        """
        return cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def statistics_grayscale_value(self, gray):
        """
        定义函数：统计相同灰度值的个数
        :param gray:
        :return: 相同灰度值出现的概率列表 P
        """
        p = np.zeros(256).astype(np.float32) # type 是 256 的 numpy 列表，用于统计 gray 中相同的数
        for i in range(gray.shape[0]):
            for j in range(gray.shape[1]):
                value = gray[i][j] # 取出 gray 的值，作为 p 的序号
                p[value] = p[value] + 1 # 相同的序号加一，从而统计得到相同灰度值的个数

        p = [pi/(gray.shape[0]*gray.shape[1]) for pi in p] # 计算相同灰度值的概率
        return p

    def calcultat_image_entropy(self, p):
        """
        定义函数：计算熵值
        :param p:
        :return:
        """
        entropy = 0
        for i in range(len(p)):
            if p[i] != 0 :
                entropy -= p[i]*math.log2(p[i])
        return entropy

class standard_deviation():
    """
    定义类：计算图像标准差，反应空间上的分辨率--->即标准差越大，分辨率越高
    """

    def __init__(self, img):
        self.img = img

    def rgb_to_gray(self):
        """
        定义函数：将RGB图片转为灰度值,B可以以不用转
        :return:
        """
        return cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def calcultat_standard_deviation(self, gray):
        gray = np.array(gray).astype(np.float32)
        return gray.std()

class spatial_frequency():
    """
    定义类：计算空间频率域值
    """
    def __init__(self, img):
        self.img = img

    def rgb_to_gray(self):
        """
        定义函数：将RGB图片转为灰度值,B可以以不用转
        :return:
        """
        return cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def calcultat_r_frequency(self, gray):
        rf = 0.0
        for i in range(gray.shape[0]):
            for j in range(gray.shape[1]-1):
                rf += (gray[i][j] - gray[i][j+1])**2
        return math.sqrt(rf/(gray.shape[0]*gray.shape[1]))

    def calcultat_c_frequency(self, gray):
        cf = 0.0
        for i in range(gray.shape[0]-1):
            for j in range(gray.shape[1]):
                cf += (gray[i][j] - gray[i+1][j])**2
        return math.sqrt(cf/(gray.shape[0]*gray.shape[1]))

    def calcultat_s_frequency(self, rf, cf):
        sf = math.sqrt(rf**2 + cf**2)
        return sf

class average_gradient():
    """
    定义类：计算空间频率域值
    """
    def __init__(self, img):
        self.img = img

    def rgb_to_gray(self):
        """
        定义函数：将RGB图片转为灰度值,B可以以不用转
        :return:
        """
        return cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
    def Gradient(self, gray):
        w = gray.shape[0]
        h = gray.shape[1]
        g = 0
        for i in range(w-1):
            for j in range(h-1):
                dx = gray[i][j] - gray[i][j+1]
                dy = gray[i][j] - gray[i+1][j]
                g += math.sqrt((dx**2 + dy**2)/2)
        g = g / (w*h)
        return g

def test_entropy():
    img = cv2.imread("D:/Desktop/IHS.png")
    print(img.shape)
    evalue = calcultat_entropy(img)
    gray = evalue.rgb_to_gray()
    plt.imshow(gray)
    plt.show()
    print(gray)
    value = evalue.statistics_grayscale_value(gray)
    print(value)
    entropy = evalue.calcultat_image_entropy(value)
    print(entropy)

def test_std():
    img = cv2.imread("D:/Desktop/IHS.png")
    print(img.shape)
    STD = standard_deviation(img)
    gray = STD.rgb_to_gray()
    result = STD.calcultat_standard_deviation(gray)
    print(result)

def test_sf():
    img = cv2.imread("D:/Desktop/IHS.png")
    print(img.shape)
    SF = spatial_frequency(img)
    gray = np.array(SF.rgb_to_gray()).astype(np.float32)
    print(gray.shape)
    rf = SF.calcultat_r_frequency(gray)
    cf = SF.calcultat_c_frequency(gray)
    result = SF.calcultat_s_frequency(rf, cf)
    print(result)
def test_gradient():
    img = cv2.imread("D:/Desktop/IHS.png")
    print(img.shape)
    gradient = average_gradient(img)
    gray = np.array(gradient.rgb_to_gray()).astype(np.float32)
    print(gray.shape)
    g = gradient.Gradient(gray)
    print(g)

if __name__ == '__main__':
    test_gradient()
    # test_std()
    # test_entropy()
    # test_sf()