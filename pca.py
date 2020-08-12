from osgeo import gdal, gdalconst
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import scipy.misc as smi
import warnings
from PIL import Image, ImageEnhance
from skimage import data, exposure, img_as_float
from skimage.color import rgb2gray

class image_enhance():
    """
    图像增强类：包括亮度和对比度
    """
    def __init__(self, img, brightness, contrast):
        self.img = img
        self.brightness = brightness
        self.contrast = contrast
    def image_brightened(self):
        enh_bri = ImageEnhance.Brightness(self.img)
        image_brightened = enh_bri.enhance(self.brightness)
        return image_brightened

    def image_contrasted(self):
        enh_con = ImageEnhance.Contrast(self.img)
        img_contrasted = enh_con.enhance(self.contrast)
        return img_contrasted

def result_to_image(img):#转换格式
    img = np.array(img)
    max_val, min_val = np.nanmax(img), np.nanmin(img)
    print(max_val, min_val)
    out = (img.astype(np.float) - min_val) / (max_val - min_val)#将像素值换成0~1的值
    out = out*255   #乘以255，像素值换成颜色值
    out = np.uint8(out)#utf-8编码格式转换
    return out

def resample_the_rgb(h, w, r, g, b):
    """
    定义重采样函数：按照全色波段的大小重新定义大小
    :param h:
    :param w:
    :param r:
    :param g:
    :param b:
    :return:
    """
    warnings.filterwarnings("ignore")
    r = smi.imresize(r, (h, w), interp='bicubic')
    g = smi.imresize(g, (h, w), interp='bicubic')
    b = smi.imresize(b, (h, w), interp='bicubic')
    resample_rgb = np.array((b, g, r)).astype(np.float32) #numpy数组
    return resample_rgb

def get_rgb_band_from_rgbtif(data, i):
    """
    定义函数：从多光谱图像中获取 R（3）、 G（2）、 B（1） 三个波段，并将其数值化
    :param data: 多光谱图像
    :param i: 波段数
    :return:
    """
    w, h = data.RasterXSize, data.RasterYSize
    return  data.GetRasterBand(i).ReadAsArray(0, 0, w, h).astype(np.float) # 数值化图像

def processing_tiff_data(file_1, file_2):
    rgb, b8 = gdal.Open(file_1), gdal.Open(file_2)
    b8_w, b8_h = b8.RasterXSize, b8.RasterYSize # 获取全色波段的长宽高
    b8 =  b8.ReadAsArray(0, 0, b8_w, b8_h).astype(np.float32) # 数值化全色图像
    bandr = get_rgb_band_from_rgbtif(rgb, 3) # 获取图像中 r g b 三个波段
    bandg = get_rgb_band_from_rgbtif(rgb, 2)
    bandb = get_rgb_band_from_rgbtif(rgb, 1)
    resample_rgb = resample_the_rgb(b8_h, b8_w, bandr, bandg, bandb)# 重采样,使得 文件1 与 文件2 的大小一样
    return resample_rgb, b8

def PCA(file_1, file_2):
    resample_rgb, b8 = processing_tiff_data(file_1, file_2)
    height, width = b8.shape
    # print(height, width)
    resample_rgb = resample_rgb.reshape((height*width, 3)) #将 RGB 图像变为 （2002*2000，3）的大小
    b8 = b8.reshape((height*width, 1))
    resample_rgb_mean = np.mean(resample_rgb, 0) # 按照第一维度(也就是括号里的0)求和，结果维度为（3，1）
    # print(resample_rgb_mean)
    resample_rgb_reduce_mean = resample_rgb - np.tile(resample_rgb_mean, (height*width, 1))
    covar = (np.matrix(resample_rgb_reduce_mean).T * np.matrix(resample_rgb_reduce_mean)) / (height*width)
    # print(covar.shape) #==>（3，3）
    value, vector = np.linalg.eig(covar)  # 求特征向量
    vector = np.fliplr(vector)  # 左右对调特征向量
    # print(vector.shape)
    resample_rgb_to_PC = np.array(np.matrix(resample_rgb) * np.matrix(vector))  # PCA正变换 ==>(2000*2000,3)
    resample_rgb_to_PC[:][0] = b8[:][0]

    resample_rgb = np.array(np.matrix(resample_rgb_to_PC) * np.linalg.inv(vector)).reshape(3,width, height)
    # print(result_to_image(resample_rgb))
    result = result_to_image(resample_rgb)
    result = cv2.merge([result[0][:][:], result[1][:][:], result[2][:][:]])
    result = Image.fromarray(result)
    enhance = image_enhance(result, 0.8, 1.5)  # 调用类
    image_brightened = enhance.image_brightened()
    image_brightened.show()
    result.save("D:/Desktop" + str("/PCA融合结果图.png"))

def main():
    rgb_file_path = "D:\\Desktop\\大三下\\空科综合实习内容一（GIS-RS）候选题目\\图像融合\\jhjz._caijian.tif"
    b_8_file_path = "D:\\Desktop\\大三下\\空科综合实习内容一（GIS-RS）候选题目\\图像融合\\really_b8_caijian.tif"
    PCA(rgb_file_path, b_8_file_path)

if __name__ == '__main__':
    main()