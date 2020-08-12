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

def b8_to_hsv(img):
    """
    定义函数： b8 转化为 hsv，img为 cv2 的数据格式
    :param img:
    :return:
    """
    rgb_b8 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    hsv_b8 = cv2.cvtColor(rgb_b8, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_b8)
    return v

def rgb_to_hsv(img):
    """
    定义函数： RGB 图像转化为 HSV 表示， img 为 cv2 的数据格式，即为（r，g，b） <<===>> (2000，2000，3)
    :param img:
    :return:
    """
    b, g, r = img[0,:,:], img[1,:,:], img[2,:,:]
    img = cv2.merge([b, g, r])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return hsv

def hsv_to_rgb(hsv_img):
    """
    定义函数：hsv 图像转为 rgb 图像，hsv_img 为 cv2 数据格式，即为（h, s, v） <<===>> (2000，2000，3)
    :param hsv_img:
    :return: 返回值为 numpy 形式的 rgb 图像
    """
    rgb = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    # r, g, b = cv2.split(rgb)
    # rgb_np = np.array((b, g, r)).astype(np.float)
    return rgb

def HSV(rgb, b8):
    """
    定义函数：开始 hsv 变换，即把 b8 的 v 去替换 rgb 中的 v 值
    :param rgb:
    :param b8:
    :return:
    """
    hsv_img = rgb_to_hsv(rgb)
    b8_v = b8_to_hsv(b8)
    width, height = b8_v.shape
    h, s, v = cv2.split(hsv_img)
    # for i in range(0, width):
    #     for j in range(0, height):
    #         v[i,j] = b8_v[i,j]
    v = b8_v
    hsv_img = cv2.merge([h, s, v])
    rgb_img = hsv_to_rgb(hsv_img)
    return rgb_img

def get_rgb_band_from_rgbtif(data, i):
    """
    定义函数：从多光谱图像中获取 R（3）、 G（2）、 B（1） 三个波段，并将其数值化
    :param data: 多光谱图像
    :param i: 波段数
    :return:
    """
    w, h = data.RasterXSize, data.RasterYSize
    return  data.GetRasterBand(i).ReadAsArray(0, 0, w, h).astype(np.float) # 数值化图像

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

def processing_tiff_data(file_1, file_2):
    rgb, b8 = gdal.Open(file_1), gdal.Open(file_2)
    print(f"RGB波段数为{rgb.RasterCount}，全色波段数为{b8.RasterCount}")  # 波段数
    print(f"RGB大小为{rgb.RasterXSize}*{rgb.RasterYSize}")
    print(f"全色波段大小为{b8.RasterXSize}*{b8.RasterYSize}")

    b8_w, b8_h = b8.RasterXSize, b8.RasterYSize # 获取全色波段的长宽高
    b8 =  b8.ReadAsArray(0, 0, b8_w, b8_h).astype(np.float32) # 数值化全色图像
    # print(b8)

    bandr = get_rgb_band_from_rgbtif(rgb, 3) # 获取图像中 r g b 三个波段
    bandg = get_rgb_band_from_rgbtif(rgb, 2)
    bandb = get_rgb_band_from_rgbtif(rgb, 1)

    resample_rgb = resample_the_rgb(b8_h, b8_w, bandr, bandg, bandb)# 重采样,使得 文件1 与 文件2 的大小一样
    print(f"多光谱图像重采样后的大小为{resample_rgb.shape}")

    rgb = HSV(resample_rgb, b8) #开始 hsv 变换
    result = Image.fromarray(result_to_image(rgb))
    enhnce = image_enhance(result, 1.5, 1.5)
    result = enhnce.image_brightened()
    # cv2.imwrite("D:\\Desktop\\hsv.png", result)
    result.save("D:\\Desktop\\hsv.png")

def main():
    rgb_file_path = "D:\\Desktop\\大三下\\空科综合实习内容一（GIS-RS）候选题目\\图像融合\\jhjz._caijian.tif"
    b8_filr_path = "D:\\Desktop\\大三下\\空科综合实习内容一（GIS-RS）候选题目\\图像融合\\really_b8_caijian.tif"
    processing_tiff_data(rgb_file_path, b8_filr_path)

if __name__ == '__main__':
    main()
