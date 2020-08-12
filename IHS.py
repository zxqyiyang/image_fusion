from osgeo import gdal
import cv2
import numpy as np
import math
import warnings
import scipy.misc as smi
from PIL import Image

def result_to_image(img):#转换格式
    img = np.array(img)
    max_val, min_val = np.nanmax(img), np.nanmin(img)
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
    resample_rgb = np.array((r, g, b)).astype(np.float32) #numpy数组
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

def IHS(file_1, file_2):
    rgb, b8 = gdal.Open(file_1), gdal.Open(file_2) # 打开图像
    b8_w, b8_h = b8.RasterXSize, b8.RasterYSize
    b8 = np.array(b8.ReadAsArray(0, 0, b8_w, b8_h).astype(np.float32))
    # print(b8.shape)
    bandr = get_rgb_band_from_rgbtif(rgb, 3)
    bandg = get_rgb_band_from_rgbtif(rgb, 2)
    bandb = get_rgb_band_from_rgbtif(rgb, 1)
    resample_rgb = resample_the_rgb(b8_h, b8_w, bandr, bandg, bandb) #numpy数组r g b


    coefficient_rgb_to_ihs = np.array([[1/3,1/3,1/3],
                                       [-math.sqrt(2)/6,-math.sqrt(2)/6,2*math.sqrt(2)/6],
                                       [1/math.sqrt(2),-1/math.sqrt(2),0]]) # 定义转换矩阵
    coefficient_ihs_to_rgb = np.array([[1, -1 / math.sqrt(2), 1 / math.sqrt(2)],
                                       [1, -1 / math.sqrt(2), -1 / math.sqrt(2)],
                                       [1, math.sqrt(2), 0]])
    # b8_, rgb_ = np.zeros((3,1)), np.zeros((3,1))
    # new_rgb = np.zeros((3, b8_w, b8_h))
    print("start prosseing IHS")
    # for i in range(b8_w):
    #     for j in range(b8_h):
    #         b8_[0] = b8[i][j]
    #         b8_[1] = b8[i][j]
    #         b8_[2] = b8[i][j]
    #
    #         rgb_[0]= resample_rgb[0][i][j]
    #         rgb_[1]= resample_rgb[1][i][j]
    #         rgb_[2]= resample_rgb[2][i][j]
    #
    #         b8_to_ihs = np.dot(coefficient_rgb_to_ihs, b8_)
    #         print(f"系数{coefficient_rgb_to_ihs.shape},rgb：{b8_.shape}，结果：{b8_to_ihs.shape}")
            # rgb_to_ihs= np.dot(coefficient_rgb_to_ihs, rgb_)

            # rgb_to_ihs[0] = b8_to_ihs[0]
            #
            # ihs_to_rgb = np.dot(coefficient_hsv_to_rgb, rgb_to_ihs)
            #
            # new_rgb[0][i][j] = ihs_to_rgb[0]
            # new_rgb[1][i][j] = ihs_to_rgb[1]
            # new_rgb[2][i][j] = ihs_to_rgb[2]

    b8 = np.array((b8, b8, b8)).astype(np.float32)
    resample_rgb, b8 = resample_rgb.reshape((3, b8_h*b8_w)), b8.reshape((3, b8_h*b8_w))
    rgb_to_ihs = np.dot(coefficient_rgb_to_ihs, resample_rgb)
    b8_to_ihs = np.dot(coefficient_rgb_to_ihs, b8)

    rgb_to_ihs, b8_to_ihs = rgb_to_ihs.reshape((3, b8_h , b8_w)), b8_to_ihs.reshape((3, b8_w, b8_h))
    rgb_to_ihs[0][:][:]= b8_to_ihs[0][:][:]
    rgb_to_ihs = rgb_to_ihs.reshape((3, b8_h*b8_w))

    ihs_to_rgb = np.dot(coefficient_ihs_to_rgb, rgb_to_ihs)

    new_rgb = ihs_to_rgb.reshape((3, b8_w, b8_h))
    result = cv2.merge([new_rgb[2,:,:], new_rgb[1,:,:], new_rgb[0,:,:]])  # 融合三色道
    # print(result)
    result = result_to_image(result)
    result = Image.fromarray(result)
    result.show()
    result.save("D:\\Desktop\\IHS融合结果图34.png")
    # cv2.imwrite("D:\\Desktop\\IHS.png", result)
    # print("success to save hsv_image")
    # cv2.imshow("image", result)  # 显示图片，后面会讲解
    # cv2.waitKey(0)  # 等待按键
    # cv2.cv.SaveImage("D:\\Desktop\\IHS.png", result)
    # x = Image.fromarray(result)
    # x.show()

def main():
    rgb_file_path = "D:\\Desktop\\大三下\\空科综合实习内容一（GIS-RS）候选题目\\图像融合\\jhjz._caijian.tif"
    b8_filr_path = "D:\\Desktop\\大三下\\空科综合实习内容一（GIS-RS）候选题目\\图像融合\\really_b8_caijian.tif"
    IHS(rgb_file_path, b8_filr_path)

if __name__ == '__main__':
    main()