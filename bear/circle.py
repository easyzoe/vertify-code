#_*_ coding:utf-8 _*_  
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color,draw,transform,feature,util,io
from glassycom import rgim_get_sub_npim_byseq,gcom_getfilelist


pscode = io.imread('E:\\picdog\\midp\\1475800911.jpg', as_grey=True)

image = rgim_get_sub_npim_byseq(pscode,2)
 
 
#image = util.img_as_ubyte(data.coins()[0:95, 70:370]) #裁剪原图片
edges =feature.canny(image, sigma=3, low_threshold=10, high_threshold=50) #检测canny边缘

fig, (ax0,ax1) = plt.subplots(1,2, figsize=(8, 5))

ax0.imshow(edges, cmap=plt.cm.gray)  #显示canny边缘
ax0.set_title('original iamge')

hough_radii = np.arange(15, 30, 2)  #半径范围
hough_res =transform.hough_circle(edges, hough_radii)  #圆变换 

centers = []  #保存中心点坐标
accums = []   #累积值
radii = []    #半径

for radius, h in zip(hough_radii, hough_res):
    #每一个半径值，取出其中两个圆
    num_peaks = 2
    peaks =feature.peak_local_max(h, num_peaks=num_peaks) #取出峰值
    centers.extend(peaks)
    accums.extend(h[peaks[:, 0], peaks[:, 1]])
    radii.extend([radius] * num_peaks)

#画出最接近的5个圆
image = color.gray2rgb(image)
for idx in np.argsort(accums)[::-1][:5]:
    center_x, center_y = centers[idx]
    radius = radii[idx]
    cx, cy =draw.circle_perimeter(center_y, center_x, radius)
    image[cy, cx] = (255,0,0)

ax1.imshow(image)
ax1.set_title('detected image')
plt.show()