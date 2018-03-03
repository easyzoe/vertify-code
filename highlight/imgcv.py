# -*- coding:utf-8 -*-

import glob,os,shutil

import numpy as np
from skimage.morphology import disk
from skimage.filters import rank
from skimage import feature
from skimage import color
import matplotlib.pyplot as plt
import cv2
from skimage import io

def test_img(afile):
    
    
    
    aimg = io.imread(afile)
    aimg = aimg[2:-2,3:-3,:]
    
    agray = color.rgb2gray(aimg)
    afgray = rank.median(agray,disk(1))
    
    eg = feature.canny(afgray)
    
    ret = np.where(eg)
    
    pt = np.array(  zip(ret[1],ret[0] ) )
    rect = cv2.minAreaRect(pt)
    print  os.path.basename(afile),"ange",rect[2]
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    im = cv2.drawContours(agray,[box],0,(0,0,255),2)  
    mask = np.zeros(agray.shape,np.float)    
    maskflag = cv2.drawContours(mask,[box],0,1,-1) 
    #cpimg = aimg.copy()
    #cpimg[maskflag == 0 ] = 0  
    
    plt.gray()
    plt.imshow(im)
    plt.show()
    ang = rect[2]
    if np.abs(rect[2]) > 1:
       rows,cols = agray.shape
       if np.abs(ang)  > 30:
          ram = cv2.getRotationMatrix2D((cols/2,rows/2),90+ang,1)
       else :
          ram = cv2.getRotationMatrix2D((cols/2,rows/2),ang,1)
       
       dst = cv2.warpAffine(aimg,ram,(cols,rows))
       maskdst = cv2.warpAffine(maskflag,ram,(cols,rows))
       dst[maskdst < 0.2 ] = 0 
       
       plt.imshow(dst)
       plt.show()
    
    a = 0 

def test_dir():
    path ="E:\\match_sample\\test\\500-1496203481-30\\"  
    #500-1496203490-40
    #500-1496203505-50
    filelist = glob.glob(r"%s*.jpg"%path)
    for afile in filelist:
          test_img(afile)
  
test_dir()