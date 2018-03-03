# -*- coding:utf-8 -*-

import glob,os,shutil

import numpy as np

from color_histogram.io_util.image import loadRGB
from color_histogram.core.hist_1d import Hist1D
from color_histogram.core.hist_3d import Hist3D
import matplotlib.pyplot as plt


srcpath = "E:\\match_sample\\someone\\1494054534-9\\1494056641-0\\"

filelist = glob.glob(r'%s*.jpg'%srcpath)

for afile in filelist:
    image = loadRGB(afile)

    # 16 bins, Lab color space, target channel L ('Lab'[0])
    #hist1D = Hist1D(image, num_bins=16, color_space='rgb')
    hist3D = Hist3D(image, num_bins=64, color_space='rgb')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    hist3D.plot(ax)
    plt.show()