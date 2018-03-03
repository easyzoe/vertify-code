#_*_ coding:utf-8 _*_  
import skimage.transform as st
import matplotlib.pyplot as plt
from skimage import io,feature

from glassycom import rgim_get_sub_npim_byseq


def  shfex_image_line(imgray) :

   edges = feature.canny(imgray)
   
   thlst = range(10,41,5)
   llenst = range(10,26,5)
   
   linenums = np.zeros([len(thlst),len(llenst)])
   
   for i in thlst:
       for j  in llenst:
          lines =  st.probabilistic_hough_line(edges, threshold=i, line_length=j,line_gap=3)
          linenums[ thlst.index(i)][ llenst.index(j) ] = len(lines)
    
    print    linenums 
   