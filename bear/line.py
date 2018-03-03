#_*_ coding:utf-8 _*_  
import skimage.transform as st
import matplotlib.pyplot as plt
from skimage import io,feature
import numpy as np; 
import os


from glassycom import rgim_get_sub_npim_byseq,gcom_getfilelist
#ʹ��Probabilistic Hough Transform.
#image = data.camera()
#
def  shfex_image_circle(afile,imgray):

 #   edges =feature.canny(image, sigma=3, low_threshold=10, high_threshold=50) #���canny��Ե
    edges = feature.canny(imgray)
    
    hough_radii = np.arange(15, 30, 2)  #�뾶��Χ
    hough_res =st.hough_circle(edges, hough_radii)  #Բ�任 

    centers = []  #�������ĵ�����
    accums = []   #�ۻ�ֵ
    radii = []    #�뾶

    for radius, h in zip(hough_radii, hough_res):
        #ÿһ���뾶ֵ��ȡ����������Բ
        num_peaks = 2
        peaks =feature.peak_local_max(h, num_peaks=num_peaks) #ȡ����ֵ
        centers.extend(peaks)
        accums.extend(h[peaks[:, 0], peaks[:, 1]])
        radii.extend([radius] * num_peaks)
    
    print len(centers)
    
    
def  shfex_image_line(afile,imgray) :

    edges = feature.canny(imgray)
   
    thlst = range(25,36,5)
    llenst = range(15,26,5)
   
    linenums = np.zeros([len(thlst),len(llenst)])
   
    for i in range(len(thlst)) :
       for j  in range(len(llenst)) :
          lines =  st.probabilistic_hough_line(edges, threshold= thlst[i], line_length=llenst[j],line_gap=3)
          linenums[i][ j ] = len(lines)
    linevector =linenums.reshape(linenums.shape[0]*linenums.shape[1])
    
    #print afile  , linevector 
    
    return linevector

def shfex_glassyelm_scan(glassypath):
    
    filelist = gcom_getfilelist(glassypath)
    
    vectors = []
    rowlen = len(filelist)
    for i in range(len(filelist) ):
    
        afile = filelist[i]
        imnp = io.imread(os.path.join(glassypath,afile),as_grey=True)
        linevector =  shfex_image_line(afile,imnp)
        if  len(vectors) == 0:
            vectors = np.zeros((rowlen, len(linevector) ))
        vectors[i] = linevector
        
    a = 0 
    emean = np.mean(vectors,axis =0 )
    estd  = np.std(vectors,axis =0 )
    
    print  emean
    print  estd
    
    return [emean,estd]


linespy_1 = shfex_glassyelm_scan('F:\\picdog\\glassy\\��ȫñ')
linespy_2 = shfex_glassyelm_scan('F:\\picdog\\glassy\\����')
        
# 1475800911.jpg  ����
    
pscode = io.imread('E:\\picdog\\midp\\1475800911.jpg', as_grey=True)

for i in range(8):
    image = rgim_get_sub_npim_byseq(pscode,i+1)
    vec = shfex_image_line('%d'%(i+1),image )
    
    shfex_image_circle('%d'%(i+1),image)
    print vec
    print 'mean  ****',np.mean(vec)
    print i+1, '           ...............        '
    print np.correlate( vec, linespy_1[0] )
    print np.correlate( vec, linespy_2[0] )
    print 'dis s'   ,np.linalg.norm(vec -   linespy_1[0] )
    
    print 'dis s  sss'   ,np.linalg.norm(vec -   linespy_2[0] )
    
    
#image = rgim_get_sub_npim_byseq(pscode,7)

#image = io.imread('F:\\picdog\\glassy\\��ȫñ\\1466330162_9068_im6.jpg', as_grey=True)

#shfex_glassyelm_scan('F:\\picdog\\glassy\\��ȫñ')

#feature.canny(image, sigma=2, low_threshold=1, high_threshold=25)
edges = feature.canny(image)
lines = st.probabilistic_hough_line(edges,threshold=30, line_length=25,line_gap=2)

print len(lines)

# ������ʾ����.
fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(16, 6))
plt.tight_layout()

#��ʾԭͼ��
ax0.imshow(image, plt.cm.gray)
ax0.set_title('Input image')
ax0.set_axis_off()

#��ʾcanny��Ե
ax1.imshow(edges, plt.cm.gray)
ax1.set_title('Canny edges')
ax1.set_axis_off()

#��plot���Ƴ����е�ֱ��
ax2.imshow(edges * 0)
for line in lines:
    p0, p1 = line
    ax2.plot((p0[0], p1[0]), (p0[1], p1[1]))
row2, col2 = image.shape
ax2.axis((0, col2, row2, 0))
ax2.set_title('Probabilistic Hough')
ax2.set_axis_off()
plt.show()