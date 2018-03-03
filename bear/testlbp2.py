# -*- coding: utf-8 -*-
from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data
from skimage.color import label2rgb,rgb2gray
import matplotlib.pyplot as plt
import numpy as np

from skimage.transform import resize  #(image, output_shape)
#img_as_ubyte
#from scipy.misc import imresize
from scipy import fftpack

from skimage import exposure

from glob import glob  



radius = 1
n_points = 8 * radius
METHOD='nri_uniform'

# 计算距离
def kullback_leibler_divergence(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    filt = np.logical_and(p != 0, q != 0)
    return np.sum(p[filt] * np.log2(p[filt] / q[filt]))
# lbp fexture
def lbp_hist(img):
    if len(img.shape) ==3 :
      image = rgb2gray(img)
    else:
     image = img
     
    lbp = local_binary_pattern(image, n_points, radius, METHOD)
    n_bins = lbp.max() + 1
    
    hist, _ = np.histogram(lbp, normed=True, bins=n_bins, range=(0, n_bins))
    return hist
    
    

def match(refs, img):
    best_score = 10
    best_name = None
    lbp = local_binary_pattern(img, n_points, radius, METHOD)
    n_bins = lbp.max() + 1
    hist, _ = np.histogram(lbp, normed=True, bins=n_bins, range=(0, n_bins))
    for name, ref in refs.items():
        ref_hist, _ = np.histogram(ref, normed=True, bins=n_bins,
                                   range=(0, n_bins))
        score = kullback_leibler_divergence(hist, ref_hist)
        if score < best_score:
            best_score = score
            best_name = name
    return best_name

def color_major(img):
    
    mgshp = img.shape
    
    if len(mgshp) != 3:
       return  []
       
    pixelnum = mgshp[0]*mgshp[1]
    
    flatimg = img.reshape([pixelnum,3])
    
    lip = 64; 
    pxth = 0.80;
    
    qua_num = 256/lip;
    np.linspace
    qua =np.linspace(0,qua_num-1,qua_num) *64
    
    
    b = flatimg.astype(float)
    d =   (  np.floor( flatimg/lip )  * lip );
    d = d.astype(int)
    
    hstnum = np.power(qua_num,3)
  
    qualist=np.zeros((hstnum,3),'int')
    
    n = 0;
    v = 0;
    m  =0;
    for k in range(hstnum)  :      
      qualist[k,0]=qua[v];
      qualist[k,1]=qua[m];
      qualist[k,2]=qua[n];
      
      n = n+1
      if n >= qua_num:
          m =m +1
          n = 0;    
          if m >= qua_num:
              v = v +1;
              m = 0;
              if v >= qua_num:
                  break
              
    u_a, i_a, i_u = np.unique(d.view(d.dtype.descr * d.shape[1]), return_index=True, return_inverse=True)            
    
    n_u = np.bitcount(i_u)  # 统计出来各种抽样颜色的值
    
    
#
def fex_rgbhist(img):
    
  if len(img.shape) !=3 :
     return []   
  rh,_= np.histogram(img[:,:,0],bins=64)
  gh,_= np.histogram(img[:,:,1],bins=64)
  bh,_= np.histogram(img[:,:,2],bins=64)
  
  rgbhist = np.append(rh,gh)
  rgbhist = np.append(rgbhist,bh)
  
  rgbhist = rgbhist/(1.0* np.sum(rgbhist) )
  
  return rgbhist
  
    
# phash 特征    
def phash(img) :
   hashFingerSize = 16 
   dctSize = 32
   
   if len(img.shape) ==3 :
      image = rgb2gray(img)
   else:
     image = img
   
   rz_image = resize(image,(dctSize,dctSize))
   
   dct_image = fftpack.dct(fftpack.dct(rz_image.T, norm='ortho').T, norm='ortho')
   
   fig = dct_image[:hashFingerSize,:hashFingerSize]
   
   fig  =fig.reshape((hashFingerSize*hashFingerSize,1))
   dct_subarr_fabs = np.fabs(fig)
   dct_subaverage = np.mean(dct_subarr_fabs)
   dct_subfinal = np.greater_equal(dct_subarr_fabs, dct_subaverage*np.ones(dct_subarr_fabs.shape))
   dct_subfinal = dct_subfinal.astype('int8')
   print dct_subfinal.T
   return dct_subfinal.T
   
# 明氏距离     
def mindistance(a,b)   :
    df = np.logical_xor(a, b)
    image_distance = np.count_nonzero(df)   
    print image_distance
    return image_distance
     
def lbp_test(afile,bfile) :
    
    ahist = lbp_hist(data.load(afile))
    bhist = lbp_hist(data.load(bfile))
    dis = kullback_leibler_divergence(ahist,bhist)
    
    print dis
    
    return dis

def  glob_dir_search(fextype,testfile):
     
     filelist = glob('D:\\abc\\pci\\*.jpg')
     
     testimg = data.load(testfile)
     
     testfex = fex_rgbhist(testimg)
     
     for  afile in filelist:
     
         afex = fex_rgbhist(data.load(afile))
         
         dis = kullback_leibler_divergence(testfex,afex)
         
         print afile,dis
         
     
     
     
glob_dir_search(1,'D:\\abc\\pci\\1466313474_1527_im6.jpg')  




    
lbp_test('D:\\abc\\image\\1475421669_5.jpg','D:\\abc\\image\\1475425936_8.jpg') 
    
brick = data.load('D:\\abc\\image\\1475421669_5.jpg')

color_major(brick)


ha = phash(brick)


colore = brick



brick = rgb2gray(brick)
grass = data.load('D:\\abc\\image\\1475422326_8.jpg')
hb =  phash(grass)

grass = rgb2gray(grass)
wall = data.load('D:\\abc\\image\\1475425936_8.jpg')
hc = phash(wall)

wall = rgb2gray(wall)


hist1 = lbp_hist(brick)
hist2 = lbp_hist(grass)
hist3 = lbp_hist(wall)

print kullback_leibler_divergence(hist1,hist2)
print kullback_leibler_divergence(hist1,hist3)
print kullback_leibler_divergence(hist1,hist3)
print kullback_leibler_divergence(hist3,hist1)



#print lbp_hist(grass)
#print lbp_hist(wall)


refs = {
    'brick': local_binary_pattern(brick, n_points, radius, METHOD),
    'grass': local_binary_pattern(grass, n_points, radius, METHOD),
    'wall': local_binary_pattern(wall, n_points, radius, METHOD)
}

# classify rotated textures
print('Rotated images matched against references using LBP:')
print('original: brick, rotated: 30deg, match result: ',
      match(refs, rotate(brick, angle=30, resize=False)))
print('original: brick, rotated: 70deg, match result: ',
      match(refs, rotate(brick, angle=70, resize=False)))
print('original: grass, rotated: 145deg, match result: ',
      match(refs, rotate(grass, angle=145, resize=False)))

# plot histograms of LBP of textures
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3,
                                                       figsize=(9, 6))
plt.gray()

ax1.imshow(brick)
ax1.axis('off')
hist(ax4, refs['brick'])
ax4.set_ylabel('Percentage')

ax2.imshow(grass)
ax2.axis('off')
hist(ax5, refs['grass'])
ax5.set_xlabel('Uniform LBP values')

ax3.imshow(wall)
ax3.axis('off')
hist(ax6, refs['wall'])

plt.show()