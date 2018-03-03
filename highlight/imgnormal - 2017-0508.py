# -*- coding:utf-8 -*-

import glob,os,shutil

import numpy as np

from skimage import io
from skimage.morphology import disk
from skimage.filters import rank
from skimage import feature
from skimage import morphology
from skimage import color,img_as_ubyte
from skimage import exposure,measure
from skimage.transform import rotate

from scipy.spatial import distance
import matplotlib.pyplot as plt

from  scipy.ndimage.morphology import binary_fill_holes

#self define function

#from hausdorff import hausdorff_distance
import time
from functools import wraps
import h5py
from hausdorff import hausdorff



fs_edge  = "edge"
fs_area  = "area"
fs_color = "color"
fs_hsv   = 'hsv'
fs_gray  = 'gray'
fs_lbp   = 'lbp'
fs_hu    = 'hu'
fs_hausdorff = 'hausdorff'
fs_center   = 'center'
fs_template = 'template'
fs_colorobj = 'colorobj'





def fn_timer(function):
     
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print ("Total time running %s: %s seconds" %
              (function.func_name, str(t1-t0))
              )
        return result
    return function_timer


def  compcenter(bwword):
     
    text = bwword
    sum_x=0.0;
    sum_y=0.0;
    area=0.0; 
    
    [height,width]= text.shape
    
    for i in range(height):
       for j in range(width):
          if text[i,j] == 1:
               sum_x = sum_x + i
               sum_y = sum_y + j
               area = area + 1 
               
    return np.array([(sum_y/area),(sum_x/area)])
    # y plot in head

#

def move_tocenter(img):
    
    center = np.round(compcenter(img))
    shape = img.shape
    ydelta =  int(   center[0] - int( np.round( shape[0]/2.0 )  ) )
    xdelta =  int(   center[1] - int( np.round( shape[1]/2.0 )  ) )
    xpad = np.zeros( ( np.abs(xdelta), shape[1] ), dtype = int )
    
    row_up = 0
    
    if xdelta > 0:
        eximg = np.row_stack( (img,xpad))
    else :
        eximg = np.row_stack( (xpad,img))
        row_up = 1
    shape = eximg.shape 
    
    ypad = np.zeros( (shape[0],np.abs(ydelta)),dtype = int )
    col_left = 0
    
    if ydelta > 0 :
        eximg = np.column_stack( (eximg,ypad ) )
    else :
        eximg = np.column_stack( (ypad,eximg ) )
        col_left = 1    
    shape = eximg.shape

    half = int ( np.round( shape[0]/2.0  )    )

    if row_up == 1 and col_left == 1:
       result = eximg[:img.shape[0],:img.shape[1]]
    elif  row_up == 1 and  col_left ==0 :
       result = eximg[:img.shape[0],-img.shape[1]:]     
    elif  row_up == 0 and  col_left ==1 :
       result = eximg[-img.shape[0]:,:img.shape[1]]     
    else:
       result = eximg[-img.shape[0]:,-img.shape[1]:]     
    
    return result

def edge_grow(aimg):
    aimg = rank.median(aimg,disk(1))
    edge = feature.canny(aimg,sigma =0.8 )
    return edge


def color_hist_obj(aimg,vex):
  
  obj = aimg[vex]
  
  rgbhgmf = lambda  img  : np.row_stack(( np.histogram(img[:,0] ,bins =64,range=(0,255),normed = True)[0],
                         np.histogram(img[:,1] ,bins =64,range=(0,255),normed = True)[0],
                         np.histogram(img[:,2] ,bins =64,range=(0,255),normed = True)[0],  
                            ) )  
  ahimg = rgbhgmf(obj).flatten()
 
  return ahimg

def color_hist_whole(img):
  
  
  
  rgbhgmf = lambda  img  : np.row_stack(( np.histogram(img[:,:,0] ,range=(0,255),bins =64,normed = True)[0],
                         np.histogram(img[:,:,1] ,range=(0,255),bins =64,normed = True)[0],
                         np.histogram(img[:,:,2] ,range=(0,255),bins =64,normed = True)[0],  
                            ) )  
  ahimg = rgbhgmf(img).flatten()
 
  return ahimg
  
# 进行目标区域选择后，再继续hist

def color_hsv_whole(aimg):
    hsv =  color.rgb2hsv(aimg)
    va = np.histogram(hsv[:,:,0],bins=128,range=(0,1),normed=True )[0]
    vb = np.histogram(hsv[:,:,1],bins=32, range=(0,1),normed=True )[0]
    vc = np.histogram(hsv[:,:,2],bins=32, range=(0,1),normed=True )[0]
    
    #va = np.histogram(hsv[:,:,0],bins=128 )[0]
    #vb = np.histogram(hsv[:,:,1],bins=32 )[0]
    #vc = np.histogram(hsv[:,:,2],bins=32 )[0]
    
    hvshist = np.concatenate((va,vb,vc))
    
    return hvshist
    
def color_hsv_objhist(aimg,vex):
    hsv =  color.rgb2hsv(aimg)
    objhsv = hsv[vex]
       
    va = np.histogram(objhsv[:,0],bins=128,range=(0,1),normed=True  )[0]
    vb = np.histogram(objhsv[:,1],bins=32, range=(0,1),normed=True  )[0]
    vc = np.histogram(objhsv[:,2],bins=32, range=(0,1),normed=True   )[0]
    
    hvshist = np.concatenate((va,vb,vc))
    
    return hvshist


    
#

def cmp_haufdorff(aimg,bimg):
    row, col  = np.where(aimg)
    apt = zip(row,col)
     
    row ,col = np.where(bimg)
    bpt  = zip(row,col)

    #dis = hausdorff_distance(apt,bpt)
    dis = hausdorff(np.array(apt,dtype='float64'),np.array(bpt,dtype='float64'))
    return dis

def cmp_area(ar,br):
    return np.abs(   ar - br *1.0  )  / np.min((ar,br) ) 
    

def edge_grow_save(feat_edge,afile,veximg,debugflag = 0):
    
    if debugflag != 1:
        return
    
  
    
    lspath = "%s\\edge"%(os.path.dirname(afile)) 
    if not os.path.exists(  lspath ) :
         os.mkdir( lspath ) 
    svname = os.path.join(lspath, os.path.basename(afile) )
    io.imsave(svname,feat_edge*255)
    
    # binary data
    lspath = "%s\\np"%(os.path.dirname(afile)) 
    if not os.path.exists(  lspath ) :
         os.mkdir( lspath ) 
    svname = os.path.join(lspath, os.path.basename(afile) )
    np.savetxt(svname,feat_edge,fmt='%d')
    
    lspath = "%s\\obj"%(os.path.dirname(afile)) 
    if not os.path.exists(  lspath ) :
         os.mkdir( lspath ) 
    svname = os.path.join(lspath, os.path.basename(afile) )
    io.imsave(svname,veximg)
    
    return   
    
   
def img_feature_grow(afile,debugflag=0):
   aimg = io.imread(afile)
   aimg = aimg[2:-2,3:-3,:]
   
   
   
   cpimg = aimg.copy()
   
   gray_img = color.rgb2gray(aimg)
   
   ers_aimg = rank.median(gray_img,disk(1))
   
   edge = feature.canny(ers_aimg,sigma =0.8 )
   
   feat_edge = move_tocenter(edge)  
   
   row,col = np.where(edge)
   
   row_m = row.min()
   row_x = row.max()
   col_m = col.min()
   col_x = col.max()
   
   eg_area =  ( (row_x - row_m ) * (col_x - col_m)*1.0 ) / ( edge.shape[0]*edge.shape[1] )

   if eg_area < 0.8 :
        vex =  morphology.binary_closing(edge,morphology.square(3) )
        
        label = measure.label(vex,neighbors=8)
        
        if label.max() > 1:
            max_label = 0
            label_num = 0            
            for b in range(1,label.max()+1 ):
                tb = (label == b)
                if label_num < tb.sum():
                   label_num = tb.sum()
                   max_label = b
            if  label_num > vex.sum()* 0.8:  # get max region
                vex = (label == max_label)                       
                
        vex = binary_fill_holes(vex)
        
        #vex = morphology.convex_hull_image(edge)
        
        #binary_fill_holes
        feat_color = color_hist_obj(aimg,vex)
        feat_hsv   = color_hsv_objhist(aimg,vex)
        feat_area  = np.array([vex.sum()])
        
        row,col = np.where(vex)
        
        obj = aimg[row.min():row.max(),col.min():col.max()]
        
        edge_grow_save(feat_edge,afile,obj,debugflag)
   else :   
        feat_area = np.array([0])   # whole image
        feat_color =  color_hist_whole(aimg)
        feat_hsv   = color_hsv_whole(aimg)
        
        edge_grow_save(feat_edge,afile,aimg,debugflag)
        
   
   
   
   m = measure.moments(edge*1.0)
   cr = m[0,1]/m[0,0]
   cc = m[1,0]/m[0,0]
   
   mu = measure.moments_central(edge*1.0,cr,cc)
   nm = measure.moments_normalized(mu)
   
   feat_hu = measure.moments_hu(nm)
   
   lbp = feature.local_binary_pattern(ers_aimg,8,1,'nri_uniform' )
   nbins = lbp.max() +1
   feat_lbp,_ = np.histogram(lbp,normed = True,bins = nbins,range=(0,nbins) )
   
   feat_center = np.round(compcenter(edge))

   
   imgfeat= {}
   imgfeat[fs_edge] = feat_edge
   imgfeat[fs_area]  = feat_area
   imgfeat[fs_color] = feat_color
   
   imgfeat[fs_gray]   = ers_aimg   # img_gray
   imgfeat[fs_hsv]   = feat_hsv
   
   imgfeat[fs_center] = feat_center
   imgfeat[fs_hu]     = feat_hu
   imgfeat[fs_lbp]    = feat_lbp
   
   return imgfeat

@fn_timer
def com_imgfeat_tohdf5(srcpath,debugflag=0):
    os.chdir(srcpath)
    filelist = glob.glob(r'*.jpg')
    
    # test 1000
    #if len(filelist) > 10000:
    #   filelist = filelist[:10000]
    
    num =  len(filelist)
    
    h5fname =  os.path.join(srcpath,"feat.hdf5")
    f = h5py.File(h5fname,'w')
    f.create_dataset('num',(1,),data =(num)) 
    f.create_dataset('filelist',(num,),data = filelist )
    for i in range(num):
        afile = os.path.join(srcpath,filelist[i] )
        aimgfeat = img_feature_grow(afile,debugflag=debugflag)
        grp = f.create_group(filelist[i])
        for k ,val in aimgfeat.items():
            if k == fs_edge:
              dty =  'u1'
            elif k == fs_gray:
              dty  = 'u1'
              val =  img_as_ubyte(val)
            else :
               dty = np.float32
            grp.create_dataset(k,val.shape,data=val,dtype = dty)
        
        if i%1000 == 0:
          print ( i )        

    f.close()
    return 

def  com_readfeat_fromh5dy(h5f,filename):
     ks = h5f[filename].keys()
     aimgfeat = {}
     for k in ks: 
         aimgfeat[k] = h5f[filename][k].value
     return aimgfeat
    
def com_readfilelist_fromh5df(h5f):
     
    return list( h5f['filelist'].value )
    

def com_gather_file(dstsubpath,afile,file_moveflag =0 ):
    if not os.path.exists(dstsubpath) :
          os.mkdir(dstsubpath)
          
    if file_moveflag == 0 :
         shutil.copy(afile  , dstsubpath)
    else :
         shutil.move(afile , dstsubpath)

def com_get_patch(img,center):
    wn = 20
    cen = center
    
    if cen[0] - wn < 0 :
        up = 0
    else :
        up = cen[0] - wn
        
    if   cen[0]+ wn >= img.shape[0]:
          down =     img.shape[0]
    else : 
           down =  cen[0]+ wn
           
    if cen[1] - wn < 0 :
        left = 0
    else :
        left = cen[1] - wn
        
    if   cen[1]+ wn >= img.shape[1]:
          right =     img.shape[1]
    else : 
           right =  cen[1]+ wn
    
    cenimg = img[ up:down ,left:right ]
    
    return cenimg
                  
  
def com_judge_feature(aimgfeat,bimgfeat,featlist,debugflag = 0 ):
    
    dislist = {}
    corr  = 1
    
    for ft in featlist:
        ftype =  ft[0]
        fthre = ft[1]
        if ftype == fs_color:
           dis = distance.cosine(aimgfeat[fs_color],bimgfeat[fs_color])
        elif ftype == fs_hsv: 
            dis = distance.cosine(aimgfeat[fs_hsv],bimgfeat[fs_hsv])    
        elif ftype == fs_colorobj: 
            dis = distance.cosine(aimgfeat[fs_colorobj],bimgfeat[fs_colorobj])              
        elif ftype == fs_hu: 
            dis = distance.cosine(aimgfeat[fs_hu],bimgfeat[fs_hu])                  
        elif ftype == fs_lbp: 
            #dis = distance.euclidean(aimgfeat[fs_lbp],bimgfeat[fs_lbp])   
             dis = distance.cosine(aimgfeat[fs_lbp],bimgfeat[fs_lbp])              
        elif ftype == fs_area: 
            dis = cmp_area(aimgfeat[fs_area],bimgfeat[fs_area])      

        elif ftype == fs_hausdorff: 
            # only  edge sum is lower, 
            ag  = aimgfeat[fs_edge].sum() / (66*64.0)
            bg  = bimgfeat[fs_edge].sum() / (66*64.0)
            if ag   >= 0.1:
                if bg > 0.1 :
                   dis  = 1
                elif  bg - ag >=  0.2* ag:
                   dis = 64
                else :
                   dis =  1
            else:
               dis = cmp_haufdorff(aimgfeat[fs_edge],bimgfeat[fs_edge])       
        elif ftype ==  fs_template:
             agray =  aimgfeat[fs_gray]  #img_as_ubyte(  )
             bgray =  bimgfeat[fs_gray]  #img_as_ubyte (  )
             
             res = feature.match_template(agray,bgray)
             dis  = res.max()
             if dis <= fthre: 
                 
                  cen = aimgfeat[fs_center]
                  cenimg = com_get_patch(agray,cen)   #  agray[ cen[0] - wn :cen[0]+ wn ,cen[1] -wn:cen[1]+wn ]
                  sres = feature.match_template(bgray,cenimg)
                  dis  = sres.max()
                  
                  if dis <= fthre:
                     cen = bimgfeat[fs_center]
                     cenimg = com_get_patch(bgray,cen)  #  bgray[ cen[0] - wn :cen[0]+ wn ,cen[1] -wn:cen[1]+wn ]
                     sres = feature.match_template(agray,cenimg)
                     dis  = sres.max()
                      
                      
        dislist[ftype] = dis
        
        if ftype == fs_template :  
             if dis <= fthre :  
                 corr =  -1
                 break
        else :
             if dis >= fthre:
                 corr  = -1 
                 break
    if debugflag == 1:
        print dislist,
    return corr    
     
@fn_timer
def com_scan_forwhat(srcpath,featlist,dstpath='',datatype=0,file_moveflag = 0,debugflag = 0):
    
    if datatype == 0:
        os.chdir(srcpath)
        filelist = glob.glob(r'*.jpg')
    elif datatype == 1:
        h5fname = os.path.join(srcpath,"feat.hdf5" )
        h5f = h5py.File(h5fname,"r")
        filelist = com_readfilelist_fromh5df(h5f)
    
    if len(dstpath) == 0:
        dstpath = srcpath
        
    
    num = len(filelist)
    
    if num  ==0 or len(featlist) == 0 :
        print "input param err"
        return 
    
    #touchflag =  np.zeros((num,num))
    
    pflag = [0]*num
    filedict = dict(zip(filelist, pflag))
     
    cmpcount = 0
    
    for i in range(num) :
        
        if   filedict[ filelist[i]  ]  == 1: 
            continue
            
        filedict[ filelist[i]  ]  = 1
        
        
        
        bfind =  0
        seq_it = int(time.time())
        dstsubpath = os.path.join(dstpath,"%d-%d"%(seq_it,i) )
        afile = os.path.join(srcpath,filelist[i])
        
        if datatype == 0:    
            aimgfeat  =  img_feature_grow(afile,debugflag = debugflag )
        elif datatype == 1:
            aimgfeat  = com_readfeat_fromh5dy(h5f,filelist[i])
            
        print filelist[i]
        
        matchindex = []
        
        for j in range(i+1,num):
        
            if  filedict[ filelist[j]  ]  == 1: 
                  continue
            
            if j%2000 == 0 :
                print (j)
            
            bfile = os.path.join(srcpath,filelist[j])
            if datatype == 0:
                
                bimgfeat  =  img_feature_grow(bfile,debugflag = debugflag )
            elif datatype == 1:
                bimgfeat  = com_readfeat_fromh5dy(h5f,filelist[j])
               
            corr = com_judge_feature(aimgfeat,bimgfeat,featlist,debugflag = debugflag )             
            cmpcount = cmpcount +1
            
            #touchflag[i,j] = corr
            #touchflag[j,i] = corr
            if debugflag == 1:
               print filelist[j],corr
            
            if corr == 1:  
               if bfind == 0: 
                  com_gather_file(dstsubpath,afile,file_moveflag)
                  bfind = 1
               com_gather_file(dstsubpath,bfile,file_moveflag)
               filedict[ filelist[j]  ]  = 1
               matchindex.extend(  [ [j,bimgfeat ] ]) 
        

        #continue        
        while len(matchindex) > 0 :
             mp = matchindex.pop()
             k = mp[0]
             kimgfeat = mp[1]
             print filelist[k]
             for t in range(i+1,num):
                  if filedict[ filelist[t] ] ==  1 or k == t : 
                       continue
                  tfile = os.path.join(srcpath,filelist[t])
                  if datatype == 0:
                      timgfeat = img_feature_grow(tfile,debugflag = debugflag )
                  elif datatype == 1:
                      timgfeat  = com_readfeat_fromh5dy(h5f,filelist[t])
            
                  corr =  com_judge_feature(kimgfeat,timgfeat,featlist,debugflag = debugflag)
                  cmpcount = cmpcount +1 
                  #touchflag[k,t] = corr
                  #touchflag[t,k] = corr
                  if debugflag == 1:
                     print filelist[t],corr
            
                  if corr == 1:  
                     com_gather_file(dstsubpath,tfile,file_moveflag)
                     filedict[ filelist[t]  ]  = 1
                     matchindex.extend(  [ [t,timgfeat ] ])                  
               
    if datatype == 1:           
        h5f.close()      
    #print touchflag
                  
    print "........end ....",cmpcount,"filenum",num,num*num      

def com_scan_forwhat_h(srcpath,featlist,dstpath='',file_moveflag = 0,debugflag = 0):
    
    h5fname = os.path.join(srcpath,"feat.hdf5" )
    if not os.path.exists(h5fname):
         com_imgfeat_tohdf5(srcpath,debugflag=debugflag)
    com_scan_forwhat(srcpath,featlist,dstpath,datatype = 1,file_moveflag = file_moveflag,debugflag = debugflag )
    
    
def scan_test():
    srcpath = 'E:\\match_sample\\image\\coarse_step2_deep\\1493626913\\'
    #srcpath = 'E:\\match_sample\\image\\image\\'
    #dstpath = 'E:\\match_sample\\image\\scan1\\'
    
    srcpath = 'E:\\match_sample\\someone\\1494052532-2\\'
    srcpath = 'E:\\match_sample\\someone\\1494054534-9\\'
    srcpath = 'E:\\match_sample\\someone\\1494070569-0\\1494162094-0\\'
    #[fs_color,0.08],
    #featlist = [ [fs_color,0.04],[fs_area,0.2],[fs_lbp,0.06],[fs_template,0.58 ]]      --- wupan shao 
    
    featlist = [ [fs_color,0.04]]   
    #com_scan_forwhat(srcpath,featlist,file_moveflag = 0, debugflag = 1 )
    com_scan_forwhat_h(srcpath,featlist,file_moveflag = 0, debugflag = 1)
    
def  img_process_hdf():
    srcpath = 'E:\\match_sample\\image\\image\\'
    h5fname = os.path.join(srcpath,"feat.hdf5" )
    if not os.path.exists(h5fname):
         com_imgfeat_tohdf5(srcpath)
   
def  img_process_someone():
    srcpath = 'E:\\match_sample\\image\\image\\'
    dstpath = 'E:\\match_sample\\sometwo\\'
    
    h5fname = os.path.join(srcpath,"feat.hdf5" )
    if not os.path.exists(h5fname):
        print "no hdf5 file exist"
        return      
    #featlist = [ [fs_hsv,0.04],[fs_area,0.2],[fs_lbp,0.008],[fs_template,0.6 ]]    ###   --- wupan shao 
    featlist = [ [fs_hsv,0.02 ] ]    
    com_scan_forwhat(srcpath,featlist,dstpath,datatype = 1,file_moveflag = 0,debugflag = 0 )
    
import sys
if __name__ == "__main__":

   scan_test()
   
   #img_process_hdf()
   #img_process_someone()
    
   
