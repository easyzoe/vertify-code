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
from sklearn.cluster import KMeans
from sklearn import  cluster
from sklearn.neighbors import NearestNeighbors

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
fs_lab      = 'lab'





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
  
  rgbhgmf = lambda  img  : np.row_stack(( np.histogram(img[:,0] ,bins =64,range=(0,255))[0],
                         np.histogram(img[:,1] ,bins =64,range=(0,255))[0],
                         np.histogram(img[:,2] ,bins =64,range=(0,255))[0],  
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


def com_color_hist(img,vex=None,fstype=fs_color):
    
    imgobj = img
  
    if fstype == fs_color :
        range1=(0,255)
        range2=(0,255)
        range3=(0,255)
        bins=[64,64,64]
    elif fstype == fs_hsv:
        range1=(0,1)
        range2=(0,1)
        range3=(0,1)   
        bins=[128,32,32]
    elif fstype == fs_lab:
        range1=(0,100)
        range2=(-128,127)
        range3=(-128,127)
        bins=[8,28,28]
    
    
    len = bins[0]+bins[1] + bins[2]
    
    if not vex is None:
       if  vex.sum() == 0:
          imghist = np.zeros( len)
          return imghist
       if fstype == fs_hsv:
           imgobj = color.rgb2hsv(img)[vex]
       elif fstype == fs_lab:
           imgobj = color.rgb2lab(img)[vex]
       else :
          imgobj =  img[vex]     
       tnum = vex.sum()*1.0          
    else :
       if fstype == fs_hsv:
           imgobj = color.rgb2hsv(img)
       elif fstype == fs_lab:
           imgobj = color.rgb2lab(img)
       else :
          imgobj =  img
       tnum = img.shape[0]*img.shape[1]*1.0
               
   
    
    if not vex is None:
       va = np.histogram(imgobj[:,0],bins=bins[0], range=range1 )[0] /tnum
       vb = np.histogram(imgobj[:,1],bins=bins[1], range=range2 )[0] /tnum
       vc = np.histogram(imgobj[:,2],bins=bins[2], range=range3 )[0] /tnum
        
       imghist = np.concatenate((va,vb,vc)) /3.0
       
    else :
       va = np.histogram(imgobj[:,:,0],bins=bins[0],range=range1 )[0] /tnum
       vb = np.histogram(imgobj[:,:,1],bins=bins[1], range=range2 )[0] /tnum
       vc = np.histogram(imgobj[:,:,2],bins=bins[2], range=range3 )[0] /tnum
        
       imghist = np.concatenate((va,vb,vc)) /3.0
    
    return imghist
 
#
def com_split_img(img):
    splitelm = 2 # 2*2 
    rowoff = int( np.round( img.shape[0]*0.5 * 0.1 ))
    coloff = int( np.round( img.shape[1]*0.5 * 0.1 ) )
    
    row  = img.shape[0]/ splitelm
    col = img.shape[1] /splitelm
    
    splitlist = []
    for i in range(splitelm):
       for j in range(splitelm):
           if i ==  splitelm - 1:
              irow_up  = i*row - rowoff
              irow_down = (i+1) * row
           else:
              irow_up = i*row
              irow_down = (i+1)*row + rowoff
           if  j == splitelm  -1 :
              jcol_left =  j * col - coloff
              jcol_right = (j+1) * col
            
           else :
              jcol_left = j*col
              jcol_right = (j+1)*col + coloff
           split = img[irow_up:irow_down,jcol_left:jcol_right]
                      
           splitlist.extend( [ split]  )
      
    return  splitlist      
    
        
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

   if eg_area < 5 :
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
            if  label_num > vex.sum()* 0.78:  # get max region
                vex = (label == max_label)                       
                
        vex = binary_fill_holes(vex)
        
        #vex = morphology.convex_hull_image(edge)
        
        #binary_fill_holes
        #feat_color = color_hist_obj(aimg,vex)
        row,col = np.where(vex)
        
        obj = aimg[row.min():row.max(),col.min():col.max()]
        gray_obj =  gray_img[row.min():row.max(),col.min():col.max()]
        feat_color = com_color_hist(obj,fstype=fs_lab)  #com_color_hist(aimg,vex,fstype=fs_color)
        feat_hsv   = com_color_hist(aimg,vex,fstype=fs_lab)  #color_hsv_objhist(aimg,vex)
        feat_area  = np.array( [vex.sum()] )
        
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
   
   # 5/21 obj  lbp
   
   #lbp = feature.local_binary_pattern(ers_aimg,8,1,'nri_uniform' )
   #lbp = feature.local_binary_pattern(gray_obj,8,1,'nri_uniform' )
   
   #nbins = lbp.max() +1
   #feat_lbp,_ = np.histogram(lbp,normed = True,bins = nbins,range=(0,nbins) )
   
   feat_lbp = com_lbp_create(gray_obj)
   
   feat_center = np.round(compcenter(edge))

   
   imgfeat= {}
   imgfeat[fs_edge] = feat_edge
   imgfeat[fs_area]  = feat_area
   imgfeat[fs_color] = feat_color
   
   imgfeat[fs_gray]   =  gray_obj   #ers_aimg   # img_gray
   imgfeat[fs_hsv]   = feat_hsv
   
   imgfeat[fs_center] = feat_center
   imgfeat[fs_hu]     = feat_hu
   imgfeat[fs_lbp]    = feat_lbp
   
   return imgfeat

def com_lbp_create(gray_obj):
   splitobj = com_split_img(gray_obj)
   
   #lbp = feature.local_binary_pattern(gray_obj,8,1,'nri_uniform' )
   #nbins = lbp.max() +1
   #feat_lbp,_ = np.histogram(lbp,normed = True,bins = nbins,range=(0,nbins) )
   feat_lbp = []
   for spobj in splitobj:
      lbp = feature.local_binary_pattern(spobj,8,1,'nri_uniform' )
      nbins = lbp.max() +1
      s_lbp,_ = np.histogram(lbp,normed = True,bins = nbins,range=(0,nbins) )
      feat_lbp.extend(s_lbp)
   
   
   return np.array ( feat_lbp )
  
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
    return (corr, dislist )  
     
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
        feat_corr_que ={}
        
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
               
            corr,dislist = com_judge_feature(aimgfeat,bimgfeat,featlist,debugflag = debugflag )             
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
               matchindex.extend(  [ [j ] ]) 
            else :
               com_up_sortcorr(feat_corr_que, dislist,j)
        
        print "step one get ",len(matchindex)
        
        for fk,val in feat_corr_que.items():
            fs_que = np.array(val)
            fs_que = fs_que.reshape((fs_que.shape[0]/2 ,2))
            a1 = fs_que[:,::-1].T  
            a2 = np.lexsort(a1)  
            nfs_que = fs_que[a2]  
            
            qnum = nfs_que.shape[0]
            
            lastval = 0
            
            featthr = featlist[0][1]
            
            for q in range(qnum):
                qv = nfs_que[q]
                
                k = int(qv[1])
                
                if qv[0] > 1.8 * featthr :
                    break
                
                kfile = os.path.join(srcpath,filelist[k])
                
                if datatype == 0:
                      kimgfeat = img_feature_grow(kfile,debugflag = debugflag )
                elif datatype == 1:
                      kimgfeat  = com_readfeat_fromh5dy(h5f,filelist[k])
                
                for tv in matchindex:
                    t =tv[0]
                    tfile = os.path.join(srcpath,filelist[t])
                    if datatype == 0:
                       timgfeat = img_feature_grow(tfile,debugflag = debugflag )
                    elif datatype == 1:
                       timgfeat  = com_readfeat_fromh5dy(h5f,filelist[t])
                       
                    corr,dislist =  com_judge_feature(kimgfeat,timgfeat,featlist,debugflag = debugflag)
                    cmpcount = cmpcount +1  
                
                    if debugflag == 1:
                       print filelist[t],corr
            
                    if corr == 1:  
                       com_gather_file(dstsubpath,kfile,file_moveflag)
                       filedict[ filelist[k]  ]  = 1
                       matchindex.extend(  [ [k ] ])   
                     
                       lastval =qv[0]
                       break                       
                     
            print fk,"other point ,maybe threold ",lastval,  nfs_que[-1,0]  
            print fk,"end of step two get ", len(matchindex)            
                                                                                   
    if datatype == 1:           
        h5f.close()      
    #print touchflag
                  
    print "........end ....",cmpcount,"filenum",num,num*num      

#  sort  by corr
def com_up_sortcorr(feat_corr_que, dislist,j):
    for k,v in dislist.items():
        if not feat_corr_que.has_key(k):
              feat_corr_que[k] = [v,j]
        else :
              feat_corr_que[k].extend([v,j])
    return 


def com_scan_forwhat_h(srcpath,featlist,dstpath='',file_moveflag = 0,debugflag = 0):
    
    h5fname = os.path.join(srcpath,"feat.hdf5" )
    if not os.path.exists(h5fname):
         com_imgfeat_tohdf5(srcpath,debugflag=debugflag)
    com_scan_forwhat(srcpath,featlist,dstpath,datatype = 0,file_moveflag = file_moveflag,debugflag = debugflag )
    
@fn_timer
def com_sklearn_entry(srcpath,dstpath='',datatype=0,file_moveflag = 0,debugflag = 0,feattype = fs_color,clustertype = 1 ,nclass =30 ):
    
    h5fname = os.path.join(srcpath,"feat.hdf5" )
    if not os.path.exists(h5fname):
         com_imgfeat_tohdf5(srcpath)
    
    print 'read data ....'    
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
    
    if num < 2:
       return 
    
    #feattype = fs_color
    
    data = []
    for i in range(num) :
        
        afile = os.path.join(srcpath,filelist[i])
        
        if datatype == 0:    
            aimgfeat  =  img_feature_grow(afile,debugflag = debugflag )
        elif datatype == 1:
            aimgfeat  = com_readfeat_fromh5dy(h5f,filelist[i])
        dval = aimgfeat[feattype]
        #dval = np.concatenate( (aimgfeat[fs_color],aimgfeat[fs_hsv],aimgfeat[fs_lbp],aimgfeat[fs_hu] )) 
        data.extend( [ dval ] )
    npdata = np.array(data)
    
    h5f.close()
    
  
    # ap 
    
    print " cluster start ...  clustertype ",clustertype
    
    if clustertype == 1:
        #nclass = 30 
        ap = cluster.KMeans(n_clusters = nclass,max_iter = 100 ).fit(npdata)
        
        print ap.inertia_ 
    elif clustertype == 2:
        ap =  cluster.AffinityPropagation().fit(npdata)
        print ap.n_iter_ 
    elif clustertype == 3    :
        sm = num*0.1 
        if sm < 1:
           sm = num
        bandwidth = cluster.estimate_bandwidth(npdata, quantile=0.2, n_samples =sm  )
        ap  = cluster.MeanShift(bandwidth=bandwidth*0.06, bin_seeding=True).fit(npdata)
        
        
    elif clustertype == 4:
        ap = cluster.AgglomerativeClustering(n_clusters=nclass, linkage='average',affinity ='cosine',
                                           connectivity=connectivity).fit(npdata)  
        print ap.n_components_ ,ap.n_leaves_                                            
                                           
    num = ap.labels_.max()
    print 'cluster end ',num
    
    for i in range(num+1):
        result = np.where(ap.labels_ == i)
        
        
        
        seq_it = int(time.time())
        dstsubpath = os.path.join(dstpath,"%d-%d"%(seq_it,i) )
        
        com_movefile_bysubset(result[0],srcpath,filelist,dstsubpath,file_moveflag,datatype)
    return     
    
            
def com_movefile_bysubset(subset,srcpath,filelist,dstsubpath,file_moveflag,datatype):
    
  
        
    subfilelist= []    
    
    for i in subset:
        afile = os.path.join(srcpath,filelist[i])
        com_gather_file(dstsubpath,afile,file_moveflag)
        if datatype == 1:
           subfilelist.append(  filelist[i] )
    
    if datatype == 1 :
        h5fname = os.path.join(srcpath,"feat.hdf5" )
        h5f = h5py.File(h5fname,"r")  
        subh5fname =  os.path.join(dstsubpath,"feat.hdf5")
        f = h5py.File(subh5fname,'w')
        
    if datatype == 1:
        num =  len(subfilelist)
        f.create_dataset('num',(1,),data =(num)) 
        f.create_dataset('filelist',(num,),data = subfilelist )  
        for i in subset:
            aimgfeat  = com_readfeat_fromh5dy(h5f,filelist[i])        
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
  
        f.close()
        h5f.close()
    
    return 

def com_any_cluster(index,distance,num,s =0 ,thre =  0.028 ):

    total_seq = None

    snb = index[s][1:][  distance[s][1:] < thre  ]
    total_seq = snb.copy()
    
    for i in range(len(snb)):
        t = snb[i]
        tp_seq = []
        tp_seq  = index[t][1:][ distance[t][1:] < thre ]
        total_seq = np.append(total_seq,tp_seq)
    total_seq = np.append(s,total_seq)
    res = np.unique(total_seq)

    return res

def com_getfeat(srcpath,featlist,datatype=0,debugflag = 0):
    
    if datatype == 1: 
       h5fname = os.path.join(srcpath,"feat.hdf5" )
       if not os.path.exists(h5fname):
          com_imgfeat_tohdf5(srcpath,debugflag=debugflag)
    
    print 'read data ....'    
    if datatype == 0:
        os.chdir(srcpath)
        filelist = glob.glob(r'*.jpg')
    elif datatype == 1:
        h5fname = os.path.join(srcpath,"feat.hdf5" )
        h5f = h5py.File(h5fname,"r")
        filelist = com_readfilelist_fromh5df(h5f)
   
    
    
    num = len(filelist)
    
   
    #feattype = fs_color
    
    data = {}
    for feattype in featlist:
       data[feattype] = []
       
    for i in range(num) :
        
        afile = os.path.join(srcpath,filelist[i])
        
        if datatype == 0:    
            aimgfeat  =  img_feature_grow(afile,debugflag = debugflag )
        elif datatype == 1:
            aimgfeat  = com_readfeat_fromh5dy(h5f,filelist[i])
        for feattype in featlist:
            
            if feattype == fs_gray:
               dval = np.append( aimgfeat[feattype].flatten(),aimgfeat[fs_center].flatten() ) 
            else :
               dval = aimgfeat[feattype]
            
            data[feattype].extend ( [dval]  ) 
    
    for feattype in featlist:    
       data[feattype] = np.array(data[feattype])
    if datatype == 1: 
       h5f.close()   
    
    return (filelist,data)

def knn_custom_match(av,bv):
    agray = av[:4224].reshape((66,64))
    bgray = bv[:4224].reshape((66,64))
    acen = av[-2:]
    bcen = bv[-2:]
    res = feature.match_template(agray,bgray)
    dis  = res.max()

    if dis > 0.9:
       return 1- dis
    
    cen = acen
    cenimg = com_get_patch(agray,cen)
    
    sres = feature.match_template(bgray,cenimg)
    sdis = sres.max()
    
    if sdis > dis:
       dis = sdis
    return 1 - dis
    
def com_Neighbors(srcpath,dstpath='',datatype=0,file_moveflag = 0,debugflag = 0):
    
    if len(dstpath) == 0:
        dstpath = srcpath
    
    featlist=  [ fs_color,fs_area,fs_lbp,fs_gray,fs_hu ]
    
    filelist,featdata = com_getfeat(srcpath,featlist,datatype=datatype,debugflag=debugflag)
    
    tmpfilelist = glob.glob(r'%s*.jpg'%(srcpath))
    
    if len(tmpfilelist) != len(filelist) :
       return 
       
    
    num = len(filelist)
    
    if num < 2:
       return
    if num > 500:
       print "NearestNeighbors  num o file is ", num
       return
       
    scanflag = np.zeros(num)
    
    index_distance = {}
    for feattype in featlist:
        if feattype == fs_color or feattype ==  fs_hu or feattype == fs_area or  feattype == fs_lbp :
           nbrs = NearestNeighbors(n_neighbors = num,algorithm='ball_tree' ).fit( featdata[feattype] )
           index_distance[feattype] = nbrs.kneighbors( featdata[feattype] )
         
    labels = {}
    labelit = 1
    
    
    
    for i in range(num):
       if scanflag[i] > 0 :
            continue
            
       res_color =  com_any_cluster( index_distance[fs_color][1],index_distance[fs_color][0],num,i,thre = 0.05)
       #res_area =  com_any_cluster( index_distance[fs_area][1],index_distance[fs_area][0],num,i,thre = 0.3)
       res_lbp =  com_any_cluster( index_distance[fs_lbp][1],index_distance[fs_lbp][0],num,i,thre = 0.15 )
       # split lbp  thre = 0.15; obj lbp 0.06  
       res_hu =  com_any_cluster( index_distance[fs_hu][1],index_distance[fs_hu][0],num,i,thre = 0.1)
       
       #select_set = list( set(res_color) &  set(res_area) & set(res_lbp) & set(res_hu)   ) 
       select_set = np.array(  list( set(res_color) &  set(res_lbp) & set(res_hu)   )  )
       
       if len( select_set) == 0 or  (   len(select_set) == 1 and  select_set[0] == i   ) :
            continue
       
       #select_set = np.array(range(num))
       
       subdata = featdata[fs_gray][select_set]
       
       
       graynbr = NearestNeighbors(n_neighbors = len(select_set),algorithm='brute',metric =knn_custom_match ).fit(  subdata  )
       grayindexdistance = graynbr.kneighbors(subdata)
       
       bpos = np.where(select_set ==i )[0][0]
       
       newselect_set = com_any_cluster(grayindexdistance[1],grayindexdistance[0],len(select_set),bpos,thre=0.4)
       
       select_set = select_set[newselect_set]
       
       if len(select_set) < 2:
          continue
       
       bnewlabelflag = 0
       
       standby = scanflag[select_set]
       if standby.sum() > 0:
           bmay =  np.unique(standby[standby > 0] )
           
           
           moveset = select_set[ scanflag[select_set] ==0 ]
           
           scanflag[select_set] = bmay[0]
           dstsubpath = labels[bmay[0]]
       

       else :
          seq_it = int(time.time())
          aplt = srcpath.split("\\")[-2]
          dstsubpath = os.path.join(dstpath,"%s-%d-%d"%(aplt,seq_it,i) )    
          scanflag[select_set] = labelit
          labels[labelit] = dstsubpath
          labelit = labelit +1   

          moveset = select_set 
          
       com_movefile_bysubset(moveset,srcpath,filelist,dstsubpath,file_moveflag,datatype=0 )
       
    return   
              
    

# onegather_try1
@fn_timer
def onegather_try1():
    srcpath = 'E:\\match_sample\\image\\image\\'
    #srcpath = 'E:\\match_sample\\image\\coarse_step2\\1493629832\\'
    dstpath = 'E:\\match_sample\\onecoarse\\'
    
    #srcpath,dstpath='',datatype=0,file_moveflag = 0,debugflag = 0,feattype = fs_color,clustertype = 1 ,nclass =30 
    
    com_sklearn_entry(srcpath,dstpath=dstpath,datatype = 1 ,file_moveflag = 0 ,feattype = fs_color,clustertype = 1,nclass =  10 )
    
    subpath =  glob.glob(r'%s*'%dstpath)
    for asub in subpath:
       print asub
       com_sklearn_entry(asub,dstpath = dstpath,datatype = 1,file_moveflag = 1,feattype = fs_color,clustertype = 2 ) 
       os.remove(os.path.join(asub,"feat.hdf5"))
       os.rmdir(asub)      
    return 
    
@fn_timer
def onegather_more_try1():
    srcpath = 'E:\\match_sample\\onecoarse_1_obj30\\'
    #srcpath = 'E:\\match_sample\\image\\coarse_step2\\1493629832\\'
    dstpath = 'E:\\match_sample\\onemore\\'
    
    #srcpath,dstpath='',datatype=0,file_moveflag = 0,debugflag = 0,feattype = fs_color,clustertype = 1 ,nclass =30 
    
   
    
    subpath =  glob.glob(r'%s*'%srcpath)
    for asub in subpath:
       print asub
       asub = "%s\\"%asub
       com_Neighbors(asub,datatype = 1,dstpath=dstpath,file_moveflag = 1 )
    return 

    
    
def scan_test():
    srcpath = 'E:\\match_sample\\onecoarse_1_obj30\\1495292823-8\\'
    #srcpath = 'E:\\match_sample\\somefour\\1494518082-0\\'
    #srcpath = 'E:\\match_sample\\image\\image\\'
    #dstpath = 'E:\\match_sample\\testout\\'
    #dstpath = 'E:\\match_sample\\kmeanlbp\\'
    
    
    #srcpath = 'E:\\match_sample\\someone\\1494052532-2\\'
    #srcpath = 'E:\\match_sample\\someone\\1494054534-9\\'
    #srcpath = 'E:\\match_sample\\someone\\1494150475-6\\'
    #[fs_color,0.08],
    #featlist = [ [fs_color,0.04],[fs_area,0.2],[fs_lbp,0.06],[fs_template,0.58 ]]      --- wupan shao 
    
    featlist = [ [fs_color,0.05]]   
    #com_scan_forwhat(srcpath,featlist,file_moveflag = 0, debugflag = 1 )
    #com_scan_forwhat_h(srcpath,featlist,file_moveflag = 0, debugflag = 1)
    #com_sklearn_entry(srcpath,dstpath = dstpath,datatype = 1 )
    com_Neighbors(srcpath,datatype = 1,debugflag = 1 )

def scan_kmean_test():
    
    srcpath = 'E:\\match_sample\\testoutkmean\\'
    dstpath = 'E:\\match_sample\\combinefeat\\'  
    subpath =  glob.glob(r'%s*'%srcpath)
    for asub in subpath:
       print asub
       com_sklearn_entry(asub,dstpath = dstpath,datatype = 1 )  
    
    
 #   E:\match_sample\

 
    
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
    featlist = [ [fs_color,0.05 ] ]    
    com_scan_forwhat(srcpath,featlist,dstpath,datatype = 1,file_moveflag = 0,debugflag = 0 )
    
import sys
if __name__ == "__main__":

   #
   scantype =  0  #  scan
                  # 2 : kmean
   if scantype == 0:
     scan_test()
     #scan_kmean_test()
     #scan_ap_test()
   elif scantype == 1 :
      img_process_hdf()
      img_process_someone()
   elif scantype == 2:
      onegather_try1()
   elif scantype == 3:
      onegather_more_try1()   
      
    