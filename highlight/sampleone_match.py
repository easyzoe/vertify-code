# -*- coding:utf-8 -*-
'''
    抽取来源于同一张图片 的 延伸照片
'''
#from os import listdir

import glob,os,shutil
from skimage import io
from skimage.feature import match_template
import numpy as np
import redis

import matplotlib.pyplot as plt
    
# 
def smp_getfilelist(path):
    
    spath = os.path.join(path,'')
    
    
    flist = glob.glob(r'%s*.jpg'%(spath))
   
    return flist
# filelist ,
# cmp_in_oneimage     
def smp_normal_one_infilelist(filelist,cmp_in_oneimage,r,type = 0):
    
    num = len(filelist)
    pflag = [0]*num
    
    filedict = dict(zip(filelist, pflag))
    
    
    match_result = {}
    #match_result = dict(zip(filelist, result))
    dstpath = os.path.dirname(filelist[0])
    dstpath = u"E:\\match_sample\\image\\sample\\"
    os.chdir(dstpath)
    subdir = glob.glob(r'[0-9]*?')
    if len(subdir) == 0:
        seq_it = 0
    else :
        abc = np.array(subdir,dtype = int)
        seq_it = abc.max() +1
         
    
    for i in range(len(filelist) ) :
        
        if filedict[  filelist[i]  ]  == 1:
              continue 
        
        filedict[  filelist[i]  ]   = 1
        
        match_result[ filelist[i]  ] = []
        
        if type == 1:        
           imax =64 
        else :
           imax = 0        
        
        vallist = []
        
        for j in range(len(filelist) ):
             if i == j :
                  continue
             if filedict [ filelist[j] ]  ==  1:
                  continue
             
             if j%3000 == 0 :
                print (' cmp  %d'%j)              
             
             res = cmp_in_oneimage(  filelist[i] ,filelist[j],r )
             
             if res[0] == 1:
                # 判断是同一张照片调整出来的
                filedict [ filelist[j] ] = 1
                match_result[ filelist[i]  ].extend( [ filelist[j] ] )
             
             if type == 0 and  imax < res[1]:
                   imax  =  res[1]
             elif type == 1 and  imax  > res[1]:
                   imax = res[1]
             #vallist.append(res[1])  
        #print (vallist) 
        
        print ("process %d  get max corrlation %f "%(seq_it,imax ) )
        smp_move_image_step(filelist[i],match_result[ filelist[i]  ],dstpath,seq_it )  
        seq_it =  seq_it +1         
                 
    print ("sample scan over")
    #  
               
def  smp_move_image_step(key,val,destpath,seq):
     
     subpath = os.path.join(destpath,'%d'%seq ) 
         
     if  not os.path.exists(subpath) :
          os.mkdir(subpath)
     
     shutil.move(key,subpath)  
     if len(val) > 0:
        for a in val:
           shutil.move(a,subpath)  
               
     
     
         

#  通过截取中心位置扫描匹配
def cmp_in_oneimage_centerblock(Afile,Bfile,r):

    #Aimage = smp_getimage_from_redis(r,Afile)   # io.imread(Afile,as_grey = True)
    Aimage  = smp_getimage_from_dict(r,Afile)
    Bimage = smp_getimage_from_dict(r,Bfile)   # io.imread(Bfile,as_grey = True)
    
    Acenter = Aimage[3:63,3:63]
    #Acenter = Aimage[13:53,13:53]
    
    result = match_template(Bimage,Acenter)
    
    result = np.round(result,3)
    
    row,col = np.where(result == result.max())
    
    
    if result.max() > 0.7  :
       return (1,result.max())
    else :
        return (0,result.max())
 
def cmp_hash_maybe(Afile,Bfile,r):

    #Aimage = smp_getimage_from_redis(r,Afile)   # io.imread(Afile,as_grey = True)
    Aimage  = smp_readhash_dict(r,Afile)
    Bimage = smp_readhash_dict(r,Bfile)   # io.imread(Bfile,as_grey = True)
    
    abc =  (Aimage - Bimage)
   
    if abc  <  22 :
       return (1,abc)
    else :
        return (0,64)
        

def smp_oneimage_center(path,type = 0 ):
    
    filelist = smp_getfilelist(path)
    
    if len(filelist) == 0:
        return
    
    if type == 0 :
       if len(filelist) > 60000:
           filelist = filelist[:60000]
    
    #r = redis.Redis(host='localhost',port=6379,db=1)
    #htablename = os.path.basename(filelist[-1])
    
    #rex = r.hkeys(htablename)
    #if len(rex)  == 3:
    #    print ("data in redis already")
    #else :
    #   smp_putimage_to_redis(filelist,r)
    imagedict = {}
    if type ==  0:
    
       smp_putimage_to_dict(filelist,imagedict)
    elif type ==  1:
      
       smp_image_hash(filelist,imagedict)
    
    if type == 0:
       result = smp_normal_one_infilelist(filelist,cmp_in_oneimage_centerblock,imagedict)
    elif type ==  1:
       result = smp_normal_one_infilelist(filelist,cmp_hash_maybe,imagedict)
     
            
    
 # put image to  redis 
def smp_putimage_to_redis(filelist,r):
    
    
    
    for f in filelist:
        img = io.imread(f,as_grey = True)
        
        htablename = os.path.basename(f)     
        datastr = img.tostring()
        datadtype = img.dtype.str
        datarecord = img.shape[0] 
        
              #先删除
        r.delete(htablename)
        
        r.hset(htablename,'data',datastr)
        r.hset(htablename,'dtype',datadtype)
        r.hset(htablename,'record',datarecord)

def smp_putimage_to_dict(filelist,gimagedict):
    
    
    for i in range(len(filelist)):
        f =  filelist[i]
        img = io.imread(f,as_grey = True)
        
        htablename = os.path.basename(f)   

        gimagedict[htablename]   =  img
        
        if i%1000 == 0 :
           print (i)
      
        
def smp_getimage_from_dict(gimagedict,finame):
    
    htablename = os.path.basename(finame)
    
    return  gimagedict[htablename]
     
        
    
def smp_getimage_from_redis(r,finame):
    
    htablename = os.path.basename(finame)
    
    hdict = r.hgetall(htablename)
    
    if len(hdict) ==0 :
        return []
    
    datastr = hdict['data']     
    datadtype =  hdict['dtype'] 
    datarecord =  int(hdict['record'] )

    casenp = np.fromstring(datastr,datadtype)
    casenp = casenp.reshape(  ( datarecord, casenp.shape[0]/datarecord ) )
    
    return casenp
    


def debug_in_oneimage_centerblock(Afile,Bfile):

    Aimage = io.imread(Afile,as_grey = True)
    Bimage = io.imread(Bfile,as_grey = True)
    
    Acenter = Aimage[3:63,3:63]
    #Acenter = Aimage[13:53,13:53]
    
    result = match_template(Bimage,Acenter)
    
    result = np.round(result,3)
    
    row,col = np.where(result == result.max())
    
    print (result.max())
    
def  test_image():
     afile = u"E:\\picdog\\bdimagedb\\人民币\\6\\1475406430_2.jpg"
     bfile = u"E:\\picdog\\bdimagedb\\人民币\\9\\1475407233_8.jpg"
     
     debug_in_oneimage_centerblock(afile,bfile)
     

#
'''
   利用hash 粗分配，减少计算量
'''
import imagehash
from PIL import Image

def smp_image_hash(filelist,imghashdict):
    
    for i in range(len(filelist)):
        fname = filelist[i]
        hashval = imagehash.average_hash(Image.open(fname))
        #hashval = imagehash.phash(Image.open(fname))
        htablename = os.path.basename(fname)   

        imghashdict[htablename]   =  hashval
        
        if i%1000 == 0 :
           print (i)
        
def smp_readhash_dict(imghashdict,finame):
    
    htablename = os.path.basename(finame)
    
    return  imghashdict[htablename]   
    
''' 
    
'''    
def smp_corraltion_matrix():
    a = 0

# r,g,b,分配匹配上，    
    
 
def test_bdimagedb():
    path =u"E:\\picdog\\bdimagedb"
    
    spath = glob.glob(r'%s\\*'%path)
    
    for ap in spath:
        
        smp_oneimage_center(ap,0) 
        
import sys

def match_nextto_hash(path):
    
   
    
    for a in range(11,200):
        d = '%s%d'%(path,a)    
        type = 0 
        smp_oneimage_center(d,type)       
    
def  get_subnum(path = "E:\\match_sample\\image\\sample\\"):
     
    sub = os.listdir(path)
    for s  in sub:
       spath = os.path.join(path,s)
       num =  len( os.listdir(spath) )
       if num > 8:
          print ("%s %d " %(s,num))       

if __name__ == "__main__":
    get_subnum()
   
    
   # path =u"E:\\match_sample\\image\\try1\\"
    #match_nextto_hash(path)
    
    
    
