# -*- coding:utf-8 -*-
# 将相同的图像回笼

import glob,os,shutil
import numpy as np
import time


from imgnormal import img_feature_grow,img_grow_distance,classify_define,do_update_roate,cmp_img_by_step,cmp_img_coarse_step


#d
def img_touch_all(path,filelist,scanflag=0,debugflag=0):
    
    num = len(filelist)
    
    pflag = [0]*num
    
    filedict = dict(zip(filelist, pflag))
    
    
    
    
    touch = np.zeros((num,num))
    
    for i in range(num) :
        
        afile = os.path.join(path,filelist[i])
        
        if  scanflag == 1 and  filedict[ filelist[i]  ]  == 1: 
            continue
            
        filedict[ filelist[i]  ]  = 1
        
        aimgfeat  =  img_feature_grow(afile,debugflag)
        print filelist[i]
        for j in range(i+1,num):
        
            if  scanflag == 1 and  filedict[ filelist[j]  ]  == 1: 
                  continue
             
            bfile = os.path.join(path,filelist[j])
        
            bimgfeat  =  img_feature_grow(bfile,debugflag)
            
            res_dis = img_grow_distance(aimgfeat,bimgfeat)
            
            corr = 0
            if classify_define(res_dis) == 1:
                 corr = 1
            else : 
                corr =   0 #do_update_roate(res_dis,aimgfeat,bimgfeat)
            
            if corr == 1:
                filedict[ filelist[j]  ]  = 1
                
            touch[i,j] = corr
            touch[j,i] = corr
            #else :
        if scanflag ==  1:
           mtch = np.where(touch[i] == 1 )
           if len(mtch[0]) >0 :
              img_result_show_i(path,filelist,i,mtch[0],path,i,file_moveflag =1 )     
                  
    print touch
    
    
    return touch

    
def img_result_show_i(path,filelist,i,matchlist,dstpath,seqit,file_moveflag = 0 ):

    dstsubpath = os.path.join(dstpath,"%d"%seqit)
    
    
    ifile = os.path.join(path,filelist[i])
    if  not os.path.exists(ifile):
        return 
        
    if not os.path.exists(dstsubpath) :
          os.mkdir(dstsubpath)
          
        
    if file_moveflag == 0 :
         shutil.copy(ifile  , dstsubpath)
    else :
         shutil.move(ifile , dstsubpath)
    
    for j in matchlist:
        jfile = os.path.join(path,filelist[j])
        
        if  not os.path.exists(jfile):
            continue
        
        if file_moveflag == 0 :
           shutil.copy(jfile , dstsubpath)
        else :
           shutil.move(jfile , dstsubpath)

           
def img_result_touch_show(path,filelist,touch):
    
    num = len(filelist)
    
    seq  = 0
    
    dstpath = path
    
    for i in range(num):
         mtch = np.where(touch[i] == 1 )
         img_result_show_i(path,filelist,i,mtch[0],dstpath,seq)    
         seq = seq +1 



 

def img_scanforone_all(path,filelist,dstpath='',coarse =0,file_moveflag = 0):
    scanflag=1
    
    num = len(filelist)
    
    pflag = [0]*num
    
    filedict = dict(zip(filelist, pflag))
    
    if len(dstpath) == 0:
       dstpath = path
   
    
    
    
    
    for i in range(num) :
        
        afile = os.path.join(path,filelist[i])
        
        if  scanflag == 1 and  filedict[ filelist[i]  ]  == 1: 
            continue
            
        filedict[ filelist[i]  ]  = 1
        
        bfind =  0
        seq_it = int(time.time())
        dstsubpath = os.path.join(dstpath,"%d"%seq_it)
        
        
        aimgfeat  =  img_feature_grow(afile)
        print filelist[i]
        for j in range(i+1,num):
        
            if  scanflag == 1 and  filedict[ filelist[j]  ]  == 1: 
                  continue
            
            if j%2000 == 0 :
                print (j)
                
            bfile = os.path.join(path,filelist[j])
        
            bimgfeat  =  img_feature_grow(bfile)
            
            if coarse == 0:
               corr = cmp_img_by_step(aimgfeat,bimgfeat)
            else :
               corr = cmp_img_coarse_step(aimgfeat,bimgfeat)
                
          
            
            if corr == 1:  
               filedict[ filelist[j]  ]  = 1
               
               if bfind ==0 :
                    bfind = 1
                    if not os.path.exists(dstsubpath) :
                        os.mkdir(dstsubpath)
                    if file_moveflag == 0 :
                       shutil.copy(afile , dstsubpath)
                    else :
                       shutil.move(afile , dstsubpath)
                       
               if file_moveflag == 0 :
                   shutil.copy(bfile , dstsubpath)
               else :
                   shutil.move(bfile , dstsubpath)
               
               
            #else :
                  
    print "........end ...."
    
    
# 如果指定源目录，则是正式的
def allone_scan(srcpath=''):
    #path = "E:\\match_sample\\image\\sample\\160\\"
    path = "E:\\match_sample\\image\\coarse\\1493508047\\"
    
    if len(srcpath) > 0 :
       file_moveflag = 1
       path = srcpath
       dstpath =  "E:\\match_sample\\image\\oneall\\"
    else :
       file_moveflag = 0
       dstpath = path
       
    os.chdir(path)
    filelist = glob.glob(r'*.jpg')
    
    
    img_scanforone_all(path,filelist,dstpath=dstpath,file_moveflag = file_moveflag)
    #img_result_touch_show(path,filelist,touch)
 
 
def test_touch():
    #path = "E:\\match_sample\\image\\sample\\160\\"
    path = "E:\\match_sample\\image\\test\\"
    os.chdir(path)
    filelist = glob.glob(r'*.jpg')
    
   
    
    touch  = img_touch_all(path,filelist,scanflag = 0,debugflag =1 )
    img_result_touch_show(path,filelist,touch)

def img_coarse(srcpath='',dstpath = ''):
    #path = "E:\\match_sample\\image\\sample\\160\\"
    path = srcpath
    os.chdir(path)
    filelist = glob.glob(r'*.jpg')
    
    img_scanforone_all(path,filelist, coarse =1, dstpath= dstpath,file_moveflag= 1 )
    
def test_coarse_test():
    #path = "E:\\match_sample\\image\\sample\\160\\"
    path = "E:\\match_sample\\image\\coarse\\1493456870\\"
    os.chdir(path)
    filelist = glob.glob(r'*.jpg')
    
    img_scanforone_all(path,filelist, coarse =1,dstpath="E:\\match_sample\\image\\test\\" )
      
        
def img_deep_coarse():

    srcpath="E:\\match_sample\\image\\coarse_step2\\"
    
    sub = os.listdir(srcpath)
    
    for s in sub:
       subpath = os.path.join(srcpath,s)
       print (subpath)
       os.chdir(subpath)
       filelist = glob.glob(r'*.jpg')
       
       img_scanforone_all(subpath,filelist, coarse =1, dstpath="E:\\match_sample\\image\\coarse_step2_deep\\",file_moveflag= 1)    
       
       
    #path = "E:\\match_sample\\image\\sample\\160\\"
   
def allone_scan_auto():
    #path = "E:\\match_sample\\image\\sample\\160\\"
    path = "E:\\match_sample\\image\\coarse_step2_deep\\"
    
    sub = os.listdir(path)
    dstpath =  "E:\\match_sample\\image\\oneall\\"
    
    for s in sub:
       subpath = os.path.join(path,s)
       print (subpath)
       os.chdir(subpath)
       filelist = glob.glob(r'*.jpg')
       img_scanforone_all(subpath,filelist,dstpath=dstpath,file_moveflag = 1)
       
    
    

import sys
    
if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'coarse':
           if len(sys.argv) > 3:
             img_coarse(srcpath = sys.argv[2],dstpath =sys.argv[3] )
           else :
             print 'eer '
            #img_coarse()
        elif sys.argv[1] == 'oneall':
           if len(sys.argv) > 2:
             allone_scan(srcpath = sys.argv[2])
           else :
             allone_scan() 
        elif sys.argv[1] == 'oneall_auto':
           allone_scan_auto()
        elif sys.argv[1] == 'deep':
           img_deep_coarse()
       
    else:
       test_touch()
           
    #test_coarse()
    
    #test_coarse_test()
    
    
    
        
        
            

   