#_*_ coding:utf-8 _*_  

import numpy as np
import sys,os
import time
from scipy import ndimage
from skimage import feature
from skimage import io
from skimage import transform
from os import listdir
from os.path import isfile, join
from skimage import exposure

from svmutil import *

from glassycom import  gcom_svm_filelist,gcom_glassylist,gcom_svm_getfilelist

import matplotlib.pyplot as plt

#F:\picdog\glassy\
orientations = 9
cell = 8
block = 3


svm_positive ='positive'
svm_negative ='negative'
svm_test_negative='test_negative'
svm_test_positive='test_positive'



## 按文件产生hog
def hg_create_save_hog_byfilelist(gpath,elm,svmfiletype,filelist):
    veclist=[];
    glassypath = os.path.join(gpath,elm)   
    
    maxrow = len(filelist)
    i = 0;
    for filename in filelist:
        try:
            if cmp(svmfiletype,svm_positive) == 0 or cmp(svmfiletype,svm_test_positive) == 0  :
                afile = os.path.join(glassypath,filename)
            else :
                afile = os.path.join(gpath,filename)
                
            img = io.imread(afile, as_grey=True)
            
            vector,hogimage= feature.hog(img, orientations=orientations, pixels_per_cell=(cell, cell), cells_per_block=(block, block), transform_sqrt=True,visualise=True)
            dimension = len(vector)
            
            if len(veclist)    == 0  :
                veclist = np.zeros( (maxrow,dimension) )
                
            veclist[i] = vector;
            
            i = i+1;
                        
            #hog_image_rescaled = exposure.rescale_intensity(hogimage, in_range=(0, 0.02))
            
           
        except Exception as e:
            print "ERROR:", e
            break
    
    if i < maxrow -1  :
    
         return
    
     
    # save file
    np.savetxt(glassypath+'\\%s_hog.csv'%svmfiletype, veclist, delimiter=',')   # X is an array
    
    
def hg_create_hogs_vetor(fpath,filelist):
    veclist=[];
    maxrow = len(filelist)
    i = 0;
    for filename in filelist:
        try:
            img = io.imread(fpath +'\\' + filename, as_grey=True)
            
            vector ,hogimage= feature.hog(img, orientations=orientations, pixels_per_cell=(cell, cell), cells_per_block=(block, block), transform_sqrt=True,visualise=True)
            dimension = len(vector)
            
            if len(veclist)    == 0  :
                veclist = np.zeros( (maxrow,dimension) )
                
            veclist[i] = vector;
            
            i = i+1;
                        
            #hog_image_rescaled = exposure.rescale_intensity(hogimage, in_range=(0, 0.02))
            
           
        except Exception as e:
            print "ERROR:", e
            break
    
    if i < maxrow -1  :
    
         return [];
         
    # save file
    return veclist


def hog_loadhog(glassypath,svmfiletype):
    hogs  = np.loadtxt('%s\\%s_%s'%(glassypath,svmfiletype,'hog.csv'),delimiter=',',dtype='float')
    return  hogs

def hog_create_hog_elm(gpath,elm):
    
    glassypath = os.path.join(gpath,elm)
    
    typ =  svm_positive
    
    filelist =  gcom_svm_getfilelist(glassypath,typ)
    hg_create_save_hog_byfilelist(gpath,elm,typ,filelist)
    
    typ =  svm_negative
    filelist =  gcom_svm_getfilelist(glassypath,typ)
    hg_create_save_hog_byfilelist(gpath,elm,typ,filelist)
    
    typ =  svm_test_negative
    filelist =  gcom_svm_getfilelist(glassypath,typ)
    hg_create_save_hog_byfilelist(gpath,elm,typ,filelist)
    
    typ =  svm_test_positive
    filelist =  gcom_svm_getfilelist(glassypath,typ)
    hg_create_save_hog_byfilelist(gpath,elm,typ,filelist)
    
     
def hog_add_label( jvmdata,label ):
    row = jvmdata.shape[0]
    labels = [label]* row
    
    output = zip(labels,jvmdata.tolist() );
    return output
    
def hog_svm_train_glassy(gpath,elm,cr,gr):
    
    glassypath = os.path.join(gpath,elm)
    
    posi = hog_loadhog(glassypath,svm_positive)
    
    posi_data = hog_add_label(posi,1)
    
    nega = hog_loadhog(glassypath,svm_negative)
    
    nega_data = hog_add_label(nega,-1)
    
    trainingdata = posi_data + nega_data
    
    labels,data = zip(*trainingdata )
    
    prob = svm_problem(labels,data);
    
    param = svm_parameter("-q")
    
    param.probability = 1
          
    param.kernel_type = RBF
    
    
    param.C = cr;
    param.gamma = gr
    
    m = svm_train(prob,param)
    
    
    
    
    return m
        
        
def svm_hog_elm_test_model(gpath,elm,m):
   
    glassypath = os.path.join(gpath,elm)
    
    test_posi = hog_loadhog(glassypath,svm_test_positive)
    test_posi_label = [1]*len(test_posi)
    
    
    test_nega = hog_loadhog(glassypath,svm_test_negative)
    test_naga_label = [-1]*len(test_nega)
    
    posi_label ,posi_acc,posi_val = svm_predict(test_posi_label,test_posi.tolist(),m,'-b 1') 
    nega_label ,nega_acc,nega_val = svm_predict(test_naga_label,test_nega.tolist(),m,'-b 1')     
    
    return  [posi_acc,nega_acc]
#  
def svm_hog_elm_train(gpath,elm):
    
    cr = 0.0001
    gr = 0.0001
    
    step = 0.05
    
    baf = 0
    smax = 0
    
    while   1:
        
        m = hog_svm_train_glassy(gpath,elm,cr,gr)
    
        test_acc = svm_hog_elm_test_model(gpath,elm,m)
        
        print  cr,gr, test_acc[0],test_acc[1]
        
        if  smax <  test_acc[0][0]    +  test_acc[1][0] :
            smax = test_acc[0][0]    +  test_acc[1][0] 
            goop=[cr,gr,smax,test_acc[0],test_acc[1]]
            svm_save_model('%s\\hog_model.mat'%(os.path.join(gpath,elm)) ,m)
          
        if baf == 0:
           cr =  cr +step
           baf = 1           
        else :
           gr = gr + step
           baf = 0
           
        if cr > 10 and gr > 10:
            break
            
        print goop
    print goop
    

def  svm_glassy_hog_entry(gpath):
    
    elmlist = gcom_glassylist(gpath)
    for elm in elmlist:
       if svm_hog_is_svm_aviable(gpath,elm) ==False:
            continue
       svm_hog_elm_train(gpath,elm)
       

       
       
# 为各个glassy 产生smv
def hog_svm_create_filelist(gpath):
    
    a = gcom_glassylist(gpath);
    
    for elm in a:
        gcom_svm_filelist(gpath,elm)
        
        print elm ,'filish ....'
        #break

        # 为各个glassy 产生smv
def  svm_hog_is_svm_aviable(gpath,elm):
    glassypath = os.path.join(gpath,elm)
    if os.path.isfile(os.path.join(glassypath,'positive.txt')):
        return True
    return False        

def hog_svm_create_hog(gpath):
    
    a = gcom_glassylist(gpath);
    
    for elm in a:
        if svm_hog_is_svm_aviable(gpath,elm) ==False:
            continue
        hog_create_hog_elm(gpath,elm)
        print elm ,'filish ....'
        #break
        
if __name__ == '__main__':

    if len(sys.argv) == 1:
        print  'useage   tt /'
    else :  
        if  cmp(sys.argv[1],'filelist') == 0:
            # python -mpdb E:\picdog\bear\hogclass.py  filelist  F:\picdog\glassy\
            # python  E:\work\bear\hogclass.py  filelist  E:\picdog\glassy  
            hog_svm_create_filelist(sys.argv[2])
        elif  cmp(sys.argv[1],'svmhog') == 0:
            # python -mpdb E:\picdog\bear\hogclass.py  svm  F:\picdog\glassy\
            # python  E:\work\bear\hogclass.py  svmlog  E:\picdog\glassy  
            svm_glassy_hog_entry(sys.argv[2])
        elif  cmp(sys.argv[1],'hog') == 0:
            # python -mpdb E:\picdog\bear\hogclass.py  svm  F:\picdog\glassy\
            # python -mpdb E:\work\bear\hogclass.py  hog  E:\picdog\glassy  
            
            hog_svm_create_hog(sys.argv[2])
        else :
           print  'useage  load /test /'
    
    
    
    print 'game over....'
    
    
  
    
    
    