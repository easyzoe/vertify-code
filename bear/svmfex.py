#_*_ coding:utf-8 _*_  
'''
   改造成适应多种 特征 进行svm的
'''
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

from glassycom import  gcom_svm_filelist,gcom_glassylist,gcom_svm_getfilelist,gcom_svm_fexture_vector,gcom_loginit,gcom_log_svm_train

import matplotlib.pyplot as plt

#F:\picdog\glassy\



svm_positive ='positive'
svm_negative ='negative'
svm_test_negative='test_negative'
svm_test_positive='test_positive'

svm_fex_rgb = 'rgb'
svm_fex_lbp = 'lbp'
svm_fex_hsv = 'hsv'
svm_fex_hog = 'hog'

g_svm_fex_list=(svm_fex_hog,svm_fex_hsv)



## 按文件产生hog
def smfex_fexture_vecotr_hog(gpath,elm,svmfiletype,filelist):
    orientations = 9
    cell = 8
    block = 3

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
    np.savetxt(glassypath+'\\%s_%s.csv'%(svmfiletype,svm_fex_hog), veclist, delimiter=',')   # X is an array
    
def smfex_readfex_vector(glassypath,svmfiletype,fextype):
    vectors  = np.loadtxt('%s\\%s_%s.csv'%(glassypath,svmfiletype,fextype),delimiter=',',dtype='float')
    return  vectors

def  smfex_create_elm_hog(gpath,elm):
    
    glassypath = os.path.join(gpath,elm)
    
    typ =  svm_positive
    
    filelist =  gcom_svm_getfilelist(glassypath,typ)
    smfex_fexture_vecotr_hog(gpath,elm,typ,filelist)
    
    typ =  svm_negative
    filelist =  gcom_svm_getfilelist(glassypath,typ)
    smfex_fexture_vecotr_hog(gpath,elm,typ,filelist)
    
    typ =  svm_test_negative
    filelist =  gcom_svm_getfilelist(glassypath,typ)
    smfex_fexture_vecotr_hog(gpath,elm,typ,filelist)
    
    typ =  svm_test_positive
    filelist =  gcom_svm_getfilelist(glassypath,typ)
    smfex_fexture_vecotr_hog(gpath,elm,typ,filelist)
    
     
def smfex_add_label( jvmdata,label ):
    row = jvmdata.shape[0]
    labels = [label]* row
    
    output = zip(labels,jvmdata.tolist() );
    return output
    
def smfex_train_glassy(gpath,elm,cr,gr,fextype):
    
    glassypath = os.path.join(gpath,elm)
    
    posi = smfex_readfex_vector(glassypath,svm_positive,fextype)
    
    posi_data = smfex_add_label(posi,1)
    
    nega = smfex_readfex_vector(glassypath,svm_negative,fextype)
    
    nega_data = smfex_add_label(nega,-1)
    
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
        
        
def smfex_test_model(gpath,elm,m,fextype):
   
    glassypath = os.path.join(gpath,elm)
    
    test_posi = smfex_readfex_vector(glassypath,svm_test_positive,fextype)
    test_posi_label = [1]*len(test_posi)
    
    
    test_nega = smfex_readfex_vector(glassypath,svm_test_negative,fextype)
    test_naga_label = [-1]*len(test_nega)
    
    posi_label ,posi_acc,posi_val = svm_predict(test_posi_label,test_posi.tolist(),m,'-b 1') 
    nega_label ,nega_acc,nega_val = svm_predict(test_naga_label,test_nega.tolist(),m,'-b 1')     
    
    return  [posi_acc,nega_acc]
    

def smfex_svm_elm(gpath,elm,fextype):
    cr = 0.01
    gr = 0.01
    m = smfex_train_glassy(gpath,elm,cr,gr,fextype)
    
    test_acc = smfex_test_model(gpath,elm,m,fextype)
    svm_save_model('%s\\hog_model.mat'%(os.path.join(gpath,elm)) ,m)
    
   
#  
def smfex_elm_train_find_opt(gpath,elm,fextype):
    
    cr = 0.0000001
    gr = 0.0000001
    
    step = 0.005
    
    baf = 0
    smax = 0
    sstd = 0.5  # 最大的方差
    jxacc = np.zeros([2])
    
    while   1:
        
        m = smfex_train_glassy(gpath,elm,cr,gr,fextype)
    
        test_acc = smfex_test_model(gpath,elm,m,fextype)
        
        print  cr,gr, test_acc[0],test_acc[1]
        
        jxacc[0] = test_acc[0][0]/100 
        jxacc[1] = test_acc[1][0]/100
        
        print elm,fextype,cr,gr,'sum:' , jxacc.sum() ,'std:',jxacc.std() 
        #最大值，或者 均方差明显过大。   
        if  ( smax <   jxacc.sum() and sstd  > jxacc.std() ) or ( smax <   jxacc.sum() and  abs( sstd  - jxacc.std()) < 0.05  ) or  (  sstd > jxacc.std() and  sstd > 0.1  )  :
            smax = jxacc.sum()
            sstd = jxacc.std()
            goop=[cr,gr,smax]
            svm_save_model('%s\\%s_model.mat'%(os.path.join(gpath,elm),fextype) ,m)
            restr = 'cr: %lf gr:%lf posacc:%lf  negaacc:%lf  sum:%lf ' %(cr,gr,test_acc[0][0],test_acc[1][0],smax)
            gcom_log_svm_train(elm,fextype,restr)
        if baf == 0:
           cr =  cr +step
           baf = 1           
        else :
           gr = gr + step
           baf = 0
           
        if cr > 10 and gr > 10:
            break
            
       

def smfex_is_avisiable(gpath,elm,fextype,stp):
    glassypath = os.path.join(gpath,elm)
    
    if cmp(fextype, svm_fex_hog) == 0  and  stp == 0:
        
        afile = '%s.txt'%svm_positive       
           
    else :
        if stp == 1:
           afile = '%s_%s.csv'%(svm_positive,fextype)
        elif stp ==0 :
           afile = '%s.csv'%(fextype)
    return os.path.isfile(  os.path.join(glassypath,afile)  )
     

#训练所有的特征
def  smfex_train_glassy_Entry(gpath):
    
    elmlist = gcom_glassylist(gpath)
    for elm in elmlist:
        for fextype in g_svm_fex_list:
            if smfex_is_avisiable(gpath,elm,fextype,1)  == False:
               continue
            smfex_elm_train_find_opt(gpath,elm,fextype)
        

def  smfex_test_glassy_Entry(gpath,fextype):
    
    elmlist = gcom_glassylist(gpath)
    for elm in elmlist:
        if smfex_is_avisiable(gpath,elm,fextype,1)  == False:
            continue
        smfex_svm_elm(gpath,elm,fextype)
        break


        
# 为各个glassy 产生smv hog listfile
def smfex_create_hog_filelist(gpath):
    
    a = gcom_glassylist(gpath);
    
    for elm in a:
        gcom_svm_filelist(gpath,elm)
        print elm,'.....'
        #break

# 为各个glassy 产生smv 所用的 四种向量 数据
def smfex_create_vector(gpath,fextype):
    
    a = gcom_glassylist(gpath);
    
    for elm in a:
        if smfex_is_avisiable(gpath,elm,fextype,0)  == False:
                  continue
        print elm,fextype,'.....'
        if cmp(fextype, svm_fex_hog) == 0 :
           smfex_create_elm_hog(gpath,elm)
           
        else :
            
            gcom_svm_fexture_vector(gpath,elm,fextype)
        
        #break  # just for test
        
if __name__ == '__main__':

    if len(sys.argv) == 1:
        print  'useage   tt /'
    else :  
        gcom_loginit()
        if  cmp(sys.argv[1],'hogfile') == 0:
            # python -mpdb E:\picdog\bear\svmfex.py  hogfile  F:\picdog\glassy\
            smfex_create_hog_filelist(sys.argv[2])
        elif  cmp(sys.argv[1],'svm') == 0:
            # python -mpdb E:\picdog\bear\svmfex.py  svm  F:\picdog\glassy\
            smfex_train_glassy_Entry(sys.argv[2])


        elif  cmp(sys.argv[1],'svmtest') == 0:
            # python -mpdb E:\picdog\bear\svmfex.py  svm  F:\picdog\glassy\
            smfex_test_glassy_Entry(sys.argv[2],sys.argv[3])

        
        elif  cmp(sys.argv[1],'vector') == 0:
            # python -mpdb E:\picdog\bear\svmfex.py  svm  F:\picdog\glassy\
            smfex_create_vector(sys.argv[2],sys.argv[3])
        else :
           print  'useage  load /test /'
    
    
    
    print 'game over....'
    
    
    
  
    
    
    