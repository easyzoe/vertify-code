#_*_ coding:utf-8 _*_  

import numpy as np;
import os,glob
from  pstext import pigtext,text_knn_cluster,new_split_word
from PIL import Image
from frogentry import frg_create_core_config,frg_download_image_from_baidu

def tc_gettextimg_frompscode(afile):
    pscode = Image.open( afile )
    reg=(118,0,290,28)  # left up- corrd , right-down corrd
    path_midpro = "E:\\picdog\\testout"
    im= pscode.crop(reg)
    titlef = '%s\\ol_%s'%(path_midpro,os.path.basename(afile));
    im.save(titlef)
           
    #tmp = matlab.pigtext5(path_midpro,'ol_%s'%fname)   
    tmp = pigtext("%s\\"%path_midpro,'ol_%s'%os.path.basename(afile))
    
    os.remove(titlef)
    


if __name__ == '__main__':
    
    path ="E:\\picdog\\pscode"
    filelist = glob.glob(r'%s\\*.jpg'%path)
    afile="E:\\picdog\\pscode\\1475847330.jpg"
    afile="E:\\picdog\\pscode\\1475843931.jpg"   # yiliu
    afile="E:\\picdog\\pscode\\1475846377.jpg"
    
    
    frg_download_image_from_baidu()
    
    #for  afile in filelist:
    #if 1:
    #    tc_gettextimg_frompscode(afile)
    
    path ="E:\\picdog\\testout"
    wn = 3
    filelist = glob.glob(r'%s\\*_%d_py.jpg'%(path,wn))
    #new_split_word(os.path.join(path,filelist[4]),wn)
    for  i in range(len(filelist)):
         afile = filelist[i]
        # new_split_word(os.path.join(path,afile),wn)
         
    #afile = filelist[51]
    #new_split_word(os.path.join(path,afile),wn)
         
    
    
    filelist = glob.glob(r'%s\\*_%d.png'%(path,wn))
    #text_knn_cluster(filelist,"E:\\picdog\\ctextout\\",wn)
    
#standyb  E:\picdog\pscode\ol_1475843885.jpg