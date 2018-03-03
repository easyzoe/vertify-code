#_*_ coding:utf-8 _*_  
'''
   自动更新title 
'''
from  glassycom import  gcom_bwcorr ,gcom_bwread,gcom_getfilelist,gcom_log_pscode_rg,gcom_glassylist
from PIL import Image
import mlab
from mlab.releases import latest_release as matlab
import os
import sys,shutil
import redis
import numpy as np; 

from fexture import ttext_update_elm

#
def test():
    png = matlab.pigword('E:\\picdog\\onlearn\\atext\\a\\','ol_1475718124_1_1.jpg')  

def tup_png_elm(r,srcpath,elm,dstpath):
    Apath = os.path.join(srcpath,elm)
    Bpath = os.path.join(dstpath,elm)
    Apath = '%s\\'%(Apath)
    filelist = gcom_getfilelist(Apath)
    
    if len(filelist ) == 0:
        return 
        
    if os.path.isdir(Bpath):
       pass;
    else:
       os.mkdir(Bpath)  
    
    
    for Afile in filelist:
        if len(elm) != int(Afile[-5:-4] ):
            continue
        try:
          png = matlab.pigword(Apath,Afile) 
        # ol_1475805445_2_3_1_3.png
        except:
           continue
        #  
         
        shutil.move(  os.path.join(Apath,png), os.path.join(Bpath,png) )
    
    # 调用更新touch info
    print 'update touchinfo .....'
    matlab.update_touchinfo(dstpath,elm)    
    
    print 'update into redis .....'
    ret = ttext_update_elm(r,dstpath,elm)
    
    print 'ttext update to redis ',ret
                     
    return

def tup_png_elms(r,srcpath,dstpath):
    
    elmlist = gcom_glassylist(srcpath)    
    
    for elm in elmlist:
        print elm
        tup_png_elm(r,srcpath,elm,dstpath)

        
    
        

if __name__ == '__main__':
         
    sys.argv[0]
    if len(sys.argv) == 1:
        print  'useage   tt /'
    else : 
        r = redis.Redis(host='localhost',port=6379,db=0)
      
       
        if  cmp(sys.argv[1],'update') == 0 :
            #test()
            tup_png_elms(r,'E:\\picdog\\onlearn\\atext\\','F:\\picdog\\textdb\\')
          
        else :
           print  'useage  load /test /'
    
    
    
    print 'game over....'

