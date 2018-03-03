#_*_ coding:utf-8 _*_  


import sys,os
import time

from glassycom import  gcom_glassylist,gcom_getfilelist
from  baidu import baidu_rg_singleimage

def tool_match_text(patha,pathb):
    aelmlist = gcom_glassylist(patha)
    
    belmlist = gcom_glassylist(pathb)
    
    
    for  a in aelmlist:
        if  a in belmlist :
          num = gcom_getfilelist(os.path.join(pathb,a))
          print a, len(num) 
        else :
          print '           .....  ',a
          cpath = os.path.join('E:\\picdog\\glassy',a)
          if os.path.isdir(cpath) ==False:
              continue
          cfilelist = gcom_getfilelist(cpath)
          for cfl  in cfilelist:
              baidu_rg_singleimage(cpath,cfl)
          
    
if __name__ == '__main__':

    if len(sys.argv) == 1:
        print  'useage   tt /'
    else :  
        if  cmp(sys.argv[1],'match') == 0:
            # python -mpdb E:\picdog\bear\hogclass.py  filelist  F:\picdog\glassy\
            # python -mpdb E:\work\bear\tool.py  match  E:\picdog\textdb E:\picdog\bdimagedb
            tool_match_text(sys.argv[2],sys.argv[3])
       
        else :
           print  'useage  load /test /'
    
    
    
    print 'game over....'