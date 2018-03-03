#_*_ coding:utf-8 _*_ 
from PIL import Image

import os
import sys
import redis
import numpy as np; 



if __name__ == '__main__':
    sys.argv[0]
    if len(sys.argv) == 1:
        print  'useage    /'
    else : 
        r = redis.Redis(host='localhost',port=6379,db=0)
      
       
        if  cmp(sys.argv[1],'simage') == 0 and len(sys.argv) == 5:
            rgim_test_passcode(r ,sys.argv[2],sys.argv[3],sys.argv[4].decode('gbk'))   
  
       
        else :
           print  'useage  load /test /'
    
    
    
    print 'game over....'    