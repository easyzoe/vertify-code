#_*_ coding:utf-8 _*_  

from  fexture import ttext_title_readfrom_redis,ttext_readfrom_redis,ttext_touch,ttext_filelist
from  glassycom import  gcom_bwcorr ,gcom_bwread,gcom_getfilelist,gcom_log_pscode_rg

from  pstext import pigtext,pigtext_data

from skimage import io
from PIL import Image

#11.12  discard  matlab api.
#import mlab
#from mlab.releases import latest_release as matlab

import os
import sys,shutil
import redis
import numpy as np; 
import time

path_midpro = 'E:\\picdog\\midp\\'

matlab_return_path='F:\\picdog\\Errglassy\\'   # 暂时固定，需要调�?
online_learn_text_path='E:\\picdog\\onlearn\\text\\'  #记录下text
online_learn_text_flag = 1


def rgtt_getTextelm_textpng(r,textelm,filename):
    return  ttext_title_readfrom_redis(r,textelm,filename)


def  rgtt_search_in_textelm(r,textelm,A):
    
    wordlen = len(textelm);
    touch  = ttext_readfrom_redis(r,textelm,ttext_touch)
    
    tchfilelist = ttext_readfrom_redis(r,textelm,ttext_filelist)
    
    if len(touch) == 0 or len(tchfilelist) == 0:
        return []
  
    basicnum = touch.shape[0];
    
    flag =  np.zeros(basicnum,'int8')
    
    thr_start = 0.5;
    
    hole = touch;
    hole = np.where(  hole <=  thr_start  ,0,1 );
    
    cordlist=np.array([]);
       
    max_i = 0;
    max_cor = 0;
       
    compnum = 0;
       
    optcfile = ''
       
    for i in range(basicnum):
           
           if flag[i] == 1 :
               continue
           
           
           tseq = np.argwhere(hole[i,:] ==1 );
           
           if len(tseq) == 0:
               break;
           
 
           Cfile = tchfilelist[i][0];
   
           C = rgtt_getTextelm_textpng(r,textelm,Cfile)
          
           corc = gcom_bwcorr(A,C);
           
           compnum = compnum +1;
           
                     
           if corc  >max_cor :
               max_cor= corc;
               max_i = i;
               optcfile  = Cfile
           
           
           
           flag[i] =1;
           
           if corc < 0.45 :
               for k in range(len(tseq) ) :
                  flag[ tseq[k][0] ] = 1 
  
           
           #if corc <= thr_start
             
                     
    opt=[max_i,max_cor,compnum,optcfile];
    return opt
    #thlist = cordlist;
   
#
def rgtt_save_learn_text(Apath,Afile,dstpath,text,topmax):
    
    normfile = os.path.join(Apath,'%s.jpg'%(Afile[:-4]) ) 
    textpng = os.path.join(Apath,Afile)
    os.remove( textpng)
    
    if online_learn_text_flag ==  0 or  topmax > 0.65:
       
       os.remove( normfile) 
       
       return 
    # 匹配很高的就可以no保存
    if topmax <= 0.3 :
        textpath = os.path.join(dstpath,'uncertain')
    else :
        textpath = os.path.join(dstpath,text)
        
    if os.path.isdir(textpath):
       pass;
    else:
       os.mkdir(textpath)  
    try:
      #shutil.move(textpng,os.path.join(textpath,Afile))
      # 
      shutil.move(normfile,os.path.join(textpath,'%s.jpg'%(Afile[:-4]) ) )
    except: 
       return 
    return        
    
   
# 在所有的已知的列表中查找匹配，目标文件路径，目标text的长�?

def rgtt_search_in_allTextElm(r,Apath,Afile,wordlen):

    re,A = gcom_bwread(os.path.join(Apath,Afile))    
     
    if re != 1:
        return []
        
    optlist = [];
    
    topopt='';
    topmax=0;
    topCfile = ''
    
    res=r.smembers('set_ttext_%d'%wordlen)
    
    for textelm in res:
        selm = textelm.decode('utf-8')
        opt = rgtt_search_in_textelm(r,selm,A)
        optlist.append(opt);
        
        if topmax <  opt[1]:
           topmax = opt[1]
           topopt = selm
           topCfile = opt[3]
    
    tmop = np.array(optlist)     
    
    print topopt, topmax
    gcom_log_pscode_rg(Afile,topopt,'recong text feasibility %f %s'%(topmax,topCfile) )
    
    
    #rgtt_save_learn_text(Apath,Afile,online_learn_text_path,topopt,topmax)
     
    return  topopt;

def rgtt_search_in_allTextElm_data(r,textdata,wordlen):

    A = textdata    
      
    optlist = [];
    
    topopt='';
    topmax=0;
    topCfile = ''
    
    res=r.smembers('set_ttext_%d'%wordlen)
    
    for textelm in res:
        selm = textelm.decode('utf-8')
        opt = rgtt_search_in_textelm(r,selm,A)
        optlist.append(opt);
        
        if topmax <  opt[1]:
           topmax = opt[1]
           topopt = selm
           topCfile = opt[3]
    
    tmop = np.array(optlist)     
    
    print topopt, topmax
    gcom_log_pscode_rg(textdata,topopt,'recong text feasibility %f %s'%(topmax,topCfile) )
    
    
    #rgtt_save_learn_text(Apath,Afile,online_learn_text_path,topopt,topmax)
     
    return  topopt;


# 在所有的已知的列表中查找匹配，目标文件路径，目标text的长�?

def rgtt_search_in_allTextElm_return_value(r,Apath,Afile,wordlen):

    re,A = gcom_bwread(os.path.join(Apath,Afile))    
     
    if re != 1:
        return []
        
    optlist = [];
    
    topopt='';
    topmax=0;
    
    res=r.smembers('set_ttext_%d'%wordlen)
    
    for textelm in res:
        selm = textelm.decode('utf-8')
        opt = rgtt_search_in_textelm(r,selm,A)
        optlist.append(opt);
        
        if topmax <  opt[1]:
           topmax = opt[1]
           topopt = selm
    
    tmop = np.array(optlist)     
    
    #print topopt, topmax
     
    return  topmax;
    
#从所有以知的title中查找最合适的    
def rgtt_rgtitle_passcode(r,pscode,fname):

    #reg=(120,0,290,30)  # left up- corrd , right-down corrd
    reg=(118,0,290,28)
    im= pscode.crop(reg)
    titlef = '%s\\ol_%s'%(path_midpro,fname);
    im.save(titlef)
    
    texts =  np.asarray (im)
    
    normtexts = pigtext_data(texts)
    
    os.remove(titlef)
    tmp = ''
    for w in  range(len(normtexts)):
    
        wordlen = normtexts[w][0]
        word    = normtexts[w][1]
        
        io.imsave('%s\\%d_ol_%s'%(path_midpro,w,fname),word*255 )
                 
        optxt = rgtt_search_in_allTextElm_data(r,word,wordlen )
        if len(optxt) ==0 :
           return  [];
        
        if len(tmp) == 0:
            tmp = optxt
        else:
            tmp=tmp+'_'+optxt
        
        # 必须删除文件
        #rgtt_search_in_allTextElm h函数中处�?
        #os.remove( 'F:\\picdog\\Errglassy\\%s' % textnames[w])
       
    if len(tmp) == 0:
       return []
      
    rgtitle = tmp.split('_')     
   # print  textnames ,'    ' , rgtitle
    return rgtitle
    
    
def rgtt_test_file_passcode(r,fpath,fname):
    
    pscode = Image.open( os.path.join(fpath,fname) )
    
    return rgtt_rgtitle_passcode(r,pscode,fname)
         

#  用于挑选匹配概率最小的title ，该类很可能是未知的文字�?
def rgtt_colect_maybenewtext(r,rootpath,normsub,uncertain):
    
    Apath = os.path.join(rootpath,normsub)
    Bpath = os.path.join(rootpath,uncertain)
    filelist = gcom_getfilelist(Apath)
    
    for afile in filelist:
        
        opval = rgtt_search_in_allTextElm_return_value(r,Apath,afile,int( afile[-5:-4] ))
        
        if opval <= 0.5 :
            print afile
            #shutil.move(os.path.join(Apath,afile),os.path.join(Bpath,afile) )
        elif opval > 0.5 :
            os.remove(os.path.join(Apath,afile))        
         
if __name__ == '__main__':
         
    sys.argv[0]
    if len(sys.argv) == 1:
        print  'useage   tt /'
    else : 
        r = redis.Redis(host='localhost',port=6379,db=0)
      
       
        if  cmp(sys.argv[1],'tt') == 0 and len(sys.argv) == 4:
            rgtt_test_file_passcode(r,sys.argv[2],sys.argv[3])   
  
        elif   cmp(sys.argv[1],'uncert') == 0 and len(sys.argv) == 5:
            while 1 :
               rgtt_colect_maybenewtext(r,sys.argv[2],sys.argv[3],sys.argv[4])
               time.sleep(30)
        else :
           print  'useage  load /test /'
    
    
    
    print 'game over....'
    
    
    
    
       