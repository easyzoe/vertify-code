#_*_ coding:utf-8 _*_  
# common function  

from os import listdir,mkdir
from os.path import isfile, join,isdir
import random
import codecs
import numpy as np;
from PIL import Image
import logging

_Debug_option = 0

def gcom_debug():
    if  _Debug_option == 1:
        return True
    else :
       return False    

# 从一个目录中获取文件列表
def  gcom_getfilelist(fpath):
    filenames = [ f for f in listdir(fpath) if isfile(join(fpath,f)) and f[0] != "." and ( cmp(f[-3:],'jpg') == 0  or cmp(f[-3:],'png') == 0 ) ]  
    return filenames

#获取中文目录命列表
def  gcom_glassylist(path):
    
    dflist= listdir(path)
    subtext=[];
    for elm in dflist:
       if cmp(elm,'a') == 0   or cmp(elm,'b') == 0 or isfile(join(path,elm)) :
          print elm
          continue
       elm = elm.decode('gbk')
       subtext.append(elm)       
       #print elm
    return   subtext    

# 将python list 写入文件中
def gcom_writelist(glassypath,name,alist):
   
    fl=  codecs.open(join(glassypath,name), 'wb', 'utf-8')
    for i in alist:
       fl.write(i)
       fl.write("\n")
    fl.close()
    
    #f = codecs.open(fn,'r','utf-8') 
    #ls = [ line.strip() for line in f] 
    
# 从文件中读取内容，形成list ，本身是list类型的数据
def gcom_readlist(glassypath,name) :
    f = codecs.open(join(glassypath,name),'r','utf-8') 
    ls = [ line.strip() for line in f] 
    f.close() 
    return ls

####################################### 以上为文件类处理函数
#向量计算欧式距离
def  gcom_ousdis(va,vb):
    dis = np.linalg.norm(va - vb)
    dis = round(dis,5)
    return dis
    
# 向量计算明式距离 
def gcom_msdis(phashA,phashB):
    num = 0;
    for i in range(len(phashA ) ) :
       if  phashA[i] <> phashB[i]:
          num = num +1;
    return  num;          
    
    

# 二值图像的读取函数 ，能否简化，调用lib库
def gcom_bwread(filename):
    try:
      im=Image.open(filename)
    except IOError:
       return (0,None)
    lx,ly=im.size;
    
    a= np.zeros((lx,ly),np.int)
    for i in range(0,lx):
       for j in range(0,ly):
           if im.getpixel((i,j)) == 255:
               a[i,j] = 1
                        
    a=a.T
    return (1,a)
    
    
#二值图像的相似度，关联系数计算函数，类似matlab 的corr2
def gcom_bwcorr(a,b):
    if a.shape != b.shape:
        return -1;
        
    av = np.mean(a);
    bv = np.mean(b);
    
   # a = double(a);
   # b = double(b)
   # a.astype = 'float'
   # b.astype = 'float'
    
    a = a - av;
    b = b- bv;
    
    pu = np.sum( a *b )
    
    pd = np.sqrt(    np.sum(a*a) * np.sum( b* b )      )
    
    r = pu/pd;
    
    return  r;    
    


def rgim_get_sub_img(im, x, y):
    assert 0 <= x <= 3
    assert 0 <= y <= 2
    WITH = HEIGHT = 68
    left = 5 + (67 + 5) * x
    top = 41 + (67 + 5) * y
    right = left + 67
    bottom = top + 67
    return im.crop((left, top, right, bottom))

def rgim_get_sub_img_byseq(im, seq):
   #ss = '%d'%((y*4+x +1))

    y = (seq-1)/4
    x = (seq-1)-y*4
    
    assert 0 <= x <= 3
    assert 0 <= y <= 2
    WITH = HEIGHT = 68
    left = 5 + (67 + 5) * x
    top = 41 + (67 + 5) * y
    right = left + 67
    bottom = top + 67
    return im.crop((left, top, right, bottom))

 ## 获取np 格式 的子图片   
def rgim_get_sub_npim_byseq(npim, seq):
   #ss = '%d'%((y*4+x +1))

    y = (seq-1)/4
    x = (seq-1)-y*4
    
    assert 0 <= x <= 3
    assert 0 <= y <= 2
    WITH = HEIGHT = 68
    left = 5 + (67 + 5) * x
    top = 41 + (67 + 5) * y
    right = left + 67
    bottom = top + 67
    return npim[top:bottom,left:right]
    
def gcom_save_subimage_byseq(destpath,text,fpath,fname,seq):
    
    dsubpath =  join(destpath,text)
    
    if isdir( dsubpath) :
        pass
    else :
        mkdir(dsubpath)
    
    afile = '%s_%d.jpg'%(fname[:-4],seq)
    dstfile = join(dsubpath,afile)
    
    pscode = Image.open(join(fpath,fname) )
    sim = rgim_get_sub_img_byseq(pscode,seq) 
    sim.save(dstfile)
    
######## 以上为图像类的补充处理函数



    
# 生成供svm 使用的文件列表 记录文件，正/非  测试正，测试非     
def gcom_svm_filelist(gpath,glassy):
    rat = 0.7
    
    negarat = 1.2
    
    gfnlist=gcom_getfilelist(join(gpath,glassy) );
    
    allnum= len(gfnlist)
    
    if allnum < 10   :
        return
    ##    
    ##if fntype == 0:
    
    posinum = int(allnum*rat )
    
    testnum = allnum - posinum
    
    posifile = gfnlist[:posinum]
    
    neganum = int (  posinum * negarat )
    
    negafile = gcom_svm_negative(gpath,glassy,neganum)
    
    test_posinum = testnum
    test_neganum = int ( test_posinum  * negarat)
    
    test_posifle = gfnlist[posinum:]
    
    test_negafile = gcom_svm_negative(gpath,glassy,test_neganum)
    
    
    gcom_writelist(join(gpath,glassy),'positive.txt',posifile)
    gcom_writelist(join(gpath,glassy),'negative.txt',negafile)
    gcom_writelist(join(gpath,glassy),'test_negative.txt',test_negafile)
    gcom_writelist(join(gpath,glassy),'test_positive.txt',test_posifle)
    
# 从glassy 列表中，抽取部分作为 非匹配 类型的列表。
def gcom_svm_negative(gpath,glassy, num):
        
    subglassy =  gcom_glassylist(gpath)
    it = 0;
    
    negalist = []
    
    while 1:
        for elm in subglassy:
            if cmp(elm,glassy ) == 0 :
                continue
            
            gfnlist=gcom_getfilelist(join(gpath,elm) ); 
            ch = random.choice(gfnlist)
            ng = elm+'\\'+ch
            negalist.append(ng)
            it = it +1
            
            if it >= num :
               return  negalist
    
    
# 读取目录下不同类型的文件列表
def gcom_svm_getfilelist(glassypath,vecottype):
    flist = []
    if cmp(vecottype,'positive') ==0 :
       flist= gcom_readlist(glassypath,'positive.txt') 
    elif  cmp(vecottype,'negative') ==0 :  
       flist= gcom_readlist(glassypath,'negative.txt') 
    elif  cmp(vecottype,'test_negative') ==0 : 
       flist= gcom_readlist(glassypath,'test_negative.txt') 
    elif  cmp(vecottype,'test_positive') ==0 : 
       flist= gcom_readlist(glassypath,'test_positive.txt') 
    return flist
 
#################################
# 直接获取向量的api
# 从glassy 列表中，抽取部分作为 非匹配 类型的列表。
def gcom_svm_fexvector_negative(gpath,glassy, num,fextype):
        
    subglassy =  gcom_glassylist(gpath)
    it = 0;
    
    negalist = []
    
    
    
    while 1:
        for elm in subglassy:
            if cmp(elm,glassy ) == 0 :
                continue
            
            elmvectors =  gcom_readlist(join(gpath,glassy),'%s.csv'%fextype)
            
            ch = random.choice(elmvectors)
        
            negalist.append(ch)
            it = it +1
            
            if it >= num :
               return  negalist
    
    
    
def gcom_svm_fexture_vector(gpath,glassy,fextype):
    rat = 0.7
    
    negarat = 1.2
    
    elmvectors =  gcom_readlist(join(gpath,glassy),'%s.csv'%fextype)
    
    
    allnum= len(elmvectors)
    
    if allnum < 10   :
        return
    ##    
    ##if fntype == 0:
    
    posinum = int(allnum*rat )
    
    testnum = allnum - posinum
    
    posi  = elmvectors[:posinum]
    test_posi = elmvectors[posinum:]
    
    
    neganum = int (  posinum * negarat )
    
    
    
    nega= gcom_svm_fexvector_negative(gpath,glassy,neganum,fextype)
    
    test_posinum = testnum
    test_neganum = int ( test_posinum  * negarat)
    
    
    test_nega = gcom_svm_fexvector_negative(gpath,glassy,test_neganum,fextype)
    
    
    gcom_writelist(join(gpath,glassy),'positive_%s.csv'%fextype,posi)
    gcom_writelist(join(gpath,glassy),'negative_%s.csv'%fextype,nega)
    gcom_writelist(join(gpath,glassy),'test_positive_%s.csv'%fextype,test_posi)
    gcom_writelist(join(gpath,glassy),'test_negative_%s.csv'%fextype,test_nega)
    
def gcom_loginit():
    logging.basicConfig(level=logging.INFO ,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='E:\\picdog\\fexsvm.log',
                filemode='a+')  
                
def gcom_online_loginit():
    logging.basicConfig(level=logging.INFO ,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='E:\\picdog\\onlearn.log',
                filemode='a+')  
######################
def gcom_log_svm_train(elm,fextype,result):
    
    str = 'svm train %s %s %s '%(elm,fextype,result)
    
    logging.info(str)

def gcom_log_pscode_rg(fname,text,logstr):
    
    str = '%s %s %s '%(fname,text,logstr)
    
    logging.info(str)
    
def gcom_log_online(fname,str):
    logging.info('%s %s '%(fname, str) )   
    
if __name__ == '__main__':
   
      
    pass