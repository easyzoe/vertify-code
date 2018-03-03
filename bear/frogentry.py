# -*- coding: utf-8 -*-
import os,glob
import yaml

from imagebytext  import  Searchimage_bytext

exec_path ="E:\\picdog\\"
rough = "baidu_rough"

# 根据textdb 目录生成 配置文件
#  --- just run once 
def  frg_create_core_config():
    path = "%stextdb\\"%exec_path
    files = os.listdir(path)
    
    idstart = 17000
    

    catalog = {}
    i = 0
    for f in files:
        af = os.path.join(path,f)
        if os.path.isdir(af) :
          id = idstart +i 
          catalog[id] = f.decode('gbk')
          
          i =  i+ 1
          
    config = {}
    config['textlist']   =   catalog
   
    fp = open("%scoreconfig.yaml"%exec_path,'w')
    yaml.dump(config,fp)
        
#  从百度上下载图片，并且格式化成 标准大小，缩小
def frg_download_image_from_baidu():
    imagenum = 50
    fp = open("%scoreconfig.yaml"%exec_path,'r')
    config = yaml.load(fp)
    
    for textid,text in config['textlist'].items():
        #dstpath = exec_path + rough +"\\" + "%d"%textid
        dstpath = exec_path + rough +"\\" + "%s"%text
        #Searchimage_bytext(text.encode('gbk'),dstpath,imagenum)
        Searchimage_bytext(text,dstpath,imagenum)
    
def frg_coreconfig_init():
    fp = open("%scoreconfig.yaml"%exec_path,'r')
    config = yaml.load(fp)
    
    gtext = {}
    for textid,text in config['textlist'].items():
         gtext[text] = textid
         gtext[textid] =  text
    return  gtext
    
 
def frg_normalfilename():
    fp = open("%scoreconfig.yaml"%exec_path,'r')
    config = yaml.load(fp)
    
    for textid,text in config['textlist'].items():
        dstpath = exec_path + rough +"\\" + "%s"%text
        
        filelist = os.listdir(dstpath)
        for afile in filelist:
            if afile.endswith(".jpg") :
                os.rename(os.path.join(dstpath,afile), os.path.join(dstpath,  "%d_%s"%(textid,afile)  ) )  
                
                
frg_normalfilename()            
        