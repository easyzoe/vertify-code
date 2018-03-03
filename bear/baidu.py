from PIL import Image
from PIL import ImageFilter
import urllib
import urllib2
import requests
import re
import json

import os,sys

from glassycom  import rgim_get_sub_img,gcom_glassylist,gcom_getfilelist,gcom_writelist,gcom_readlist,gcom_save_subimage_byseq


def baidu_image_upload(im):
    UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2272.89 Safari/537.36"
    url = "http://image.baidu.com/pictureup/uploadshitu?fr=flash&fm=index&pos=upload"

    im.save("./query_temp_img.png")
    raw = open("./query_temp_img.png", 'rb').read()

    files = {
        'fileheight'   : "0",
        'newfilesize'  : str(len(raw)),
        'compresstime' : "0",
        'Filename'     : "image.png",
        'filewidth'    : "0",
        'filesize'     : str(len(raw)),
        'filetype'     : 'image/png',
        'Upload'       : "Submit Query",
        'filedata'     : ("image.png", raw)
    }

    resp = requests.post(url, files=files, headers={'User-Agent':UA})

    #  resp.url
    redirect_url = "http://image.baidu.com" + resp.text
    
    os.remove("./query_temp_img.png")
    return redirect_url



def baidu_stu_lookup(im):
    redirect_url = baidu_image_upload(im)

    #print redirect_url
    resp = requests.get(redirect_url)

    html = resp.text

    return baidu_stu_html_extract(html)


def baidu_stu_html_extract(html):
    pattern = re.compile(r"'multitags':\s*'(.*?)'")
    matches = pattern.findall(html)
    if not matches:
        return '[ERROR?]'

    tags_str = matches[0]

    result =  list(filter(None, tags_str.replace('\t', ' ').split()))

    return '|'.join(result) if result else '[UNKOWN]'

def baidu_rg_singleimage(fpath,fname):
    sng = Image.open( os.path.join(fpath,fname) )
    try:
        result = baidu_stu_lookup(sng)
        print result
    except :
        return
        
	return   

def baidu_rgimage(fpath,fname):
    pscode = Image.open( os.path.join(fpath,fname) )
    bodyname ={};
    for y in range(2):
        for x in range(4):
            im2 = rgim_get_sub_img(pscode, x, y)    
            ss = '%d'%((y*4+x +1))
            bodyname[ss] = []
			
            try:
                result = baidu_stu_lookup(im2)
            except :
			    continue 
		    
            try:
			    print ss, result 
            except:
                  continue
			   
            bodyname[ss]=result
            
    return  bodyname        

import glob

def dir_baidu(srcpath):
    os.chdir(srcpath)
    filelist = glob.glob(r'*.jpg')
    for i in range(len(filelist)):
       bodyname = baidu_rg_singleimage(srcpath,filelist[i])
       print bodyname       
    
    
def  baidu_pscode_any(fpath,textlst,dstpath):
    filelist = gcom_getfilelist(fpath)
    skip = 0 
    for afile in filelist:
        skip =  skip +1
     
        
        bodyname = baidu_rgimage(fpath,afile)
        
        for key,val in bodyname.items():
            if len(val) == 0:
                continue
            m = {}
            w = val.split('|')
            for  sw in w :     # sw
               for a in textlst:
                  alist = a.split('|')
                  for ai in alist :   #ai
                     if len(ai) == 0:
                         continue
                     if cmp(sw,ai) == 0 :
                        
                        m[alist[0]] = 1;
           
            if len(m) ==  1:
                print afile ,key ,m.keys()[0]            
                gcom_save_subimage_byseq(dstpath,m.keys()[0],fpath,afile,int(key))    
        try:
           os.remove(os.path.join(fpath,afile))
        except:
            continue
#   
if __name__ == '__main__':
    sys.argv[0]
    if len(sys.argv) == 1:
        print  'useage    /'
    else : 
        
#python -mpdb  E:\picdog\bear\baidu.py batch F:\pichog\passcode F:\picdog\textdb F:\pichog\test
     
       
        if  cmp(sys.argv[1],'image') == 0 and len(sys.argv) == 4:
            
             print  baidu_rgimage(sys.argv[2],sys.argv[3])   
        elif cmp(sys.argv[1],'dir') == 0 and len(sys.argv) == 3:
             dir_baidu(sys.argv[2])
        elif cmp(sys.argv[1],'batch') == 0 and len(sys.argv) == 5:            
            while 1:
               textlst  =  gcom_readlist(sys.argv[3],'text.txt')  # gcom_glassylist(sys.argv[3])
               #gcom_writelist(sys.argv[3],'text.txt',textlst)
               baidu_pscode_any(sys.argv[2],textlst,sys.argv[4])   
       
        else :
           print  'useage  load /test /'
    
    
    
    print 'game over....'       
    
    