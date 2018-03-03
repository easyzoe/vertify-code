# -*- coding:utf-8 -*-

import glob,os,shutil,sys
import redis
import numpy as np
from skimage import io
from scipy.spatial import distance


from imgnormal import img_macro_feature_grow,img_macro_watershed,img_marco_baiduimg_hdf5,img_macro_get_textfeat,img_macro_feat ,com_judge_feature
from rgtitle  import rgtt_rgtitle_passcode,rgtt_test_file_passcode
from glassycom import rgim_get_sub_img,rgim_get_sub_npim_byseq

import yaml


exec_path ="E:\\picdog\\"
roughimage='roughimage'

def frg_coreconfig_init():
    fp = open("%scoreconfig.yaml"%exec_path,'r')
    config = yaml.load(fp)
    
    gtext = {}
    for textid,text in config['textlist'].items():
         gtext[text] = textid
         gtext[textid] =  text
    return  gtext
def test_single_file():
    path = "E:\\picdog\\pscode\\"
    
    r = redis.Redis(host='localhost',port=6379,db=0)
    
    gtext = frg_coreconfig_init()
    
    disth = 0.15
    
    afile = "1466234566_3.jpg"
    rgtexts = rgtt_test_file_passcode(r,path,afile)

def test_pscode():
    path = "E:\\picdog\\pscode\\"
    filelist = os.listdir(path)
    r = redis.Redis(host='localhost',port=6379,db=0)
    
    gtext = frg_coreconfig_init()
    
    disth = 0.15
    
    for afile in filelist:
        rgtexts = rgtt_test_file_passcode(r,path,afile)
        print rgtexts, afile
        continue 
        
        pscode = io.imread(os.path.join(path,afile) )
        
        imagetag = {}
        imagetagval = {}
        
        for text in rgtexts:
            textid = gtext[text]
            print text,textid
            res = img_macro_get_textfeat(u"E:\\picdog\\baidu_rough",text)   
            
            imgflag = np.zeros(8)
            
            for i in range(1,9):
                sbimg = rgim_get_sub_npim_byseq(pscode,i)
                iimg_feat = img_macro_feat(sbimg)
                betterdis = beter_in_batchset(iimg_feat,res[1] )
                imgflag[i -1 ] = betterdis
                
               # print betterdis
                  
            #     
            imgflag < 0.1
            sortindex = np.argsort(imgflag)[:4]
            select = sortindex[ imgflag[sortindex] < 0.15 ]
            
            print "rough select ", select
            for k in select :
                if imagetag.has_key(k+1) :
                   #imagetag[k+1].append(textid)
                   if imgflag[k] <  imagetagval[k+1]  :
                      imagetag[k+1] = [textid]
                      imagetagval[k+1] = imgflag[k]                   
                else :
                   imagetag[k+1] = [textid]
                   imagetagval[k+1] = imgflag[k]
            
        # Tag image 
        for i in range(1,9):
            sbimg = rgim_get_sub_npim_byseq(pscode,i)
            if  imagetag.has_key(i) :
                 astr = gtext[imagetag[i][0]]
                 sbimag_name =  astr  +  "_s_%d_"%i +   afile
            else :
                 sbimag_name =   "_s_%d_"%i +   afile
            io.imsave(os.path.join('%s%s'%(exec_path,roughimage) ,sbimag_name), sbimg )
                
            
def  beter_in_batchset(imgfeat,featset):  
     mindis = 1           
     for kfeat in featset:
         featlist=[['wholehist',0.5]]
         #resdis = com_judge_feature(imgfeat,kfeat,featlist)
         featdis = distance.euclidean(imgfeat['wholehist'],kfeat['wholehist'])
         
         if mindis  > featdis:
              mindis = featdis
     return mindis
     
                    

if __name__ == "__main__":

    #img_macro_feature_grow(u"E:\\picdog\\baidu_rough\\七星瓢虫\\21a4462309f79052a481851406f3d7ca7bcbd57a.jpg")
    #img_marco_baiduimg_hdf5(u"E:\\picdog\\baidu_rough", u"E:\\picdog")
    
    #res = img_macro_get_textfeat(u"E:\\picdog\\baidu_rough",u'七星瓢虫')
    test_pscode()
    #test_single_file()