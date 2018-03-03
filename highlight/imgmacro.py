# -*- coding:utf-8 -*-

import glob,os,shutil,sys


from imgnormal import img_macro_feature_grow,img_macro_watershed,img_marco_baiduimg_hdf5,img_macro_get_textfeat

def test_origin_baidufile():
     path = u"E:\\picdog\\baidu_rough\\七星瓢虫\\"
     filelist = os.listdir(path)
     for afile in filelist:
         if afile.endswith(".jpg"):
              try :
                 img_macro_feature_grow(os.path.join(path,afile).encode('gbk' ))
              except:
                continue

if __name__ == "__main__":

    #img_macro_feature_grow(u"E:\\picdog\\baidu_rough\\七星瓢虫\\21a4462309f79052a481851406f3d7ca7bcbd57a.jpg")
    #img_marco_baiduimg_hdf5(u"E:\\picdog\\baidu_rough", u"E:\\picdog")
    
    res = img_macro_get_textfeat(u"E:\\picdog\\baidu_rough",u'七星瓢虫')