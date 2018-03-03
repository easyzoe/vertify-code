#_*_ coding:utf-8 _*_ 

from  glassycom import  gcom_bwcorr ,gcom_bwread,gcom_ousdis,gcom_msdis,rgim_get_sub_img,gcom_log_pscode_rg,gcom_save_subimage_byseq,gcom_loginit,gcom_glassylist
from PIL import Image

import os
import sys
import redis
import numpy as np; 
from svmutil import *
from skimage import feature
from skimage import io
import time

from  fexture import ttext_filelist,fex_read_elm
from  pmatlab import pmat_ingleAImage_python
# 图片对应的点击位置
gloof={ '1':(40,45), '2':(110,45), '3':(180,45), '4':(250,45),'5':(40,120), '6':(110,120), '7':(180,120), '8':(250,120)  }

path_midpro = 'E:\\picdog\\midp\\'
gl_glassy_entry_path = 'F:\\picdog\\glassy\\'
online_learn_image_path='E:\\picdog\\onlearn\\image\\' 
#mjrat,opt_rgb,opt_phash,opt_lpb,opt_hsv  序号


vector_opt = 0
vector_mean = 1
vector_std  = 2


fex_rgb = 'rgb'
fex_lbp = 'lbp'
fex_hsv = 'hsv'
fex_phash='phash'
fex_hog = 'hog'
fex_mjrat='cmajor'
fex_msd  ='msd'

methold_phash =  1
methold_plbpcolor = 2
methold_svm     =  3
methold_meanstd  = 4


# 计算hog 特征
def rgim_fexture_hog(path,Afile):
    
    img = io.imread(os.path.join(path,Afile), as_grey=True)
    orientations = 9
    cell = 8
    block = 3
    
    vector,hogimage= feature.hog(img, orientations=orientations, pixels_per_cell=(cell, cell), cells_per_block=(block, block), transform_sqrt=True,visualise=True)
    
    return vector
    
    
            
## 产生该文件的特征列表   
def rgim_fexture_for_singleimage(path,Afile):
   
    vecdict = {}
   
    AfilePaht = os.path.join(path,Afile);
    try:
       mes = pmat_ingleAImage_python(AfilePaht)
    except:
    
       return  vecdict;
    
    
    pnum = np.uint8(mes[0]);
    if  pnum <> 5 :
        return vecdict;
    
    
    
    # 5    64    64    59    32    12
    #%%总数  len  rgbhst , phach  lbp ,hsv , mjr
    len1=mes[1];
    
    vecdict[fex_rgb]   = mes[pnum+1:pnum+1+np.uint8(mes[1]) ];    
    vecdict[fex_phash]  = np.int8( mes[pnum+1+np.uint8(mes[1]):pnum+1+np.uint8(mes[1]+mes[2]) ] );    
        
    vecdict[fex_lbp]    = mes[pnum+1+np.uint8(mes[1]+mes[2]):pnum+1+np.uint8(mes[1]+mes[2]+mes[3]) ]
    vecdict[fex_hsv]    = mes[pnum+1+np.uint8(mes[1]+mes[2]+mes[3]):pnum+1+np.uint8(mes[1]+mes[2]+mes[3]+mes[4]) ]
    vecdict[fex_mjrat]    = mes[pnum+1+np.uint8(mes[1]+mes[2]+mes[3]+mes[4]):pnum+1+np.uint8(mes[1]+mes[2]+mes[3]+mes[4]+mes[5]) ]
   
    vecdict[fex_hog]   = rgim_fexture_hog(path,Afile)
   
    return  vecdict
    


# 计算某个特征 的最优相似程度    
def  rgim_searchopt_ven_whole(Av, filelist,fetx,touch,touchthr,skipthr,distype):
     #return opt
     # fetx 特征向量列表 ,比如lbp,hsv, rgblist 即image的特征
     # distype = 0   ousdis  距离类型
          
     elmtouch = touch;
     elmfilelist = filelist;
                       
     opt =[];
     
     basicnum = elmtouch.shape[0];
     
     flag = np.zeros(basicnum, 'int8');
                        
     
     compnum = 0;
     
     if distype == 1:
        pacorlist = np.ones(basicnum,'uint8')*64
        
     else :
        pacorlist = np.ones(basicnum,'float32')
     
     for i in range(basicnum ):         
          if flag[i] ==  1:
            continue          
         
          C = fetx[i];
          
          if distype == 1:
              corc = gcom_msdis(Av,C) 
          else :                      
             corc =  gcom_ousdis(Av,C)             
          
          compnum =  compnum +1
                       
          pacorlist[i] = corc
                    
          
          flag[i] = 1;
          
          bm = np.where( elmtouch[i,:] < touchthr );                   
          if len( bm[0]  )  == 0 :
              #break
              continue                      
          # 包括本身节点，无影响，都赋值为1了。
          #if corc > skipthr :
          #   flag[ bm[0] ] = 1
               
     opt_va  = np.min(pacorlist)
     mean_va = np.mean(pacorlist);
     std_va  = np.std(pacorlist);
          
     opt = np.array([opt_va,mean_va,std_va],'float32');
     
     # print 或者log 
     seq = np.where( pacorlist ==  opt_va );      
     #print elmfilelist[seq] ,compnum 
          
     # 最优值， 平均值  方差
     return opt     

# major 输入图像的主要成分，elmmajor 某类图像的统计成分     
def rgim_rgb_major(major,elmmajor) :
     mlen = len(major);
     am = major.reshape((mlen/4,4));
     
     seq = np.zeros(mlen/4)
     bl = elmmajor[:,0:3].tolist()
     i = 0;
     rat = 0;
     for x  in  am[:,0:3]:
          if  x.tolist() in bl:
               seq[i] = 1
               rat = rat + am[i,3]
         
          i = i+1;
     
     #print seq     
     r = np.sum(seq)/(mlen/4) if np.sum(seq)/(mlen/4) > rat else  rat
     return np.array([1-r,0,0])
     
     # 0 表示非常匹配， 1 表示完全不匹配，和其它相关性统一尺度   

    
def  rgim_Afile_in_elm(vecdict,elm,elmfextures):
     
     elmtg = elmfextures
     
     optelm ={}
     
     if len(elmtg) == 0:
         return [];
     
     #return [Argb,Aphash,Albp,Ahsv,Amajor]    Avlist
     # filelist drgb,drgb_touch,dphash,dphash_touch,dlbp,dlbp_touch,dhvs,dhvs_touch,crgbmajor,dper
     #searchopt_ven(Av, filelist,fetx,touch,touchthr,skipthr,distype):
     optelm[fex_rgb]   = rgim_searchopt_ven_whole(vecdict[fex_rgb] ,elmtg[0],elmtg[1],elmtg[2],   0.07,0.1, 0)     
     optelm[fex_phash] = rgim_searchopt_ven_whole(vecdict[fex_phash],elmtg[0],elmtg[3],elmtg[4], 5,   11,  1)     
     optelm[fex_lbp]  = rgim_searchopt_ven_whole(vecdict[fex_lbp]  ,elmtg[0],elmtg[5],elmtg[6],0.03,0.1,   0)     
     optelm[fex_hsv]  = rgim_searchopt_ven_whole(vecdict[fex_hsv],elmtg[0],elmtg[7],elmtg[8],0.03,0.1,   0)     
     
       
     
     optelm[fex_mjrat] = rgim_rgb_major(vecdict[fex_mjrat],elmtg[9])
     
     # 每个特征计算出来的 mean 和std 和被elm 统计的 mean 和std 计算差距，
     msd = elmtg[10]
    
    # phash  lbp ,  rgdb  hsv 
    # 向量分量 mean( pstouch ) ,std(pstouch),min(pstouch),max(pstouch)
     #  optelm[fex_rgb]   : opt_va,mean_va,std_va
      #rgb 
     rgb_diff = [ optelm[fex_rgb][0] -      msd[2][2], optelm[fex_rgb][1] -    msd[2][0],optelm[fex_rgb][2] -    msd[2][1] ] 
     phash_diff = [ optelm[fex_phash][0] -  msd[0][2], optelm[fex_phash][1] -  msd[0][0],optelm[fex_phash][2] -  msd[0][1] ]     
     lpb_diff = [ optelm[fex_lbp][0] -      msd[1][2], optelm[fex_lbp][1] -    msd[1][0],optelm[fex_lbp][2] -    msd[1][1] ]  
     hsv_diff = [ optelm[fex_hsv][0] -    msd[3][2], optelm[fex_hsv][1] -    msd[3][0],optelm[fex_hsv][2] -    msd[3][1] ]  
      
     optelm[fex_msd] = np.array([rgb_diff,rgb_diff,lpb_diff,hsv_diff])
     
     #print optelm
     return optelm
 
 # 精简版的函数，只计算phash 部分。
def  rgim_Afile_Cut_in_elm(vecdict,elm,elmfextures):
     
     elmtg = elmfextures
     
     optelm ={}
     
     if len(elmtg) == 0:
         return [];
     
     #return [Argb,Aphash,Albp,Ahsv,Amajor]    Avlist
     # filelist drgb,drgb_touch,dphash,dphash_touch,dlbp,dlbp_touch,dhvs,dhvs_touch,crgbmajor,dper
     #searchopt_ven(Av, filelist,fetx,touch,touchthr,skipthr,distype):
      
     optelm[fex_phash] = rgim_searchopt_ven_whole(vecdict[fex_phash],elmtg[0],elmtg[3],elmtg[4], 5,   11,  1)        
     
     #print optelm
     return optelm
     

def rgim_spiltimage(pscode,fname):
    #simagelist=[];
    #暂时需要文件传输
    subimage ={};
    for y in range(2):
        for x in range(4):
            im2 = rgim_get_sub_img(pscode, x, y)
            ss = '%d'%((y*4+x +1))
            simname = 'im%s_%s'%(ss,fname);
            im2.save('%s%s'%(path_midpro,simname))
            subimage[ss]=simname;
            #simagelist.append(im2)
            
    return subimage;  

'''
   重要函数，识别passcode图像，判定选择
'''    
def rgim_rg_psascode(r,texts,pscode,fname,gpath):
    #get elm 
    elmfexture ={}
    for elm in texts :
       elmfexture[elm] = fex_read_elm(r,elm)
       if len(elmfexture[elm]) == 0 :
           print elm ,'dont exist ' , elm, 'fexture information'
           return  np.zeros(8,'int');
    
    subimage = rgim_spiltimage(pscode,fname)
    
    
    # get fexture_value,获取所有 子图片的特征信息
    simage_fexture = {};
    for key in subimage:
        simage_fname = subimage[key] 
        simage_fexture[key] =   rgim_fexture_for_singleimage(path_midpro,simage_fname)
    # 
    elmoptall={};
    elmlabelall={}
    for elm in texts :
        opt_elm ={};
        label_elm={}
        for key in subimage :
              opt = rgim_Afile_in_elm(simage_fexture[key], elm,elmfexture[elm]  )
              opt_elm[key] = opt
              #label = rgim_rg_bysvmmodel_elm(gpath,elm,simage_fexture[key])
              #label_elm[key]  = label 
              
        elmoptall[elm] = opt_elm
        #elmlabelall[elm] = label_elm  
    rgim_image_log_elmdata(fname,texts,elmoptall,fex_rgb)
    rgim_image_log_elmdata(fname,texts,elmoptall,fex_phash)
    rgim_image_log_elmdata(fname,texts,elmoptall,fex_lbp)
    rgim_image_log_elmdata(fname,texts,elmoptall,fex_hsv)
    #msd  数据部分格式 mean( pstouch ) ,std(pstouch),min(pstouch),max(pstouch) 
    #elmoptall 分elm作为key，然后8个图像的特征 列表。
    # 每个特征项：最优值， 平均值  方差
    #mjrat,opt_rgb,opt_phash,opt_lpb,opt_hsv
    # 判断匹配程度
    
    phashflag = rgim_methold_phash(fname,subimage,texts,elmoptall)
    print texts, fname,'phash select flag',phashflag
    
    rgim_image_log_format(fname,texts,'mixphash',phashflag)  
    
    tagflag = np.ones(8,'int8')
    tagflag = tagflag - phashflag  
    negaresult = rgim_search_inAllElm(r,fname,simage_fexture,tagflag,texts)
    negaflag = negaresult[0]
    minphash = negaresult[1]
    rgim_image_log_format(fname,texts,'negaflag',negaflag) 
    rgim_image_log_format(fname,texts,'minphash',minphash)  
    
    lcflag = rgim_methold_plbpcolor(fname,subimage,texts,elmoptall,negaflag)
    print texts, fname,'lbp color select flag',lcflag
    rgim_image_log_format(fname,texts,'mixlbpcolor',lcflag)  
    
    msdflag = rgim_methold_msd(fname,subimage,texts,elmoptall,negaflag)
    print texts, fname,'msd  flag',msdflag
    rgim_image_log_format(fname,texts,'mixmsd',msdflag)  
    
    sortflag = rgim_methold_sortseq(fname,subimage,texts,elmoptall)
    print texts, fname,'sort select flag',sortflag
    
    rgim_image_log_format(fname,texts,'mixsort',sortflag)  
    
    
    
    #phashflag  属于确定选上的
     
    tflag =  lcflag + msdflag + sortflag
    
    mixflag = tflag >= 2 
             
    mixflag = mixflag.astype(int)
    
    mixflag =  mixflag + phashflag
    
    mixflag = mixflag > 0
    
    mixflag = mixflag.astype(int)
    
    if np.sum(mixflag) == 0:  # 没有出现重复
        mixflag = tflag  
    
    rgim_image_log_format(fname,texts,'result',mixflag)
    
    if np.sum(mixflag) == 0:
        mixflag = rgim_methold_exclusive(fname,subimage,texts,elmoptall,negaflag)
        rgim_image_log_format(fname,texts,'exclusive',mixflag)  
    
    # remove ,
    #############  后面在增加处理步骤，
    for key in subimage:
        simage_fname = subimage[key] 
        os.remove('%s%s'%(path_midpro,simage_fname))
    
    return  mixflag

def rgim_image_log_format(fname,texts,metholdstr,flag):
    textstr = ''
    if isinstance(texts,list) == True:
        for elm in texts:
            if len( textstr) == 0 :
                textstr = elm
            else :
               textstr = '%s-%s'%(textstr,elm)
    else:
        textstr = texts
        
    str = '[select flag  %s %d:%d:%d:%d:%d:%d:%d:%d ]'%(metholdstr,flag[0],flag[1],flag[2],flag[3],flag[4],flag[5],flag[6],flag[7])
    gcom_log_pscode_rg(fname,textstr,str)

def rgim_image_log_elmdata(fname,texts,elmoptall,fextype):
    for elm in texts :
        datastr = ''
        for i  in range(8) :
            key = '%d'%(i+1)
            opt = elmoptall[elm][key]
            datastr = '%s %f'%(datastr,opt[fextype][vector_opt])
        gcom_log_pscode_rg(fname,elm, fextype + datastr)    
        
def rgim_image_log_dis(fname,text,typestr,smdis):
    textstr = text   
    str = '[level distance  %s %f:%f:%f:%f:%f:%f:%f:%f ]'%(typestr,smdis[0],smdis[1],smdis[2],smdis[3],smdis[4],smdis[5],smdis[6],smdis[7])
    gcom_log_pscode_rg(fname,textstr,str)
        
'''
   图像匹配 方法接口，需要考虑多种可能方式，规则化接口
'''    
def rgim_methold_phash(fname,subimage,texts,elmoptall):

    sflag = np.zeros(8, 'int8');
    allflag = np.zeros(8, 'int8');
    # key 是1 ,2,3,4,5,6,7,8
    for elm in texts :
        sflag = np.zeros(8, 'int8');
        for key in subimage :
            opt = elmoptall[elm][key]
            if  rgim_phash_makesure(opt[fex_phash][vector_opt]) :
                sflag[int(key)-1] = 1    
        rgim_image_log_format(fname,elm,'phash',sflag)  
        allflag = allflag + sflag
        
    mflag = allflag > 0
    
    mflag = mflag.astype(int)
    
    return  mflag;

def rgim_methold_plbpcolor(fname,subimage,texts,elmoptall,negaflag):
    
    texts_smdis = {}
    
    # key 是1 ,2,3,4,5,6,7,8
    #每个特征项：最优值， 平均值  方差
    for elm in texts :
        smdis  = np.zeros(8)
        for key in subimage :
            opt = elmoptall[elm][key]
            # 每幅图片的opt数据
            dv = np.array([ opt[fex_rgb][vector_opt] ,opt[fex_lbp][vector_opt] , opt[fex_hsv][vector_opt] ])
            dis = np.linalg.norm(dv)
            idx =  int(key) -1 
            smdis[idx] = dis

        # 
        texts_smdis[elm] = smdis

    allflag = np.zeros(8, 'int8');   
    
    for elm in texts :
        smdis = texts_smdis[elm]
        sflag =  rgim_dis_select(smdis,negaflag)
        rgim_image_log_dis(fname,elm,'lpbcolor',smdis)
        rgim_image_log_format(fname,elm,'lbpcolor',sflag)  
        
        allflag = allflag + sflag
    for i in range(8):
        if allflag[i] > 1 :
           allflag[i] = 0
  
    return  allflag;

def rgim_methold_msd(fname,subimage,texts,elmoptall,negaflag):
    texts_mds = {}
    for elm in texts :
        smdis  = np.zeros(8)
        for key in subimage :  
            opt = elmoptall[elm][key]
            # 每幅图片的opt数据
            
            dis = np.linalg.norm(opt[fex_msd])
            idx =  int(key) -1 
            smdis[idx] = dis   

        texts_mds[elm] = smdis
    
    allflag = np.zeros(8, 'int8');   
    
    for elm in texts :
        smdis = texts_mds[elm]
        sflag =  rgim_dis_select(smdis,negaflag)
        rgim_image_log_dis(fname,elm,'msd',smdis)
        rgim_image_log_format(fname,elm,'msd',sflag)  
        allflag = allflag + sflag
    for i in range(8):
        if allflag[i] > 1 :
           allflag[i] = 0
  
    return  allflag; 

# 将data指按从小到大的顺序进行排序
def rgim_sort_levelflag(data):
    # 
    uni = np.unique(data)
    levelflag = np.zeros(8,'int')
    for i in range(len(uni) ):
        
       levelflag[ data == uni[i] ] = i
    return  levelflag
    

# 计算顺序的方法，进行选择
def rgim_methold_sortseq(fname,subimage,texts,elmoptall):
    texts_sortseq={}
    for elm in texts :
        rgb_seq  = np.zeros(8)
        lbp_seq  = np.zeros(8)
        phash_seq = np.zeros(8)
        hsv_seq   = np.zeros(8) 
        for key in subimage :  
            opt = elmoptall[elm][key]
            # 每幅图片的opt数据
            
            
            idx =  int(key) -1 
            rgb_seq[idx] = opt[fex_rgb][vector_opt]
            phash_seq[idx] = opt[fex_phash][vector_opt]
            lbp_seq[idx] = opt[fex_lbp][vector_opt]
            hsv_seq[idx] = opt[fex_hsv][vector_opt]
        
        rgb_st =   rgim_sort_levelflag (rgb_seq) 
        phash_st = rgim_sort_levelflag(phash_seq)
        lbp_st =rgim_sort_levelflag(lbp_seq)        
        hsv_st = rgim_sort_levelflag(hsv_seq)
        all_st = rgb_st + phash_st + lbp_st + hsv_st;
        texts_sortseq[elm] = all_st
    
    allflag = np.zeros(8, 'int8');   
    
    tnega = np.zeros(8)
    
    for elm in texts :
        
        sflag = rgim_dis_select(texts_sortseq[elm],tnega)
        
        allflag = allflag + sflag
        
    for i in range(8):
        if allflag[i] > 1 :
           allflag[i] = 0
        
    return  allflag
    
    
def rgim_dis_select(smdis,negaflag):
     
    n = np.sum(negaflag)
    
    exmdis = np.zeros( 8 - n)
    
    Tsmdis = np.zeros(8)
    Tsmdis = smdis
    
    mx = np.max(Tsmdis)
     
    j = 0
    for i in range(8):
        
        if negaflag[i] == 0:
            exmdis[j] = Tsmdis[i]
            j = j+1
        else :
           Tsmdis[i] =  mx  # 避免后面计算中选中
       
    
    # 超过3倍误差
    std = np.std(exmdis)
    mean = np.mean(exmdis)
    
 
    val = (mean - Tsmdis - 2*std > 0)
    if np.sum( val  )  == 0 :
        
        val = (mean - Tsmdis - std > 0)     
        if  np.sum( val  )  == 0 :
            sflag = val.astype(int)   #没有可选的
        else :
            sflag = val.astype(int)       
    else:
        sflag = val.astype(int)
    
   
    return sflag
###  反向选择试试 ，排除法
def  rgim_exclusive_select(data,negaflag):
    
    mean = np.mean(data)
    nflag = data < mean
    nflag = nflag.astype(int)
    nflag = nflag * (1 - negaflag)
    return nflag    

def rgim_methold_exclusive(fname,subimage,texts,elmoptall,negaflag):
    
    allflag = np.zeros(8, 'int8');
    
    for elm in texts :
        rgb_seq  = np.zeros(8)
        lbp_seq  = np.zeros(8)
        phash_seq = np.zeros(8)
        hsv_seq   = np.zeros(8) 
        for key in subimage :  
            opt = elmoptall[elm][key]
            # 每幅图片的opt数据
            
            
            idx =  int(key) -1 
            rgb_seq[idx] = opt[fex_rgb][vector_opt]
            phash_seq[idx] = opt[fex_phash][vector_opt]
            lbp_seq[idx] = opt[fex_lbp][vector_opt]
            hsv_seq[idx] = opt[fex_hsv][vector_opt]
        # 超过 平均值以上的都会 设置成0
        nflag = rgim_exclusive_select(rgb_seq,negaflag) *  rgim_exclusive_select(phash_seq,negaflag) *  rgim_exclusive_select(lbp_seq,negaflag) *  rgim_exclusive_select(hsv_seq,negaflag)
        
       
        allflag = allflag + nflag
        
    for i in range(8):
        if allflag[i] > 1 :
           allflag[i] = 0
        
    return  allflag
        

# 用于全局搜索最匹配，利用phash 
#  elm 需要排除的单元，text
def rgim_search_inAllElm(r,fname,simage_fexture, tagflag,excuelms) :   
    pointst  = time.clock()
    
    glassylist = gcom_glassylist(gl_glassy_entry_path)
    
    negaflag = np.zeros(8, 'int8');
    minphash = 64+np.zeros(8, 'int');
    mphashelm =  ['']*8
    
    for elm in  glassylist:
        bconflag = 0;
        for telm in excuelms:
            if cmp(elm,telm) == 0:
               bconflag = 1
               break
        if bconflag  ==  1:
            continue
                
        elmfexture =   fex_read_elm(r,elm)
        
        for i in range(8):
            if tagflag[i] == 1 and negaflag[i] == 0  :
                key = '%d'%(i+1)
                opt =  rgim_Afile_Cut_in_elm(simage_fexture[key], elm,elmfexture )   # 调用精简版的api 函数只 计算phash
                
                if minphash[i] >    opt[fex_phash][vector_opt] :
                   minphash[i] = opt[fex_phash][vector_opt] 
                   mphashelm[i] = elm
                if rgim_phash_makesure( opt[fex_phash][vector_opt] ) :     #### 重要参数
                    negaflag[i] = 1  #   该图像可以被排除，其它类别中有该图片的更合适的匹配
                    gcom_log_pscode_rg(fname,elm,key+' should be more like *************')
                    break
                
                 
    pointgap =  time.clock() - pointst 
    print 'time gap .....' ,pointgap       
    gcom_log_pscode_rg (fname,'', mphashelm[0]+mphashelm[1] + mphashelm[2] + mphashelm[3] + mphashelm[4] + mphashelm[5] + mphashelm[6] + mphashelm[7] )
    
    return  [negaflag,minphash]
       
    

def rgim_phash_makesure(val):
    if val <= 8:     # !!!!!!!!!!!!!!!!!!重要参数
       return True
    else :
       return  False
       
def rgim_lbp_makesure(val):
    if val < 0.08:
       return True
    else : 
       return  False

# 根据svm model 判断 image
# vectors 
def rgim_rg_bysvmmodel_vector(gpath,elm,fextype,vectors):
    
    modelname='%s_model.mat'%(fextype)
    
    glassypath = os.path.join(gpath,elm)
    
    modelfile = os.path.join(glassypath,modelname)
    
    vlen = len(vectors)
    
    if os.path.isfile(modelfile) == False :
        return  [0]*vlen 
        
    m = svm_load_model(modelfile)
    
    label = [1]*vlen
    print elm,fextype
    reslabel ,res_acc,res_val = svm_predict(label,vectors,m,'-b 1') 
    
    return reslabel

# 计算每个特征的判定标签
def rgim_rg_bysvmmodel_elm(gpath,elm,vecdict):
    label = {}
    for key,val in vecdict.items():
        label[key] = rgim_rg_bysvmmodel_vector(gpath,elm,key,[val.tolist()]) 
        
    return  label
       

def rgim_rg_passcodeonline(r,pscode,texts,fname):
    
    
    sflag = rgim_rg_psascode(r,texts,pscode,fname,gl_glassy_entry_path)
    randcode =''
    
    for a in range(1,9):
       if sflag[a-1] != 1:
          continue 
       b = gloof['%d'%(a)]
       if len(randcode) > 0    :
          randcode = '%s,%d,%d'%(randcode,b[0],b[1]) 
       else:
         randcode =  '%d,%d'%(b[0],b[1]) 
    if len(randcode) == 0:
       print 'select none .....'
    else :
       print 'select .....' ,sflag,'......coord .........',randcode
    ### 10.06 多输出sflag
    return  [randcode,sflag ] 
    
    
#  textstr  ---> '人参|游泳圈'   
def rgim_test_passcode(r,fpath,fname,textstr):
    pscode = Image.open( os.path.join(fpath,fname) )
    texts = textstr.split('|')
    rgim_rg_passcodeonline(r,pscode,texts,fname)

#save sub image to glassy 
def rgim_save_subimage(Apath,fname,texts,sflag):    
    for i  in range(8):
        if sflag[i] == 1:
           gcom_save_subimage_byseq(online_learn_image_path,texts[0],Apath,fname,i+1)  
    
    
#
'''
   
'''    
#   E:\picdog\test  1466234756_4.jpg
'''
python -mpdb E:\picdog\bear\rgimage.py simage  E:\picdog\test  1466234756_4.jpg  七星瓢虫  F:\picdog\glassy

python -mpdb E:\picdog\bear\rgimage.py simage  E:\picdog\midp  1475718693.jpg  袋鼠  
python -mpdb E:\picdog\bear\rgimage.py simage  E:\picdog\midp  1475718621.jpg 箱子



'''
if __name__ == '__main__':
    sys.argv[0]
    if len(sys.argv) == 1:
        print  'useage    /'
    else : 
        r = redis.Redis(host='localhost',port=6379,db=0)
      
       
        if  cmp(sys.argv[1],'simage') == 0 and len(sys.argv) == 5:
            gcom_loginit()
            rgim_test_passcode(r ,sys.argv[2],sys.argv[3],sys.argv[4].decode('gbk'))   
  
       
        else :
           print  'useage  load /test /'
    
    
    
    print 'game over....'    