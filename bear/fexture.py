#_*_ coding:utf-8 _*_  
import redis
import numpy as np
import os
import sys

from  glassycom import  gcom_glassylist,gcom_getfilelist,gcom_bwread


glassy_path = 'F:\\picdog\\glassy\\'
g_ttext_path  ='F:\\picdog\\textdb\\'

# 基础数据组织，使用特征接口，

# base  特征，每个样本都拥有的，并且需要两两相互系数/距离的
basefx_phash='phash'
basefx_lbp  ='lbp'
basefx_rgb  = 'rgb'
basefx_hvs  = 'hsv'

# Statistics  类型的数据
statfx_cmajor='cmajor'
statfx_meanstd='msd'
stat_filelist ='filelist'
# 
# 
df_suffix='.csv'
df_suffix_txt='.txt'
df_touch    ='touch'

########################
#title text fexture
ttext_touch= 'touch'
ttext_filelist='filelist'
 

##########################

#### redis table/hash/set  name
set_glassy='set_glassy_set'
set_ttext_pfix='set_ttext'

# 基础分量特征
basefexture={basefx_phash:'int8',basefx_lbp:'float32',basefx_rgb:'float32',basefx_hvs:'float32'}
statfexture={stat_filelist:'str',statfx_cmajor:'float32',statfx_meanstd:'float32'}
ttextfexture={ttext_touch:'float32',ttext_filelist:'str'}

### glyheader : glassy name /header ,mush be unicode ; str.decode('gbk')
#### 通用的数据入库api
def fexcom_addto_redis(path,glyhder,r,statcase,casetype,touch,tableextend):
    
    asuffix = df_suffix
    if cmp(casetype,'str') == 0 :
        asuffix = df_suffix_txt
    
    if len(touch) == 0 :
        fl = '%s%s%s'%(path,statcase,asuffix)
    else :
        fl = '%s%s_%s%s'%(path,touch,statcase,asuffix)
    
    if os.path.isfile(fl) == False :
        return -1    
    
    frmdata = np.loadtxt(fl,delimiter=',',dtype=casetype)   
    
    if len(frmdata) == 0 :
       return -1
    
    datastr = frmdata.tostring()
    datadtype = frmdata.dtype.str
    datarecord = frmdata.shape[0] 
    
    ###   记录以hash形式存在
    if len(touch) == 0 :
       htablename = 'h_%s_%s%s'%(glyhder,statcase,tableextend)
    else :
       htablename = 'h_%s_%s_%s%s'%(glyhder,statcase,touch,tableextend)
    print htablename
    
    #先删除
    r.delete(htablename)
    
    r.hset(htablename,'data',datastr)
    r.hset(htablename,'dtype',datadtype)
    r.hset(htablename,'record',datarecord)
    
    return 0    
# 从redis 读出内容，  tableextend 供text ttile部分使用
def fexcom_readfrom_redis(r,glyhder,statcase,touch,tableextend)  : 
    
    if len(touch) == 0 :
       htablename = 'h_%s_%s%s'%(glyhder,statcase,tableextend)
    else :
       htablename = 'h_%s_%s_%s%s'%(glyhder,statcase,touch,tableextend)
    
    hdict = r.hgetall(htablename)
    
    if len(hdict) ==0 :
        return []
    
    datastr = hdict['data']     
    datadtype =  hdict['dtype'] 
    datarecord =  int(hdict['record'] )

    casenp = np.fromstring(datastr,datadtype)
    casenp = casenp.reshape(  ( datarecord, casenp.shape[0]/datarecord ) )
    
    return casenp    
    
# 读取文件写入redis
def fexstat_addto_redis(path,glyhder,r,statcase,casetype):
    
   return fexcom_addto_redis(path,glyhder,r,statcase,casetype,'','')


# 读取文件写入redis,base 类型，需要写入两份
def fexbase_addto_redis(path,glyhder,r,statcase,casetype):
    ar =  fexcom_addto_redis(path,glyhder,r,statcase,casetype,'','')
    if ar == -1 :
        return ar
    return fexcom_addto_redis(path,glyhder,r,statcase,casetype,df_touch,'')
   
def fex_update_glassylist(glassylist,r):
    r.delete(set_glassy)
    for elm in  glassylist:
       r.sadd(set_glassy,elm)
    

def fex_redis_load(r,gpath):

    glassylist = gcom_glassylist(gpath)
    fex_update_glassylist(glassylist,r)
    
    for elm in  glassylist:
        for key in statfexture:
           fexstat_addto_redis('%s%s\\'%(gpath,elm),elm,r,key,statfexture[key]) 
        for key in  basefexture:
           fexbase_addto_redis('%s%s\\'%(gpath,elm),elm,r,key,basefexture[key]) 
     
def fex_read_elm(r,elm):

    elmfilelist = fexcom_readfrom_redis(r,elm,stat_filelist,'','')
    
    if len(elmfilelist) == 0:
       return []
    #r,glyhder,statcase,touch,tableextend
    drgb = fexcom_readfrom_redis(r,elm,basefx_rgb,'','')
    drgb_touch = fexcom_readfrom_redis(r,elm,basefx_rgb,df_touch,'')
    
    dphash = fexcom_readfrom_redis(r,elm,basefx_phash,'','')
    dphash_touch = fexcom_readfrom_redis(r,elm,basefx_phash,df_touch,'')
    
    dlbp = fexcom_readfrom_redis(r,elm,basefx_lbp,'','')
    dlbp_touch = fexcom_readfrom_redis(r,elm,basefx_lbp,df_touch,'')
    
    dhvs = fexcom_readfrom_redis(r,elm,basefx_hvs,'','')
    dhvs_touch = fexcom_readfrom_redis(r,elm,basefx_hvs,df_touch,'')
    
    crgbmajor = fexcom_readfrom_redis(r,elm,statfx_cmajor,'','')
    dmsd = fexcom_readfrom_redis(r,elm,statfx_meanstd,'','')
   

    ##########  重要的顺序
    return [elmfilelist[:,0],drgb,drgb_touch,dphash,dphash_touch,dlbp,dlbp_touch,dhvs,dhvs_touch,crgbmajor,dmsd]
     
### load title text
def ttext_addto_redis(path,glyhder,r):
   
    for key in ttextfexture :
        fexcom_addto_redis(path,glyhder,r,key,ttextfexture[key],'','_ttext')
    
def ttext_title_addto_redis(subpath,txfile,r,htablename) :
    afile =  '%s%s'%(subpath,txfile)  
    re,ta = gcom_bwread(afile)   
    if re != 1 :
       return -1
    
    datastr = ta.tostring()
    datadtype = ta.dtype.str
    datarecord = ta.shape[0] 
    
    ###   记录以hash形式存在, 一个key包括多个png 记录
  
    #print htablename
    r.hset(htablename,txfile,datastr)
    r.hset(htablename,'dtype',datadtype)  # 
    r.hset(htablename,'record',datarecord)
       
# 从redis 读出内容 title 内容
def ttext_title_readfrom_redis(r,text,textfile)  : 
    
    htablename = 'h_%s_title'%(text)
    

    datastr = r.hget(htablename,textfile)
    
    if len(datastr) == 0:
        return []
        
    
    datadtype =  r.hget(htablename,'dtype')  
    datarecord =  int( r.hget(htablename,'record') )

    casenp = np.fromstring(datastr,datadtype)
    casenp = casenp.reshape(  ( datarecord, casenp.shape[0]/datarecord ) )
    
    return casenp    

def ttext_update_elm(r,ttpath,text):
    
    textlen= len(text)
    if textlen > 4 :
        return  -1
       
    text_set='%s_%d'%(set_ttext_pfix,textlen)  
    
    if  r.sismember(text_set,text) ==False :
        return  -2
    
    subpath = '%s%s\\'%(ttpath,text)
    ttext_addto_redis(subpath,text,r)
    
    textfilelist = gcom_getfilelist(subpath)
    
    htablename = 'h_%s_title'%(text)
    #先删除
    r.delete(htablename)
    
    print htablename
    
    for txfile in  textfilelist:
        ttext_title_addto_redis(subpath,txfile,r,htablename)
            
    return 0
    
    
def ttext_redis_load(r,ttpath):
    glassylist = gcom_glassylist(ttpath)
    
    for i in range(1,5):
        r.delete('%s_%d'%(set_ttext_pfix,i))
        
    for text in glassylist:
        textlen= len(text)
        if textlen > 4 :
            continue 
        text_set='%s_%d'%(set_ttext_pfix,textlen)  
        r.sadd(text_set,text)
        subpath = '%s%s\\'%(ttpath,text)
        ttext_addto_redis(subpath,text,r)
        
        textfilelist = gcom_getfilelist(subpath)
        
        htablename = 'h_%s_title'%(text)
        #先删除
        r.delete(htablename)
        
        print htablename
        
        for txfile in  textfilelist:
            ttext_title_addto_redis(subpath,txfile,r,htablename)
            
            
    
def ttext_readfrom_redis(r,glyhder,statcase):
    tableextend='_ttext'
    return fexcom_readfrom_redis(r,glyhder,statcase,'',tableextend)
       

### load title text

     
if __name__ == '__main__':
    
    
    if len(sys.argv) == 1:
        print  'useage  load /test /'
    else : 
        r = redis.Redis(host='localhost',port=6379,db=0)
        if  cmp(sys.argv[1],'load') == 0:
            fex_redis_load(r,glassy_path) 
        elif cmp(sys.argv[1],'textload') == 0:
            ttext_redis_load(r,g_ttext_path)        
        elif  cmp(sys.argv[1],'get') == 0 and len(sys.argv) == 4:
            print  fexcom_readfrom_redis(r,sys.argv[2].decode('gbk'),sys.argv[3],'','')    # 输入的是字符串，
        elif  cmp(sys.argv[1],'gettext') == 0 and len(sys.argv) == 4:   
            print  ttext_readfrom_redis(r,sys.argv[2].decode('gbk'),sys.argv[3])    # 输入的是字符串，
        elif  cmp(sys.argv[1],'gettitle') == 0 and len(sys.argv) == 4:     
            print  ttext_title_readfrom_redis(r,sys.argv[2].decode('gbk'),sys.argv[3])    # 输入的是字符串，  
        else :
           print  'useage  load /test /'
    
    
    
    print 'game over....'
    
    