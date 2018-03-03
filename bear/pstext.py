# -*- coding: utf-8 -*-
from skimage import io,filters,img_as_ubyte,img_as_float,draw,measure,color
from scipy   import ndimage
import numpy as np
import os,time,shutil
from sklearn.cluster import KMeans

from scipy.ndimage import label, generate_binary_structure

import matplotlib.pyplot as plt
b
from glassycom import gcom_debug,gcom_bwcorr
from skimage import feature,filters,morphology
 NameErrorB9keras.jsonnmMMMMMMMMMMMMMMMMMMMUIfrom sklearn import  cluster
import cv2
from sklearn.neighbors import NearestNeighbors

#text regnication api
'''
    old : pigtext   text 识别， 采用文件方式传递数据
          新增该方法的 数据传递方式
          new_split_word   text 识别方法二
          
          im2single   
'''

def  pigtext(lpath,filename):
    
   texts = io.imread( os.path.join(lpath,filename), as_grey=True )
   
   texts = img_as_ubyte(texts)
   
   mtext =im2single(texts);
   
   rststr = ''
   
   if len(mtext) == 0:
      return rststr
      
   textnum = len(mtext)
   
   for m in range(textnum):
     
       stext = mtext[m]
       wn = stext['wn']
       yy = stext['text']
       
       subf=  '%s%s_%d_%d_py.jpg'%(lpath,filename[:-4],m, wn)
       
       #if gcom_debug():
       io.imsave(subf,yy);       
       
       
       subf=  '%s%s_%d_%d_py.jpg'%(lpath,filename[:-4],m, wn)
       
       #testa = imread(subf)                              

       txeg = textedge( yy )
          
          
       subf=  '%stxeg%s_%d_%d_py.jpg'%(lpath,filename[:-4],m, wn)
       if gcom_debug():
          io.imsave(subf,txeg*255);    
 
       znum= wn
       if znum  <= 2 :              
           if znum > 1:
               bwwords  = bwword(txeg,znum);
           else: 
              bwwords = [[]];
              bwwords[0] = txeg;  
           
      
            
           jtxeg = jamming(txeg,bwwords);
           
           if not jtxeg is None:
           
               subf=  '%sjamm%s_%d_%d_py.jpg'%(lpath,filename[:-4],m, wn)
               if gcom_debug():
                 io.imsave(subf,jtxeg*255); 
           else :
              jtxeg = txeg           
       else :
           jtxeg = txeg       
              
               
       if znum > 1:
          bwwords  = bwword(jtxeg,znum);
          
          newnum  = len(bwwords)
          if newnum != znum :
               # 失败
               break;
               
          # 检查word是否合法
       else :
          bwwords = [[]];
          bwwords[0] = jtxeg;  
           
       # 格式化           
       nmtexteg = normalizeword(bwwords);
       
       if len(nmtexteg) == 0:       
           break                           

       subf= '%s%s_%d_%d.png'%(lpath,filename[:-4],m,wn);
       io.imsave(subf,nmtexteg*255);       

       if len(rststr) == 0:
           rststr =  '%s_%d_%d.png'%(filename[:-4],m,wn);      
       else :
          rststr = rststr+',' + '%s_%d_%d.png'%(filename[:-4],m,wn);
   return  rststr
  
# imput : texts img
def test_pigtext_data(lpath,filename):
    texts = io.imread( os.path.join(lpath,filename), as_grey=True )
    
    return  pigtext_data(texts)
   
   
def  pigtext_data(texts):
    
   if len(texts.shape) == 3:
        texts =  color.rgb2gray(texts)
        
   texts = img_as_ubyte(texts)
   
   mtext =im2single(texts);
       
   textnum = len(mtext)
   
   out_normal_texts = []
   
   for m in range(textnum):
     
       stext = mtext[m]
       wn = stext['wn']
       yy = stext['text']
       
       nmtexteg = new_split_word(yy,wn)     
       out_normal_texts.extend( [ [ wn,nmtexteg ] ] )
       
   return  out_normal_texts
   
 
def getMaxRect(bweg):
    row,col = np.where(bweg)
    rect = [row.min(),row.max(),col.min(),col.max()]
    return rect
    
def is_maybe_inword(ilabel,labels,word,whole):
    cen_col,cen_row =  compcenter(word)
    wrect =  getMaxRect(word)
    
    mbword = word + (labels == ilabel)
    mbrect = getMaxRect(mbword)
    icen_col,icen_row = compcenter( labels == ilabel )
    dis  = np.linalg.norm( np.array([icen_col,icen_row])  - np.array( [ cen_col,cen_row ] )   )
    
    print dis 
    
    sdsize =  20
    
    if whole < 2 :
       if mbrect[1]-mbrect[0] >  sdsize or mbrect[3]- mbrect[2] > sdsize or dis  > 20  :
          return (0,-1)
    
         
    
    if ( wrect[3] - wrect[2] ) < 12 and (mbrect[3]- mbrect[2] ) < 16  and  dis < 12 :
       return  (1,dis)
    # 
    if whole == 2 and dis > 14:
       return (0,-1)
    if whole < 2 and   dis > 9 :
       return (0,-1)  
    return (1,dis)
    
 
def sort_split_word(texteg,wn):
    total_area = texteg.sum()*1.0
    
    labels = measure.label(texteg,neighbors=8)
    rgps = measure.regionprops(labels)
    
    sort_lab = np.zeros((len(rgps),2))
    for i  in range(len(rgps)):
        lab = rgps[i]
        rt = lab.area /total_area
        sort_lab[i] = [rt,lab.label]
    
    label_sort_area = np.argsort(-sort_lab[:,0]) 
    
    words = []
    
    bwhole = 0 

    for i in range(len(label_sort_area)):
        ilabel = int( sort_lab[label_sort_area[i]][1] )
        hword  = ( labels == ilabel ) 
        if i == 0 :
            word = labels == ilabel
            rect = getMaxRect(word)
            words.extend([word])
            
            if rect[3]- rect[2] >  18 :
                # whole dont'break;
                bwhole = 1
                print 'whole body'
                #break
                if rect[3]- rect[2]  > 30  and   sort_lab[label_sort_area[i]][0] > 0.85:
                   # complete whole body
                   bwhole = 2 
                                       
            continue
        match = []
        for k in range(len(words)):
            tpwd = words[k]
            binword,dis = is_maybe_inword(ilabel,labels,tpwd,bwhole) 
            if binword ==0  and len(words) < wn and  bwhole  < 2 : 
         
                word = labels == ilabel
                words.extend([word])
                break
            elif binword == 1 and dis == 0  :
               words[k] = tpwd + ( labels == ilabel ) 
               break
            elif binword == 1  and dis > 0 :
               match.extend([[dis,k ]]) 
        
        match = np.array(match)
        
        if len(match) == 0:
            continue
        if len(match) == 1:
            mk = int( match[0][1] )
            words[mk] = words[mk] + ( labels == ilabel ) 
        else :
             mk =  np.argsort(match[:,0])[0]
             
             words[mk] = words[mk] + ( labels == ilabel ) 
        
    
    if bwhole > 0  :
        if len(words) ==1 :
           wholeword = words[0]
        else :
           wholeword = words[0] + words[1]
        ret = np.where(wholeword)
        pt = np.array(  zip(ret[1],ret[0] ) )
        ap =  cluster.KMeans(n_clusters=wn).fit(pt)
        words = [[]]* wn
        for k in range(wn): 
            labelpt = pt[np.where(ap.labels_ == k)[0]]
            tmp = np.zeros(wholeword.shape)
            tmp[labelpt[:,1],labelpt[:,0]] = 1
            words[k] = tmp > 0
           
    return words 

    
def split_words_bykeam(texteg,wn):
    ret = np.where(texteg)
    pt = np.array(  zip(ret[1],ret[0] ) )
    ap =  cluster.KMeans(n_clusters=wn).fit(pt)
    words = [[]]* wn
    for k in range(wn): 
        labelpt = pt[np.where(ap.labels_ == k)[0]]
        tmp = np.zeros(texteg.shape)
        tmp[labelpt[:,1],labelpt[:,0]] = 1
        words[k] = tmp > 0
       
    return words 
    
def test_new_split_word(afile,wn):
    yy  = io.imread(afile)
    filename = os.path.basename(afile)
    
    return new_split_word(yy,wn,filename = filename)
    
def new_split_word(yy,wn,filename=''):
    
    txeg = textedge( yy )
    
    if len(filename ) >0 :
        plt.gray()
        subf=  'E:\\picdog\\ctextout\\%s_y_%d.png'%(filename[:-4],wn)
           
        #if gcom_debug():
        io.imsave(subf,txeg*255);
       
    if wn >= 3:
      words =  split_words_bykeam(txeg,wn)
    else :
      #words  = move_win_forword(txeg,wn)
       words  = sort_split_word(txeg,wn)
    
    #words  = bwword(txeg,wn);  
    
    #jtxeg = jamming(txeg,words);
    #if not jtxeg is None:
    #   words = bwword(jtxeg,wn);      
    
    n = 0
    for word in words:
       if len(filename ) >0 :
           subf=  'E:\\picdog\\ctextout\\%s_%d_%d.png'%(filename[:-4], n,wn)
           word =  word > 0
           #if gcom_debug():
           io.imsave(subf,word*255);
       
           n = n+1
       
    nmwords= normalizeword(words)
    
    if len(filename ) >0 :
        subf =  'E:\\picdog\\ctextout\\%s_%d_nm.png'%(filename[:-4],wn)
         
        io.imsave(subf,nmwords*255);
       
    return nmwords
    
       
       
def im2single(textimg):

   smimg = img_as_float(textimg)
   
   ny = filters.sobel_v(smimg)
   absny = np.abs(ny)
   
   thval = absny[absny > 0 ].mean() 
   meanflag = absny > thval*1.1
   mfcol = meanflag.sum(axis = 0) 
   tp = mfcol[:-1]+mfcol[1:]
   
   sortindex =  np.argsort(-tp)
   sortval =  tp[sortindex]
   if sortval[0] > 40:
       split = sortindex[0:2]
       split.sort()
   else :
      split = []
   #print split 
   
   if len(split) == 0 :
      eg = feature.canny(smimg)
  
      btxhill =  eg.sum(axis =0 )  # np.sum(btx == 0 ,axis = 0 ) 
      
      gap  = 5
      
      nozpos = (btxhill == 0)
      mask = np.ones(gap)
      sppos = morphology.binary_erosion(nozpos,mask)
      pos = np.where(sppos > 0 )[0]
      if 0 not in pos:
          pos = np.append(0,pos)
      split = pos[ np.where(pos[1:]-pos[:-1]  > 1 )[0] + 1 ]
      
   if split[0] > 16:
       if 0 not in split:
          split = np.append(0,split)       
   print split
   
   exsplit = split
   
   wordnum = len(split) -1 
   
   split = [0]*2*wordnum
   
   for tk in range(wordnum):
       if tk ==0 :
          split[2*tk] =  exsplit[tk]
       else :
          split[2*tk] = exsplit[tk] +1
       split[2*tk+1]  = exsplit[tk+1]
       
   sub=int(np.floor( len(split)/2))
   
   mt =[];
   
   # 11-12 会将后面的空余部分也识别成一个word，需要和matlab对比，先暂且按数目控制，word个数暂时没有超过3个。
   pwdn = 0
   for sj in range(sub):
      if split[2*sj+1]-split[2*sj] < 10:
         continue
      
      word ={}
      pwdn = pwdn + 1
      
      if pwdn > 2:
          continue
      # 这个地方固定有序号      
      rect=[ split[2*sj],3,split[2*sj+1],29];
      
      wlen= split[2*sj+1]-split[2*sj]-1;
      if wlen < 26:
          wn=1;
      elif wlen < 46:
          wn=2;
      elif wlen < 64 :  #%%估计  
          wn=3;
      elif wlen < 80:
          wn=4;           
      elif wlen < 100:
          wn=5;  
      else :
          wn = 6 ; #%% 错误情况 
      
      if wn >= 5:
         return  []    
      
      word['wn'] = wn
      word['text'] = textimg[rect[1]:rect[3],rect[0]:rect[2] ]
      
      mt.append(word)
      
      
      #print rect, wn
   
   return mt;    

   # 返回子text 列表   
            
      
    
def text_marrhildreth(word,sigma):
      
   yy = word*1.2-70;
   
   #imglog = ndimage.gaussian_laplace(yy,sigma=0.48,mode='mirror')        
   
   m=5;
   n=5;
   
   w= np.zeros([m,n]);
   h_m=(m-1)/2;
   h_n=(n-1)/2;
   
   for i in range(m):
     for j in range(n):
         y = i +1  - h_m
         x = j +1  - h_n
         w[i,j]=(1/(sigma*sigma))*((y*y+x*x)/(sigma*sigma)-2)*np.exp(-(y*y+x*x)/(2*sigma*sigma)); 
   

   w=w/np.sum(w);    


   imglog = ndimage.correlate(yy,w,mode='nearest')
   
   m = imglog.shape[0]
   n = imglog.shape[1]
   
   tmp = np.zeros(4)
   txeg = np.zeros(imglog.shape)
   
   for i in range(1,m-1):
      for j in range(1,n-1):
        
        tmp[0]=np.sum(np.sum(imglog[i-1:i+1,j-1:j+1]));
        tmp[1]=np.sum(np.sum(imglog[i-1:i+1,j:j+1+1]));
        tmp[2]=np.sum(np.sum(imglog[i:i+1+1,j-1:j+1]));
        tmp[3]=np.sum(np.sum(imglog[i:i+1+1,j:j+1+1]));  
        
        Ma = np.max(tmp)
        Mi = np.min(tmp)
        if  Ma > 0.1 and Mi < -0.1:
            imglog[i,j] = 255
         
   txeg = (imglog==255)
   
   txeg = txeg.astype(int)
   
   
   print sigma, np.sum(txeg),'.....'
   
   return txeg
  

def  textedge(text)  :
  pnum = 0;
  texteg =[];
  psigma = 0;
  for sigma in range(55,64,2):
       sigma = sigma/100.0
       tg = text_marrhildreth(text,sigma)
       tmp = np.sum(tg)
       if pnum < tmp:
           texteg = tg
           pnum = tmp
           psigma = sigma
   
  #print psigma,pnum
  
  return  texteg   
  

def bwword(txeg,wn)   :

    # 一个字符，则无需处理； 暂未有4个字以上的
    if wn == 1 or wn > 4 :    
       return
    
    [yyr,yyc] = txeg.shape
    
    initpos = [];
    
    possd = yyc /(wn*2.0)
    
    for i in range(wn):
        initpos.extend([8,possd*(2*(i+1)-1)])
        
    ipos = np.array(initpos).reshape(wn,2)
    
    [r,c] = np.where(txeg==1)
    
    q = np.column_stack((r, c))
    
    if len(q) == 0:
       return
       
    clf = KMeans(n_clusters=wn,init=ipos,n_init=1)
    #clf = KMeans(n_clusters=wn)
    s = clf.fit(q)


    #中心
    print clf.cluster_centers_

    #每个样本所属的簇
    #print clf.labels_
    
    bwwords = []
    
    for region in range(wn):
       
       ptext = np.zeros(txeg.shape)
       
       for k in range( q.shape[0] ): 
            if clf.labels_[k]  ==  region :
               ptext[r[k],c[k] ] = 1
       '''
       plt.gray()
       plt.imshow(ptext)
       plt.show()
       '''
       bwwords.append(ptext)
       
       #compent(ptext)
       
   #  考虑数据传递
    return  bwwords

# #  computer centter of word
def  compcenter(bwword):
     
    text = bwword
    sum_x=0.0;
    sum_y=0.0;
    area=0.0; 
    
    [height,width]= text.shape
    
    for i in range(height):
       for j in range(width):
          if text[i,j] == 1:
               sum_x = sum_x + i
               sum_y = sum_y + j
               area = area + 1 
               
    return np.array([(sum_y/area),(sum_x/area)])
    # y plot in head
    
# jamming word
def  jamming(texteg,bwwords) :
     
     ise = 0
     colth = 40  #小于20×2的col ，认为不可能存在干扰
     pos = 0  #1 =  head  2 mid 3: tail
     
     jammdiscrete = 9  # matlab 10
     wordouter  = 8
     
     wc = len(bwwords)
     
     jtxeg = texteg
     
     [r,c] = texteg.shape
     
     if c < colth  and  wc == 2:
        return 
     elif  c < colth/2 and  wc  == 1:
        return
        
     cenpoint = []
     
     for j in range(wc):
         [ploty,plotx] = compcenter(bwwords[j])
         cenpoint.append( [ploty,plotx] )
         
     
     mbjamm = []
     
     [pr,pc] = np.where(texteg==1)
     
     n = pr.shape[0]
     
     for  xp in range(n):
         
          disset = []
          
          for j in range(wc):
             dis = np.linalg.norm( np.array(  [ pc[xp],pr[xp] ]  )   - np.array(   [ cenpoint[j][0],cenpoint[j][1]  ]  )     )
             
             disset.extend([dis])
          
          if np.min(disset)   > jammdiscrete :
             mbjamm.append([pc[xp],pr[xp] ])
             
    
     s = generate_binary_structure(2,2)
    
     labeled_array, num_features = label(texteg, structure=s)  
     
     # 为 动作准备 jtxeg = bwareaopen(L,2);       
     sdyreg =[]
     for rgi in range(1,num_features+1):
         if np.sum( labeled_array  == rgi  ) < 2 :
             sdyreg.append(rgi)             
     
     
     
     mr=  len(mbjamm)
     
     reg = []
     
     for xp in range(mr):
         
          trg  = labeled_array[ mbjamm[xp][1],mbjamm[xp][0]  ]
          
          if len(reg) == 0:
              reg.append(trg)
              continue
          
          bff = 0
          
          if trg in reg:
              bff = 1
          
          if bff == 0:
              reg.append(trg)
          
     # get all need check region  
     
     
     rr = len(reg)
     
     jamreg = []
     
     for ri in range(rr):
         [pr,pc] = np.where(labeled_array == reg[ri] ) 
         
         n = len(pr)
         
         if  n > 100 :  #%%如果某个region 点大于100个，明显不正常，忽略该region 
            continue
        
         bvalid = 0
         
         for j in range(wc):
         
            tpweg =  bwwords[j] - ( labeled_array == reg[ri] )
            
            tpweg = np.abs(tpweg)
            
            if np.sum(tpweg)  == 0:
                bvalid  = 1
                break
            [ploty,plotx] = compcenter(tpweg)
            
            
            tdis = []
            
            cpt =np.array([ploty,plotx])
            
            for xp in range(n):
                apnt = np.array( [ pc[xp], pr[xp]] )
                
                dis = np.linalg.norm(apnt - cpt)
                
                tdis.append(dis)
                
            npdis = np.array(tdis)     
            mntmp = np.min(npdis)     
            if mntmp < wordouter:
                if mntmp > wordouter - 1 and np.sum( npdis < wordouter )  <= 2:
                    [ty,tx] = compcenter(( labeled_array == reg[ri] ))
                    
                    ccdis =  np.linalg.norm(   np.array([ty,tx]) - cpt   )
                    
                    if ccdis < jammdiscrete :  #% 虽然有距离比较小的点，但整体质心偏离很多，也认为是jamm,
                        bvalid = 1 
                       
                else :
                     bvalid = 1
                                  
            if bvalid ==  1:
                 break

         if bvalid == 0:
             jamreg.append(reg[ri])         
             
     rr = len(jamreg)         
     
     for ri in range(rr):
         labeled_array[labeled_array==jamreg[ri]] = 0
         
     
     

     for xp in range(mr)  :
     
        if (  labeled_array[  mbjamm[xp][1],mbjamm[xp][0]   ]  > 0     ):
            #pass
          labeled_array[  mbjamm[xp][1],mbjamm[xp][0]   ]  = 0
     
     #孤立的点去掉
     for xp in range(len(sdyreg) ) :
         if sdyreg[xp] in  jamreg:
            pass
         else :
           print 'delete lone point'
           labeled_array[labeled_array ==  sdyreg[xp]  ]  = 0
     
     labeled_array[labeled_array > 0 ]  = 1     
     
     jtxeg = labeled_array
     
     return jtxeg     

def getformatword(txeg,wordsize):

   ex_eg = txeg
   [r,c] = ex_eg.shape
   [ploty,plotx]= compcenter(ex_eg);
   
   col_mid = ploty;
   row_mid = plotx;
   
   left = int( np.round(col_mid)  - wordsize );
   up =   int( np.round(row_mid) - wordsize);
   right = int( np.round(col_mid) + wordsize);
   down = int( np.round(row_mid) + wordsize);
   
   bcf = 0;
   
   if left <= 0:
       bcf = 1
       #ex_eg = [  (  np.zeros ( r,1-left ) ), ex_eg];      
       ex_eg = np.hstack( (np.zeros (( r,1-left) ),ex_eg  )  )               
   
   
   if  right >= c:
       bcf = 1;
       #ex_eg = [ ex_eg,   zeros(r,right -c )  ] ;
       ex_eg = np.hstack( (ex_eg, np.zeros((r,right -c +1 ))  )  )
   
   
   if bcf == 1:
       r,c = ex_eg.shape; 
   
   
   
   if up <=0 :
       bcf = 1;
       #ex_eg = [ ( zeros ( 1-up,c)) ;ex_eg];       
       ex_eg = np.vstack(  ( np.zeros (( 1-up,c)) , ex_eg   )   )   
   
   if down >= r:
       bcf = 1;
       #ex_eg = [ ex_eg;  (zeros(down-r,c)) ];       
       ex_eg = np.vstack(  ( ex_eg  , np.zeros( (down-r+1,c) ) )     )
       
   
   
   if bcf ==1 :
       [ploty,plotx]= compcenter(ex_eg);
   
       col_mid = ploty;
       row_mid = plotx;

       left = int( np.round(col_mid)  - wordsize ); 
       up =   int(np.round(row_mid) - wordsize );
       right = int( np.round(col_mid) + wordsize );
       down = int(np.round(row_mid) + wordsize );
                
   
   # matlab  全区间
   fmword = ex_eg[up:down+1,left:right+1];
   
   fmword= fmword.astype(int);
   
   return fmword
   


def  normalizeword(bwwords):
   #%%%  默认字体大小   
   wordsize=8;  

   wc = len(bwwords)
   
   nmtexteg = [];
   
   if wc ==0 :
      return  nmtexteg
   
    ##    一致性化顺序
   cseq=[];
    
   for j in range(wc):
    
       [r,c] = np.where(bwwords[j] == 1 )
       
       cfirst = np.min(c)
       
       cseq.append([cfirst,j])
   
   a = 0   
   abc = sorted(cseq,key=lambda cseq :cseq[0])
    
   normtext = []
    
   for j in range(wc):
       wj = abc[j][1]
       
       weg = bwwords[wj]
       if len(normtext) == 0:
          normtext = getformatword(weg,wordsize)
       else :
          normtext = np.hstack( ( normtext,getformatword(weg,wordsize) ) )
       
    

   #split = np.zeros((2*wordsize+1,1),'int')        
    
   return  normtext
         

def compent(wordeg) :
 
   s = generate_binary_structure(2,2)
   labeled_array, num_features = label(wordeg, structure=s)
   
def text_knn_cluster(filelist,dstpath,wn=3):
    
    textdata = []
    for afile in filelist:
        text =  io.imread(afile)
        text = text.flatten().astype(float)
        if text.shape[0]  ==  (17*17*wn):
           textdata.extend([text])
        else :
           print afile        
    textdata = np.array(textdata)
    num = textdata.shape[0]
    ap =  NearestNeighbors(n_neighbors=50, radius=1.0, metric=text_knn_dis).fit(textdata)
    text_distance,text_index =    ap.kneighbors( textdata  )
    a = 0 
    
    scanflag = np.zeros(num)
    
    labels = {}
    labelit = 1
    
    for i in range(num):
       if scanflag[i] > 0 :
            continue
       
       # 分割时，门限为0.15  ； 没有分割采用0.05
       
       #if numbin > 1000:
       #   numbin = 1000
       select_set =  com_any_cluster( text_index,text_distance,num,i,thre = 0.55 )
     
       
       if len( select_set) == 0 or  (   len(select_set) == 1 and  select_set[0] == i   ) :
            continue
       
       
       standby = scanflag[select_set]
       if standby.sum() > 0:
           bmay =  np.unique(standby[standby > 0] )
           
           
           moveset = select_set[ scanflag[select_set] ==0 ]
           
           scanflag[select_set] = bmay[0]
           dstsubpath = labels[bmay[0]]
       

       else :
          seq_it = int(time.time())
          
          dstsubpath = os.path.join(dstpath,"%d-%d"%(seq_it,i) )    
          scanflag[select_set] = labelit
          labels[labelit] = dstsubpath
          labelit = labelit +1   

          moveset = select_set 
          
       text_movefile_bysubset(moveset,filelist,dstsubpath,0 )
    
    print scanflag[scanflag ==0].shape[0] 
    
     
def text_knn_dis(atext,btext):
    npatext =  atext.reshape(17,atext.shape[0]/17)
    npbtext =  btext.reshape(17,btext.shape[0]/17)
    
    corr = gcom_bwcorr(npatext,npbtext)
    return 1- corr
         
def com_any_cluster(index,distance,num,s =0 ,thre =  0.028 ):

    total_seq = None

    snb = index[s][1:][  distance[s][1:] < thre  ]
    total_seq = snb.copy()
    
    for i in range(len(snb)):
        t = snb[i]
        tp_seq = []
        tp_seq  = index[t][1:][ distance[t][1:] < thre ]
        total_seq = np.append(total_seq,tp_seq)
    total_seq = np.append(s,total_seq)
    res = np.unique(total_seq)

    return res
    
def text_movefile_bysubset(subset,filelist,dstsubpath,file_moveflag):
    
  
        
    subfilelist= []    
    
    for i in subset:
        afile = filelist[i]
        if not os.path.exists(dstsubpath) :
          os.mkdir(dstsubpath)
          
        if file_moveflag == 0 :
           shutil.copy(afile  , dstsubpath)
        else :
           shutil.move(afile , dstsubpath)
    return   

 #  bwlabel  
# bwareaopen    
           
#pigtext('D:\\abc\\text\\','th_1466235057_138.jpg')   

#wd1 =textimg[3:25,1:56]



#imglog = ndimage.gaussian_laplace(wd1,0.55)






