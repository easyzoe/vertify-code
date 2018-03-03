# -*- coding: utf-8 -*-
from skimage import io,filters,img_as_ubyte,img_as_float
from scipy   import ndimage
import numpy as np
import os
from sklearn.cluster import KMeans

from scipy.ndimage import label, generate_binary_structure

import matplotlib.pyplot as plt

from glassycom import gcom_debug
from skimage import feature,filters



#text regnication api

def  pigtext(lpath,filename):
    
   texts = io.imread( os.path.join(lpath,filename), as_grey=True )
   
   texts = img_as_ubyte(texts)
   
   mtext =im2single(texts);
   
   rststr = ''
   
   if len(mtext) == 0:
      return rststr
      
   textnum = len(mtext)
   
   for m in range(textnum):
       
       
       #
       #
       stext = mtext[m]
       wn = stext['wn']
       yy = stext['text']
       
       subf=  '%s%s_%d_%d_py.jpg'%(lpath,filename[:-4],m, wn)
       
       if gcom_debug():
          io.imsave(subf,yy);       
       
       
       subf=  '%s%s_%d_%d_py.jpg'%(lpath,filename[:-4],m, wn)
       
       #testa = imread(subf)                              

       txeg = textedge( yy )
           
       subf=  '%stxeg%s_%d_%d_py.jpg'%(lpath,filename[:-4],m, wn)
       if gcom_debug():
          io.imsave(subf,txeg*255);       
       
       znum= wn
            
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
   

def im2single(textimg):
   
   smimg = img_as_float(textimg)
   
   ny = filters.sobel_v(smimg)
   absny = np.abs(ny)
   
   thval = absny[absny > 0 ].mean() 
   meanflag = absny > thval*1.1
   mfcol = meanflag.sum(axis = 0) 
   tp = mfcol[:-1]+mfcol[1:]
   split = np.where(tp > 40 )[0]
   
   
   sy = ndimage.sobel(smimg, axis=1, mode='constant')      
   grady=np.abs(sy)
   
   th= 0.2;
   
   colpt = grady < th
   colsum = colpt.sum(axis = 0 )
   split  = np.where(colsum[:-1] <=3 ) [0]
             
   print split 
   
   
   if len(split) == 1 :
      eg = feature.canny(smimg)
  
      btxhill =  eg.sum(axis =0 )  # np.sum(btx == 0 ,axis = 0 ) 
      
      gap  = 5
      
      split = []
      
      fnd = np.where(btxhill > 0 )
      
      seq = fnd[0]
      
      for ui in range(len(seq) ):
          if ui == 0:
             split.extend([ui])
             
          elif ui == ( len(seq)-1 ) :
              split.extend([seq[ui]+2])
          
          if seq[ui] - seq[ui-1] >= 2*gap:
               split.extend( [seq[ui-1]+3 ]  )                           
               split.extend( [seq[ui]-3 ] )                           
      
   
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
      rect=[ split[2*sj]+1,3,split[2*sj+1]+1,29];
      
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
      
      
      print rect, wn
   
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
   
  print psigma,pnum
  
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
    print clf.labels_
    
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
   
   
   if  right > c:
       bcf = 1;
       #ex_eg = [ ex_eg,   zeros(r,right -c )  ] ;
       ex_eg = np.hstack( (ex_eg, np.zeros((r,right -c ))  )  )
   
   
   if bcf == 1:
       r,c = ex_eg.shape; 
   
   
   
   if up <=0 :
       bcf = 1;
       #ex_eg = [ ( zeros ( 1-up,c)) ;ex_eg];       
       ex_eg = np.vstack(  ( np.zeros (( 1-up,c)) , ex_eg   )   )   
   
   if down > r:
       bcf = 1;
       #ex_eg = [ ex_eg;  (zeros(down-r,c)) ];       
       ex_eg = np.vstack(  ( ex_eg  , np.zeros( (down-r,c) ) )     )
       
   
   
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
   
#  bwlabel  
# bwareaopen    
           
#pigtext('D:\\abc\\text\\','th_1466235057_138.jpg')   

#wd1 =textimg[3:25,1:56]



#imglog = ndimage.gaussian_laplace(wd1,0.55)






