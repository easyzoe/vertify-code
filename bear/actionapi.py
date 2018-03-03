#_*_ coding:utf-8 _*_  
# action api  for web operation


import requests
import urllib
import redis
import time
import matplotlib.pyplot as plt
from PIL import Image
import StringIO
import os,sys
import re
import datetime

import cookielib

from rgtitle import  rgtt_rgtitle_passcode
from rgimage import  rgim_rg_passcodeonline,path_midpro,rgim_save_subimage
from glassycom  import  gcom_online_loginit,gcom_log_online


requests.packages.urllib3.disable_warnings()

RET_OK = 0
RET_ERR = -1
MAX_TRIES = 3
Randcodegap = 20


seatMaps = [
    ('1', u'硬座'),  # 硬座/无座
    ('3', u'硬卧'),
    ('4', u'软卧'),
    ('7', u'一等软座'),
    ('8', u'二等软座'),
    ('9', u'商务座'),
    ('M', u'一等座'),
    ('O', u'二等座'),
    ('B', u'混编硬座'),
    ('P', u'特等座')
]
# 二等座 大写字母O


def hasKeys(dict,keys):
    for key in keys :
       if dict.has_key(key) ==False :
            return False
    return  True

#hasKeys(obj, ['status', 'httpstatus', 'data'])

def chooo_initSession():
    session = requests.Session()
    session.headers = {
            'Accept': 'application/x-ms-application, image/jpeg, application/xaml+xml, image/gif, image/pjpeg, application/x-ms-xbap, */*',
            'Accept-Encoding': 'gzip, deflate',
            'Accept-Language': 'zh-CN',
            'User-Agent': 'Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 6.1; WOW64; Trident/4.0; SLCC2; .NET CLR 2.0.50727; .NET CLR 3.5.30729; .NET CLR 3.0.30729; Media Center PC 6.0; .NET4.0C)',
            'Referer': 'https://kyfw.12306.cn/otn/index/init',
            'Host': 'kyfw.12306.cn',
            'Connection': 'Keep-Alive'
    }
        
    return  session
        
        
def chooo_updateHeaders(session, url):
    d = {
            'https://kyfw.12306.cn/otn/resources/js/framework/station_name.js': {
                'method': 'GET',
                'Referer': 'https://kyfw.12306.cn/otn/'
            },
            'https://kyfw.12306.cn/otn/login/init': {
                'method': 'GET',
                'Referer': 'https://kyfw.12306.cn/otn/'
            },
            'https://kyfw.12306.cn/otn/passcodeNew/getPassCodeNew?module=login&rand=sjrand&': {
                'method': 'GET',
                'Referer': 'https://kyfw.12306.cn/otn/login/init'
            },
            'https://kyfw.12306.cn/otn/passcodeNew/getPassCodeNew?module=passenger&rand=randp&': {
                'method': 'GET',
                'Referer': 'https://kyfw.12306.cn/otn/confirmPassenger/initDc'
            },
            'https://kyfw.12306.cn/otn/passcodeNew/checkRandCodeAnsyn': {
                'method': 'POST',
                'Referer': 'https://kyfw.12306.cn/otn/login/init',
                'Cache-Control': 'no-cache',
                'x-requested-with': 'XMLHttpRequest',
                'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8'
            },
            'https://kyfw.12306.cn/otn/login/loginAysnSuggest': {
                'method': 'POST',
                'Referer': 'https://kyfw.12306.cn/otn/login/init',
                'Cache-Control': 'no-cache',
                'x-requested-with': 'XMLHttpRequest',
                'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8'
            },
            'https://kyfw.12306.cn/otn/login/userLogin': {
                'method': 'POST',
                'Referer': 'https://kyfw.12306.cn/otn/login/init'
            },
            'https://kyfw.12306.cn/otn/index/init': {
                'method': 'GET',
                'Referer': 'https://kyfw.12306.cn/otn/login/init'
            },
            'https://kyfw.12306.cn/otn/leftTicket/init': {
                'method': 'POST',
                'Referer': 'https://kyfw.12306.cn/otn/index/init',
                'Content-Type': 'application/x-www-form-urlencoded'
            },
            'https://kyfw.12306.cn/otn/leftTicket/log?': {
                'method': 'GET',
                'Referer': 'https://kyfw.12306.cn/otn/leftTicket/init',
                'x-requested-with': 'XMLHttpRequest',
                'Cache-Control': 'no-cache',
                'If-Modified-Since': '0'
            },
            'https://kyfw.12306.cn/otn/leftTicket/query?': {
                'method': 'GET',
                'Referer': 'https://kyfw.12306.cn/otn/leftTicket/init',
                'x-requested-with': 'XMLHttpRequest',
                'Cache-Control': 'no-cache',
                'If-Modified-Since': '0'
            },
            'https://kyfw.12306.cn/otn/leftTicket/queryT?': {
                'method': 'GET',
                'Referer': 'https://kyfw.12306.cn/otn/leftTicket/init',
                'x-requested-with': 'XMLHttpRequest',
                'Cache-Control': 'no-cache',
                'If-Modified-Since': '0'
            },
            'https://kyfw.12306.cn/otn/login/checkUser': {
                'method': 'POST',
                'Referer': 'https://kyfw.12306.cn/otn/leftTicket/init',
                'Cache-Control': 'no-cache',
                'If-Modified-Since': '0',
                'x-requested-with': 'XMLHttpRequest',
                'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8'
            },
            'https://kyfw.12306.cn/otn/leftTicket/submitOrderRequest': {
                'method': 'POST',
                'Referer': 'https://kyfw.12306.cn/otn/',
                'Cache-Control': 'no-cache',
                'x-requested-with': 'XMLHttpRequest',
                'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8'
            },
            'https://kyfw.12306.cn/otn/confirmPassenger/initDc': {
                'method': 'POST',
                'Referer': 'https://kyfw.12306.cn/otn/leftTicket/init',
                'Content-Type': 'application/x-www-form-urlencoded',
                'Cache-Control': 'no-cache'
            },
            'https://kyfw.12306.cn/otn/confirmPassenger/getPassengerDTOs': {
                'method': 'POST',
                'Referer': 'https://kyfw.12306.cn/otn/confirmPassenger/initDc',
                'Cache-Control': 'no-cache',
                'x-requested-with': 'XMLHttpRequest',
                'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8'
            },
            'https://kyfw.12306.cn/otn/confirmPassenger/checkOrderInfo': {
                'method': 'POST',
                'Referer': 'https://kyfw.12306.cn/otn/confirmPassenger/initDc',
                'Cache-Control': 'no-cache',
                'x-requested-with': 'XMLHttpRequest',
                'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8'
            },
            'https://kyfw.12306.cn/otn/confirmPassenger/getQueueCount': {
                'method': 'POST',
                'Referer': 'https://kyfw.12306.cn/otn/confirmPassenger/initDc',
                'Cache-Control': 'no-cache',
                'x-requested-with': 'XMLHttpRequest',
                'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8'
            },
            'https://kyfw.12306.cn/otn/confirmPassenger/confirmSingleForQueue': {
                'method': 'POST',
                'Referer': 'https://kyfw.12306.cn/otn/confirmPassenger/initDc',
                'Cache-Control': 'no-cache',
                'x-requested-with': 'XMLHttpRequest',
                'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8'
            },
            'https://kyfw.12306.cn/otn/confirmPassenger/queryOrderWaitTime?': {
                'method': 'GET',
                'Referer': 'https://kyfw.12306.cn/otn/confirmPassenger/initDc',
                'x-requested-with': 'XMLHttpRequest'
            },
            'https://kyfw.12306.cn/otn/confirmPassenger/resultOrderForDcQueue': {
                'method': 'POST',
                'Referer': 'https://kyfw.12306.cn/otn/confirmPassenger/initDc',
                'Cache-Control': 'no-cache',
                'x-requested-with': 'XMLHttpRequest',
                'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8'
            },
            'https://kyfw.12306.cn/otn//payOrder/init?': {
                'method': 'POST',
                'Referer': 'https://kyfw.12306.cn/otn/confirmPassenger/initDc',
                'Cache-Control': 'no-cache',
                'Content-Type': 'application/x-www-form-urlencoded'
            },
            'https://kyfw.12306.cn/otn/queryOrder/initNoComplete': {
                'method': 'GET',
                'Referer': 'https://kyfw.12306.cn/otn//payOrder/init?random=1417862054369'
            },
            'https://kyfw.12306.cn/otn/queryOrder/queryMyOrderNoComplete': {
                'method': 'POST',
                'Referer': 'https://kyfw.12306.cn/otn/queryOrder/initNoComplete',
                'Cache-Control': 'no-cache',
                'x-requested-with': 'XMLHttpRequest',
                'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8'
            }
    }
    l = [
        'https://kyfw.12306.cn/otn/passcodeNew/getPassCodeNew?module=login&rand=sjrand&',
        'https://kyfw.12306.cn/otn/passcodeNew/getPassCodeNew?module=passenger&rand=randp&',
        'https://kyfw.12306.cn/otn/leftTicket/log?',
        'https://kyfw.12306.cn/otn/leftTicket/query?',
        'https://kyfw.12306.cn/otn/leftTicket/queryT?',
        'https://kyfw.12306.cn/otn/confirmPassenger/queryOrderWaitTime?',
        'https://kyfw.12306.cn/otn//payOrder/init?'
    ]
    for s in l:
        if url.find(s) == 0:
            url = s
    if not url in d:
        print(u'未知 url: %s' % url)
        return -1
    session.headers.update({'Referer': d[url]['Referer']})
    keys = [
        'Referer',
        'Cache-Control',
        'x-requested-with',
        'Content-Type'
    ]
    for key in keys:
        if key in d[url]:
            session.headers.update({key: d[url][key]})
        else:
            session.headers.update({key: None})
                
def choo_get(session, url):
        chooo_updateHeaders(session,url)
        tries = 0
        while tries < 3:
            tries += 1
            try:
                r = session.get(url, verify=False, timeout=16)
            except requests.exceptions.ConnectionError as e:
                print('ConnectionError(%s): e=%s' % (url, e))
                continue
            except requests.exceptions.Timeout as e:
                print('Timeout(%s): e=%s' % (url, e))
                continue
            except requests.exceptions.TooManyRedirects as e:
                print('TooManyRedirects(%s): e=%s' % (url, e))
                continue
            except requests.exceptions.HTTPError as e:
                print('HTTPError(%s): e=%s' % (url, e))
                continue
            except requests.exceptions.RequestException as e:
                print('RequestException(%s): e=%s' % (url, e))
                continue
            except:
                print('Unknown exception(%s)' % (url))
                continue
            if r.status_code != 200:
                print('Request %s failed %d times, status_code=%d' % (
                    url,
                    tries,
                    r.status_code))
            else:
                return r
        else:
            return None

def choo_post(session, url, payload):
        chooo_updateHeaders(session,url)
        if url == 'https://kyfw.12306.cn/otn/passcodeNew/checkRandCodeAnsyn':
            if payload.find('REPEAT_SUBMIT_TOKEN') != -1:
                session.headers.update({'Referer': 'https://kyfw.12306.cn/otn/confirmPassenger/initDc'})
            else:
                session.headers.update({'Referer': 'https://kyfw.12306.cn/otn/login/init'})
        tries = 0
        while tries < MAX_TRIES:
            tries += 1
            try:
                r = session.post(url, data=payload, verify=False, timeout=16)
            except requests.exceptions.ConnectionError as e:
                print('ConnectionError(%s): e=%s' % (url, e))
                continue
            except requests.exceptions.Timeout as e:
                print('Timeout(%s): e=%s' % (url, e))
                continue
            except requests.exceptions.TooManyRedirects as e:
                print('TooManyRedirects(%s): e=%s' % (url, e))
                continue
            except requests.exceptions.HTTPError as e:
                print('HTTPError(%s): e=%s' % (url, e))
                continue
            except requests.exceptions.RequestException as e:
                print('RequestException(%s): e=%s' % (url, e))
                continue
            except:
                print('Unknown exception(%s)' % (url))
                continue
            if r.status_code != 200:
                print('Request %s failed %d times, status_code=%d' % (
                    url,
                    tries,
                    r.status_code))
            else:
                return r
        else:
            return None      

###  获取验证码，然后验证
def choo_getrandcode(rds,session, url):
    chooo_updateHeaders(session,url)
    #
    #r = session.get(url, verify=False, stream=True, timeout=16)
    r = choo_get(session,url)
    
    if not r :
        return []
    #if 'text/html' in  r.headers['Content-Type'] :    
    
    pscode = Image.open(StringIO.StringIO(r.content) )
    fname = '%d.jpg'%(time.time())   
    
    gcom_log_online(fname,'randn code recognition  begin ... ')
    
    pscode.save('%s%s'%(path_midpro,fname))
     
    texts = rgtt_rgtitle_passcode(rds,pscode,fname)
    
    textstr = ''
    for elm in texts:
        textstr = '%s %s' %(textstr,elm)
    print '*************** recognition text word :', textstr        
   
    randcode = rgim_rg_passcodeonline(rds,pscode,texts,fname)
    
    # randcode 里面多了sflag
    
    ####
    return [randcode ,fname,texts]

def choo_keeptimegap(timest):
    pointgap =  time.clock() - timest 
    print 'test .... time end ',time.clock()
    print 'time gap .....' ,pointgap    
    gapth  = 20    
    if pointgap < gapth :
       print 'will sleep '
       time.sleep(gapth- int (pointgap))   
            
def choo_checkRandCodeAnsyn(rds,session, module,iskeeplive):
        d = {
            'login': {  # 登陆验证码
                'rand': 'sjrand',
                'referer': 'https://kyfw.12306.cn/otn/login/init'
            },
            'passenger': {  # 订单验证码
                'rand': 'randp',
                'referer': 'https://kyfw.12306.cn/otn/confirmPassenger/initDc'
            }
        }
        if not module in d:
            print(u'无效的 module: %s' % (module))
            return [RET_ERR,'']
        tries = 0
              
        
        randcode_num = 0;
        
        while tries < MAX_TRIES:
            #tries += 1,保持一定的时间间隔
            time_st  = time.clock()
            
            #if randcode_num > 0 : 
            print 'test .... time start ',time_st
            
            url = 'https://kyfw.12306.cn/otn/passcodeNew/getPassCodeNew?module=%s&rand=%s&' % (module, d[module]['rand'])
            #if tries > 1:
            #url = '%s%1.16f' % (url, random.random())
            print(u'正在等待验证码...')
            randcode_num = randcode_num +1
            
            pscres  = choo_getrandcode(rds,session, url)
            if len(pscres ) == 0:
                gcom_log_online('  ','randn code recognition fail ,http requests error... ') 
                choo_keeptimegap(time_st)     
                continue 
                
            rancode = pscres[0][0]
            sflag   = pscres[0][1]
            fname  = pscres[1]
            texts  = pscres[2]
            
            
            #### 2016 验证码
            if len(rancode) == 0 :
                gcom_log_online(fname,'randn code recognition fail ,select none ... ') 
                choo_keeptimegap(time_st)                
                continue
                    
            
            url = 'https://kyfw.12306.cn/otn/passcodeNew/checkRandCodeAnsyn'
            parameters = [
               ('randCode', rancode),
                ('rand', d[module]['rand'])
            ]
            if module == 'login':
                parameters.append(('randCode_validate', ''))
            else:
                parameters.append(('_json_att', ''))
                parameters.append(('REPEAT_SUBMIT_TOKEN', repeatSubmitToken))
            payload = urllib.urlencode(parameters)
            print(u'正在校验验证码...')
            r = choo_post(session,url, payload)
            if not r:
                print(u'校验验证码异常')
                gcom_log_online(fname,'randn code recognition fail, post error ') 
                choo_keeptimegap(time_st)
                continue
            # {"validateMessagesShowId":"_validatorMessage","status":true,"httpstatus":200,"data":{"result":"1","msg":"randCodeRight"},"messages":[],"validateMessages":{}}
            obj = r.json()
            if (
                    hasKeys(obj, ['status', 'httpstatus', 'data'])
                    and hasKeys(obj['data'], ['result', 'msg'])
                    and (obj['data']['result'] == '1')):
                print(u'校验验证码成功')
                gcom_log_online(fname,'randn code recognition ok ................') 
                
                rgim_save_subimage(path_midpro,fname,texts,sflag)
                
                if iskeeplive  == 1:
                   choo_keeptimegap(time_st)
                   continue
                   
                else :
                   return    [RET_OK,rancode] 
                
            else:
                gcom_log_online(fname,'randn code recognition fail, image flag mistake ') 
                choo_keeptimegap(time_st) 
                
                print(u'校验验证码失败')
               
                continue
        else:
            return [RET_ERR,'']
            
def choo_loginlearn_Randcode(rds,session):

    ret = choo_checkRandCodeAnsyn(rds,session,'login',1)
    if ret[0] == RET_ERR:
        return RET_ERR
            
            
def choo_login(rds,session):
        url = 'https://kyfw.12306.cn/otn/login/init'
        r = choo_get(session,url)
        if not r:
            print(u'登录失败, 请求异常')
            return RET_ERR
        if session.cookies:
            cookies = requests.utils.dict_from_cookiejar(session.cookies)
            if cookies['JSESSIONID']:
                jsessionid = cookies['JSESSIONID']
        
        ret = choo_checkRandCodeAnsyn(rds,session,'login',0)
        if ret[0] == RET_ERR:
            return RET_ERR
        
        username='××××××××'
        password='×××××××'
        rancode = ret[1]
        
        print(u'正在登录...')
        url = 'https://kyfw.12306.cn/otn/login/loginAysnSuggest'
        parameters = [
            ('loginUserDTO.user_name', username),
            ('userDTO.password', password),
            ('randCode', rancode),
            ('randCode_validate', ''),
            #('ODg3NzQ0', 'OTIyNmFhNmQwNmI5ZmQ2OA%3D%3D'),
            ('myversion', 'undefined')
        ]
        payload = urllib.urlencode(parameters)
        r = choo_post(session,url, payload)
        if not r:
            print(u'登录失败, 请求异常')
            return RET_ERR
        # {"validateMessagesShowId":"_validatorMessage","status":true,"httpstatus":200,"data":{"loginCheck":"Y"},"messages":[],"validateMessages":{}}
        obj = r.json()
        if (
                hasKeys(obj, ['status', 'httpstatus', 'data'])
                and hasKeys(obj['data'], ['loginCheck'])
                and (obj['data']['loginCheck'] == 'Y')):
            print(u'登陆成功^_^')
            url = 'https://kyfw.12306.cn/otn/login/userLogin'
            parameters = [
                ('_json_att', ''),
            ]
            payload = urllib.urlencode(parameters)
            r = choo_post(session,url, payload)
            return RET_OK
        else:
            print(u'登陆失败啦!重新登陆...')
            
            return RET_ERR

def choo_save_cookie(session,username):
    sciptpath = os.path.split(os.path.realpath(__file__))[0] 
    
    #实例化一个LWPcookiejar对象
    new_cookie_jar = cookielib.LWPCookieJar(username )

    #将转换成字典格式的RequestsCookieJar（这里我用字典推导手动转的）保存到LWPcookiejar中
    requests.utils.cookiejar_from_dict({c.name: c.value for c in session.cookies}, new_cookie_jar)

    #保存到本地文件
    new_cookie_jar.save( os.path.join(sciptpath, username), ignore_discard=True, ignore_expires=True)
    
    
def choo_load_cookie(session,username):
   sciptpath = os.path.split(os.path.realpath(__file__))[0] 
   #实例化一个LWPCookieJar对象
   load_cookiejar = cookielib.LWPCookieJar()
   #从文件中加载cookies(LWP格式)
   load_cookiejar.load(os.path.join(sciptpath, username), ignore_discard=True, ignore_expires=True)
   #工具方法转换成字典
   load_cookies = requests.utils.dict_from_cookiejar(load_cookiejar)
   #工具方法将字典转换成RequestsCookieJar，赋值给session的cookies.
   session.cookies = requests.utils.cookiejar_from_dict(load_cookies)


def choo_getPassengerDTOs(session):
        url = 'https://kyfw.12306.cn/otn/confirmPassenger/getPassengerDTOs'
        parameters = [
            ('', ''),
        ]
        payload = urllib.urlencode(parameters)
        r = choo_post(session,url, payload)
        if not r:
            print(u'获取乘客信息异常')
            return []
        obj = r.json()
        if (
                hasKeys(obj, ['status', 'httpstatus', 'data'])
                and hasKeys(obj['data'], ['normal_passengers'])
                and obj['data']['normal_passengers']):
            normal_passengers = obj['data']['normal_passengers']
            print normal_passengers
            return normal_passengers
        else:
            print(u'获取乘客信息失败')
            
            return []            
            
def choo_queryTickets(session,from_city_name,to_city_name,train_date,from_station_telecode,to_station_telecode):
        purpose_code = "ADULT"
       
        url = 'https://kyfw.12306.cn/otn/leftTicket/init'
        parameters = [
            ('_json_att', ''),
            ('leftTicketDTO.from_station_name', from_city_name),
            ('leftTicketDTO.to_station_name', to_city_name),
            ('leftTicketDTO.from_station', from_station_telecode),
            ('leftTicketDTO.to_station', to_station_telecode),
            ('leftTicketDTO.train_date', train_date),
            ('back_train_date', ''),
            ('purpose_codes', purpose_code),
            ('pre_step_flag', 'index')
        ]
        #payload = urllib.urlencode(parameters)
        #r = choo_post(session,url, payload)
        #if not r:
        #    print(u'查询车票异常')

        url = 'https://kyfw.12306.cn/otn/leftTicket/log?'
        parameters = [
            ('leftTicketDTO.train_date', train_date),
            ('leftTicketDTO.from_station', from_station_telecode),
            ('leftTicketDTO.to_station', to_station_telecode),
            ('purpose_codes', purpose_code),
        ]
        url += urllib.urlencode(parameters)
        r = choo_get(session,url)
        if not r:
            print(u'查询车票异常')

        url = 'https://kyfw.12306.cn/otn/leftTicket/queryT?'
        parameters = [
            ('leftTicketDTO.train_date', train_date),
            ('leftTicketDTO.from_station', from_station_telecode),
            ('leftTicketDTO.to_station', to_station_telecode),
            ('purpose_codes', purpose_code),
        ]
        url += urllib.urlencode(parameters)
        r = choo_get(session,url)
        if not r:
            print(u'查询车票异常')
            return [RET_ERR,[]]
        obj = r.json()
        if (hasKeys(obj, ['status', 'httpstatus', 'data']) and len(obj['data'])):
            trains = obj['data']
            return [RET_OK,trains]
        else:
            print(u'查询车票失败')
           
            return [RET_ERR,[]]
# 
def nolg_canbuytrain(trains):
    # 分析查询的车票
    canbuyflag =[];
    
    # sdict['twoseat'] =

    for trainindex in range(0,len(trains)):
        train = trains[trainindex]
        t = train['queryLeftNewDTO']
        
         
        if t['canWebBuy'] == 'N':
            continue
            
        # 仅分析能预订
        i = 0 
        sdict ={}
        while i < (len(t['yp_info']) / 10):
            tmp = t['yp_info'][i * 10:(i + 1) * 10]
            price = int(tmp[1:5])
            left = int(tmp[-3:])
            
            # tmp[0]  座位类型
            if cmp( tmp[0],'O') == 0 and left > 0 :
                
               sdict['trainindex'] = trainindex
               sdict['station_train_code'] = t['station_train_code']
               
               if tmp[6] == '3':
                 sdict['standseat'] = left
               else :
                  sdict['twoseat'] = left
               ## 二等座有票
             
               canbuyflag.append(sdict)                  
            i = i + 1
        
    
    return canbuyflag

def  nolg_selecttrain(canbuyflag, hptrain):
    for  sdt in  canbuyflag:
         if cmp(sdt['station_train_code'],hptrain) == 0:
             return  sdt['trainindex']
             
    return  -1
    
    


    
    
# 返回unicode类型的数据
def  nolg_localstations():
    sciptpath = os.path.split(os.path.realpath(__file__))[0] 
    if not os.path.isfile(os.path.join(sciptpath,'station_name.js')):
        return None
    with open('station_name.js') as fp:
        data = fp.read()
        data = unicode(data, 'utf-8')
    
    station_list = data.split('@')
    if len(station_list) < 1:
            print(u'站点数据库初始化失败, 数据异常')
            return None
    station_list = station_list[1:]
    stations = []
    for station in station_list:
        items = station.split('|')  # bji|北京|BJP|beijing|bj|2
        if len(items) < 5:
                print(u'忽略无效站点: %s' % (items))
                continue
        stations.append({'abbr': items[0],
                             'name': items[1],
                             'telecode': items[2],
                             'pinyin': items[3],
                             'pyabbr': items[4]})
    return stations             
    
# 无需登录 使用 nolg 前缀
def nolg_initStation(session):
        url = 'https://kyfw.12306.cn/otn/resources/js/framework/station_name.js'
        r = choo_get(session,url)
        if not r:
            print(u'站点数据库初始化失败, 请求异常')
            return None
        data = r.text
        sciptpath = os.path.split(os.path.realpath(__file__))[0] 
        
        with open(os.path.join(sciptpath,'station_name.js'), 'wb') as fp:
             fp.write( unicode.encode(data,'utf-8') )
        station_list = data.split('@')
        if len(station_list) < 1:
            print(u'站点数据库初始化失败, 数据异常')
            return None
        station_list = station_list[1:]
        stations = []
        for station in station_list:
            items = station.split('|')  # bji|北京|BJP|beijing|bj|2
            if len(items) < 5:
                print(u'忽略无效站点: %s' % (items))
                continue
            stations.append({'abbr': items[0],
                             'name': items[1],
                             'telecode': items[2],
                             'pinyin': items[3],
                             'pyabbr': items[4]})
        return stations           
# 检查日期
def nolg_checkDate(date):
    m = re.match(r'^\d{4}-\d{1,2}-\d{1,2}$', date)  # 2014-01-01
    if m:
        today = datetime.datetime.now()
        fmt = '%Y-%m-%d'
        today = datetime.datetime.strptime(today.strftime(fmt), fmt)
        train_date = datetime.datetime.strptime(m.group(0), fmt)
        delta = train_date - today
        if delta.days < 0:
            print(u'乘车日期%s无效, 只能预订%s以后的车票' % (
                train_date.strftime(fmt),
                today.strftime(fmt)))
            return False
        else:
            return True
    else:
        return False
        
def nolg_getStationByName(name,stations):
    matched_stations = []
    for station in stations:
        if (
                station['name'] == name
                or station['abbr'].find(name.lower()) != -1
                or station['pinyin'].find(name.lower()) != -1
                or station['pyabbr'].find(name.lower()) != -1):
            matched_stations.append(station)
    count = len(matched_stations)
    if count <= 0:
        return None
    elif count == 1:
        return matched_stations[0]
    else:
        for i in xrange(0, count):
            print(u'%d:\t%s' % (i + 1, matched_stations[i]['name']))
        print(u'请选择站点(1~%d)' % (count))
        index = raw_input()
        if not index.isdigit():
            print(u'只能输入数字序号(1~%d)' % (count))
            return None
        index = int(index)
        if index < 1 or index > count:
            print(u'输入的序号无效(1~%d)' % (count))
            return None
        else:
            return matched_stations[index - 1]

# 预订车票
def choo_initOrder(session,trains,selindex,train_date,from_city_name,to_city_name):
        purpose_code = "ADULT"
        url = 'https://kyfw.12306.cn/otn/login/checkUser'
        parameters = [
            ('_json_att', ''),
        ]
        payload = urllib.urlencode(parameters)
        r = choo_post(session,url, payload)
        if not r:
            print(u'初始化订单异常')

        print(u'准备下单喽')
        url = 'https://kyfw.12306.cn/otn/leftTicket/submitOrderRequest'
        parameters = [
            #('ODA4NzIx', 'MTU0MTczYmQ2N2I3MjJkOA%3D%3D'),
            ('myversion', 'undefined'),
            ('secretStr', trains[selindex]['secretStr']),
            ('train_date', train_date),
            ('back_train_date', ''),
            ('tour_flag', 'dc'),              # 单程
            ('purpose_codes', purpose_code),
            #('query_from_station_name', unicode.encode(from_city_name,'utf-8')   ),
            #('query_to_station_name', unicode.encode(to_city_name,'utf-8')   ),
            ('undefined', '')
        ]
        # TODO 注意:此处post不需要做urlencode, 比较奇怪, 不能用urllib.urlencode(parameters)
        payload = ''
        length = len(parameters)
        for i in range(0, length):
            payload += parameters[i][0] + '=' + parameters[i][1]
            if i < (length - 1):
                payload += '&'
        r = choo_post(session,url, payload)
        if not r:
            print(u'下单异常')
            return RET_ERR
        # {"validateMessagesShowId":"_validatorMessage","status":true,"httpstatus":200,"messages":[],"validateMessages":{}}
        obj = r.json()
        if not (hasKeys(obj, ['status', 'httpstatus'])
                and obj['status']):
            print(u'下单失败啦')
          
            return RET_ERR

        print(u'订单初始化...')
        session.close()  # TODO
        url = 'https://kyfw.12306.cn/otn/confirmPassenger/initDc'
        parameters = [
            ('_json_att', ''),
        ]
        payload = urllib.urlencode(parameters)
        r = choo_post(session,url, payload)
        if not r:
            print(u'订单初始化异常')
            return RET_ERR
        data = r.text
        s = data.find('globalRepeatSubmitToken')  # TODO
        e = data.find('global_lang')
        if s == -1 or e == -1:
            print(u'找不到 globalRepeatSubmitToken')
            return RET_ERR
        buf = data[s:e]
        s = buf.find("'")
        e = buf.find("';")
        if s == -1 or e == -1:
            print(u'很遗憾, 找不到 globalRepeatSubmitToken')
            return RET_ERR
        repeatSubmitToken = buf[s + 1:e]

        s = data.find('key_check_isChange')
        e = data.find('leftDetails')
        if s == -1 or e == -1:
            print(u'找不到 key_check_isChange')
            return RET_ERR
        keyCheckIsChange = data[s + len('key_check_isChange') + 3:e - 3]

        return repeatSubmitToken            

def choo_checkOrderInfo(rds,session,passengers,repeatSubmitToken ):
        ret = choo_checkRandCodeAnsyn(rds,session,'passenger',0) 
  

            
        if ret[0] == RET_ERR:
            return RET_ERR
        
      
        rancode = ret[1]
        
        
        passengerTicketStr = ''
        oldPassengerStr = ''
        passenger_seat_detail = '0'  # TODO [0->随机][1->下铺][2->中铺][3->上铺]
        for p in passengers:
            if p['index'] != 1:
                passengerTicketStr += 'N_'
                oldPassengerStr += '1_'
            passengerTicketStr += '%s,%s,%s,%s,%s,%s,%s,' % (
                p['seattype'],
                passenger_seat_detail,
                p['tickettype'],
                p['name'],
                p['cardtype'],
                p['id'],
                p['phone'])
            oldPassengerStr += '%s,%s,%s,' % (
                p['name'],
                p['cardtype'],
                p['id'])
        passengerTicketStr += 'N'
        oldPassengerStr += '1_'
        
        passengerTicketStr = passengerTicketStr
        oldPassengerStr = oldPassengerStr

        print(u'检查订单...')
        url = 'https://kyfw.12306.cn/otn/confirmPassenger/checkOrderInfo'
        parameters = [
            ('cancel_flag', '2'),  # TODO
            ('bed_level_order_num', '000000000000000000000000000000'),  # TODO
            ('passengerTicketStr', passengerTicketStr),
            ('oldPassengerStr', oldPassengerStr),
            ('tour_flag', 'dc'),
            ('randCode', rancode),
            #('NzA4MTc1', 'NmYyYzZkYWY2OWZkNzg2YQ%3D%3D'),  # TODO
            ('_json_att', ''),
            ('REPEAT_SUBMIT_TOKEN', repeatSubmitToken),
        ]
        payload = urllib.urlencode(parameters)
        r = choo_post(session,url, payload)
        if not r:
            print(u'检查订单异常')
            return RET_ERR
        # {"validateMessagesShowId":"_validatorMessage","status":true,"httpstatus":200,"data":{"submitStatus":true},"messages":[],"validateMessages":{}}
        obj = r.json()
        if (
                hasKeys(obj, ['status', 'httpstatus', 'data'])
                and hasKeys(obj['data'], ['submitStatus'])
                and obj['status']
                and obj['data']['submitStatus']):
            print(u'检查订单成功')
            return RET_OK
        else:
            print(u'检查订单失败')
            
            return RET_ERR
            
            
def choo_auto_order(rds,session,train_date,from_city_name,to_city_name,hptrain,stations):
    
    ret = choo_login(rds,session)
    
    if ret != RET_OK  :
    
        return -1
        
    choo_save_cookie(session,'×××××××')
    
    stasion_1 = nolg_getStationByName(from_city_name,stations)
    stasion_2 = nolg_getStationByName(to_city_name,stations)
    #from_city_name,to_city_name,train_date,from_station_telecode,to_station_telecode
    tre = choo_queryTickets(session,from_city_name,to_city_name,train_date,stasion_1['telecode'],stasion_2['telecode'])
    
    if tre[0] == RET_OK :
      trains = tre[1]
      canbuy = nolg_canbuytrain(trains)
      selindex = nolg_selecttrain(canbuy,'D3077')
      if selindex  == -1:
         return -3
    else:
        return  -2
       
  
    ret = choo_initOrder(session,trains,selindex,train_date,from_city_name,to_city_name)
    
def choo_auto_order_cookie(rds,session,train_date,from_city_name,to_city_name,hptrain,stations):
 
    stasion_1 = nolg_getStationByName(from_city_name,stations)
    stasion_2 = nolg_getStationByName(to_city_name,stations)
    #from_city_name,to_city_name,train_date,from_station_telecode,to_station_telecode
    tre = choo_queryTickets(session,from_city_name,to_city_name,train_date,stasion_1['telecode'],stasion_2['telecode'])
    
    if tre[0] == RET_OK :
      trains = tre[1]
      canbuy = nolg_canbuytrain(trains)
      selindex = nolg_selecttrain(canbuy,'D3077')
      if selindex  == -1:
         return -3
    else:
        return  -2
       
  
    ret = choo_initOrder(session,trains,selindex,train_date,from_city_name,to_city_name)
    
    return  ret
    
def choo_batch_getpasscode(rds,session):
    
    d = {
            'login': {  # 登陆验证码
                'rand': 'sjrand',
                'referer': 'https://kyfw.12306.cn/otn/login/init'
            },
            'passenger': {  # 订单验证码
                'rand': 'randp',
                'referer': 'https://kyfw.12306.cn/otn/confirmPassenger/initDc'
            }
        }
      
    url = 'https://kyfw.12306.cn/otn/passcodeNew/getPassCodeNew?module=%s&rand=%s&' % ('passenger', d['passenger']['rand'])
    
    choo_getPassengerDTOs(session)    
            
    while  1:
    
       chooo_updateHeaders(session,url)
       r = session.get(url, verify=False, stream=True, timeout=16)
       #if 'text/html' in  r.headers['Content-Type'] :    
    
       pscode = Image.open(StringIO.StringIO(r.content) )
       fname = '%d.jpg'%(time.time())   
    
       pscode.save('%s%s'%('F:\\pichog\\passcode02\\',fname))
       time.sleep(3)
     
        
if __name__ == '__main__':
    
    rds = redis.Redis(host='localhost',port=6379,db=0)
    
    gcom_online_loginit()
    session = chooo_initSession()
    
    stations = nolg_localstations()
    
    if len(sys.argv) == 1:
        print  'useage    /'
    else : 
     
 # python E:\picdog\bear\actionapi.py train     
 # python -mpdb  E:\picdog\bear\actionapi.py loginlearn  
        if  cmp(sys.argv[1],'train') == 0 :
            tre = choo_auto_order(rds,session,'2016-10-13',u'南京',u'武汉','G113',stations)    
        elif  cmp(sys.argv[1],'cook') == 0 :
            choo_load_cookie(session,'×××××××')   
            passengers  = choo_getPassengerDTOs(session)
            ret = choo_auto_order_cookie(rds,session,'2016-10-13',u'南京',u'武汉','G113',stations)    
            
            if isinstance(ret,unicode):
                repeatSubmitToken = ret
                choo_checkOrderInfo(rds,session,passengers,repeatSubmitToken)
            else :
               pass
        elif  cmp(sys.argv[1],'batch') == 0 :
             choo_load_cookie(session,'×××××××')   
             choo_batch_getpasscode(rds,session)
        elif cmp(sys.argv[1],'loginlearn') == 0 :
             choo_loginlearn_Randcode(rds,session)            
        else :
           print  'useage  load /test /'
           
           
    
    
   
    
    #choo_login(rds,session)
    
    #nor_pasg = choo_getPassengerDTOs(session)