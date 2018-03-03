#_*_ coding:utf-8 _*_  

#from actionapi import  *
import redis
import os

        

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

            
            
            
if __name__ == '__main__':
  

    stations = nolg_localstations()
    nolg_getStationByName(u'北京',stations)
    