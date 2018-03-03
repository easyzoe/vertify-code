from actionapi import  *
import redis
import os


if __name__ == '__main__':
    
    rds = redis.Redis(host='localhost',port=6379,db=0)
    
    
    session = chooo_initSession()
    
    