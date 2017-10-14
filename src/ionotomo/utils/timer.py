from time import gmtime, mktime
from timeit import default_timer

def clock():
    '''UTC time since epoch'''
    return default_timer()
    return mktime(gmtime())
