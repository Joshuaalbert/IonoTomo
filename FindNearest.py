
# coding: utf-8

# In[44]:

'''Compare algorithms for find nearest'''
import math
import numpy as np

def find_nearest1(array,value):
    '''not right function and non vector'''
    idx,val = min(enumerate(array), key=lambda x: abs(x[1]-value))
    return idx

def find_nearest2(array, values):
    '''not right function'''
    indices = np.abs(np.subtract.outer(array, values)).argmin(0)
    return indices

def find_nearest3(array, values):
    values = np.atleast_1d(values)
    indices = np.abs(np.int64(np.subtract.outer(array, values))).argmin(0)
    out = array[indices]
    return indices

def find_nearest4(array,value):
    '''not right function and non vector'''
    idx = (np.abs(array-value)).argmin()
    return idx


def find_nearest5(array, value):
    '''not right function and non vector'''
    idx_sorted = np.argsort(array)
    sorted_array = np.array(array[idx_sorted])
    idx = np.searchsorted(sorted_array, value, side="left")
    if idx >= len(array):
        idx_nearest = idx_sorted[len(array)-1]
    elif idx == 0:
        idx_nearest = idx_sorted[0]
    else:
        if abs(value - sorted_array[idx-1]) < abs(value - sorted_array[idx]):
            idx_nearest = idx_sorted[idx-1]
        else:
            idx_nearest = idx_sorted[idx]
    return idx_nearest

def find_nearest6(array,value):
    xi = np.argmin(np.abs(np.ceil(array[None].T - value)),axis=0)
    return xi

def bisection(array,value,lower = -np.inf, upper=np.inf):
    '''Given an ``array`` , and given a ``value`` , returns an index j such that ``value`` is between array[j]
    and array[j+1]. ``array`` must be monotonic increasing. j=-1 or j=len(array) is returned
    to indicate that ``value`` is out of range below and above respectively.'''
    n = len(array)
    if (value < array[0]):
        return -1,lower
        res = -1# Then set the output
    elif (value > array[n-1]):
        return n, upper
    #array = np.append(np.append(-np.inf,array),np.inf)
    jl = 0# Initialize lower
    ju = n-1# and upper limits.
    while (ju-jl > 1):# If we are not yet done,
        jm=(ju+jl) >> 1# compute a midpoint,
        if (value >= array[jm]):
            jl=jm# and replace either the lower limit
        else:
            ju=jm# or the upper limit, as appropriate.
        # Repeat until the test condition is satisfied.
    if (value == array[0]):
        return 0,array[0]
        res = -1# Then set the output
    elif (value == array[n-1]):
        return n-1,array[n-1]
    else:
        return jl, array[jl]

if __name__=='__main__':
    array = np.arange(100000)

    val = array[50000]+0.55
    print( bisection(array,val))
    get_ipython().magic(u'timeit bisection(array,val)')
    print( find_nearest1(array,val))
    get_ipython().magic(u'timeit find_nearest1(array,val)')
    print( find_nearest2(array,val))
    get_ipython().magic(u'timeit find_nearest2(array,val)')
    print( find_nearest3(array,val))
    get_ipython().magic(u'timeit find_nearest3(array,val)')
    print( find_nearest4(array,val))
    get_ipython().magic(u'timeit find_nearest4(array,val)')
    print( find_nearest5(array,val))
    get_ipython().magic(u'timeit find_nearest5(array,val)')
    print( find_nearest6(array,val))
    get_ipython().magic(u'timeit find_nearest6(array,val)')
    


# In[8]:

(2, 2)
100000 loops, best of 3: 4.36 µs per loop
3
10 loops, best of 3: 143 ms per loop
3
10000 loops, best of 3: 203 µs per loop
[2]
1000 loops, best of 3: 380 µs per loop
3
1000 loops, best of 3: 197 µs per loop
3
1000 loops, best of 3: 876 µs per loop
[2]
1000 loops, best of 3: 1.05 ms per loop


# In[ ]:



