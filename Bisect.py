
# coding: utf-8

# In[ ]:

def bisect(array,value):
    '''Given an ``array`` , and given a ``value`` , returns an index j such that ``value`` is between array[j]
    and array[j+1]. ``array`` must be monotonic increasing. j=-1 or j=len(array) is returned
    to indicate that ``value`` is out of range below and above respectively.'''
    #return bisection(array,value)
    n = len(array)
    if (value < array[0]):
        return -1
        res = -1# Then set the output
    elif (value > array[n-1]):
        return n
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
        return 0
        res = -1# Then set the output
    elif (value == array[n-1]):
        return n-1
    else:
        return jl

