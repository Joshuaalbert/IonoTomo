import numpy as np

def cho_back_substitution(L,y,lower=True,modify=False):
    if len(y.shape) == 2:
        if not modify:
            x = np.copy(y)
        else:
            x = y
        if lower:
            i = 0
            while i < L.shape[0]:
                x[i,:] /= L[i,i]
                x[i+1:,:] -= np.outer(L[i+1:,i],x[i,:])
                i += 1  
        else:
            i = L.shape[0] - 1
            while i >= 0:
                x[i,:] /= L[i,i]
                x[:i,:] -= np.outer(L[:i,i],x[i,:])
                i -= 1
    else:
        if not modify:
            x = np.copy(y)
        else:
            x = y
        if lower:
            i = 0
            while i < L.shape[0]:
                x[i] /= L[i,i]
                x[i+1:] -= L[i+1:,i]*x[i]
                i += 1  
        else:
            i = L.shape[0] - 1
            while i >= 0:
                x[i] /= L[i,i]
                x[:i] -= L[:i,i]*x[i]
                i -= 1
    return x

def cho_solve(L,b,modify=False):
    #second can be modified always as it is last reference
    return cho_back_substitution(L.T,cho_back_substitution(L,b,True,modify),False,True)

