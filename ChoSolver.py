
# coding: utf-8

# In[ ]:

import numpy as np
#from numba import jit

#@jit
def choBackSubstitution(L,y,lower=True,modify=False):
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

#@jit
def choSolve(L,b,modify=False):
    #second can be modified always as it is last reference
    return choBackSubstitution(L.T,choBackSubstitution(L,b,True,modify),False,True)

if __name__ == '__main__':
    #from scipy.linalg import cho_solve
    from scipy.linalg.lapack import dpotrs
    N = 5
    y = np.random.uniform(size=N)
    Y = np.random.uniform(size=[N,2])
    a = np.random.uniform(size=[N,N])
    a = a.T.dot(a)
    L = np.linalg.cholesky(a)

    X = choSolve(L,Y,False)
    xa = choSolve(L,Y[:,0],False)
    xb = choSolve(L,Y[:,1],False)
    assert np.alltrue(np.isclose(X[:,0],xa)),"a fails"
    assert np.alltrue(np.isclose(X[:,1],xb)),"b fails"
    get_ipython().magic('load_ext line_profiler')
    get_ipython().magic('lprun -f choBackSubstitution choSolve(L,y,False)')
    #with y vec mod (no copy)
    get_ipython().magic('timeit -n 10 choSolve(L,y,False)')
    #built in
    #%timeit cho_solve((L,True),y)
    get_ipython().magic('timeit -n 10 dpotrs(L,y,1,0)')
    #x1 = cho_solve((L,True),y)
    x1 = dpotrs(L,y,1,0)
    x2 = choSolve(L,y,False)
    #x1 = dpotrs(L,y,1,1)
    print("same:",np.alltrue(np.isclose(x1[0],x2)))
    times1 = []
    times2 = []
    Ns = 10**np.linspace(1,4,20)
    from time import clock
    for N in Ns:
        N = int(N)
        y = np.random.uniform(size=N)
        a = np.random.uniform(size=[N,N])
        a = a.T.dot(a)
        L = np.linalg.cholesky(a)
        t1 = clock()
        #x1 = cho_solve((L,True),y)
        x1 = dpotrs(L,y,1,0)
        times1.append(clock()-t1)
        t1 = clock()
        x2 = choSolve(L,y,False)
        times2.append(clock()-t1)
    import pylab as plt
    plt.plot(Ns,times1,label='scipy.linalg.cho_solve')
    plt.plot(Ns,times2,label='my choSolve')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.show()

