import numpy as np
from ionotomo.utils.cho_solver import *

def test_cho_solver():
    from scipy.linalg.lapack import dpotrs
    N = 5
    y = np.random.uniform(size=N)
    Y = np.random.uniform(size=[N,2])
    a = np.random.uniform(size=[N,N])
    a = a.T.dot(a)
    L = np.linalg.cholesky(a)

    X = cho_solve(L,Y,False)
    xa = cho_solve(L,Y[:,0],False)
    xb = cho_solve(L,Y[:,1],False)
    assert np.alltrue(np.isclose(X[:,0],xa)),"a fails"
    assert np.alltrue(np.isclose(X[:,1],xb)),"b fails"
    #with y vec mod (no copy)
    #built in
    #x1 = cho_solve((L,True),y)
    x1 = dpotrs(L,y,1,0)
    x2 = cho_solve(L,y,False)
    #x1 = dpotrs(L,y,1,1)
    assert np.all(np.isclose(x1[0],x2))
#    times1 = []
#    times2 = []
#    Ns = 10**np.linspace(1,4,10)
#    from time import clock
#    for N in Ns:
#        N = int(N)
#        y = np.random.uniform(size=N)
#        a = np.random.uniform(size=[N,N])
#        a = a.T.dot(a)
#        L = np.linalg.cholesky(a)
#        t1 = clock()
#        #x1 = cho_solve((L,True),y)
#        x1 = dpotrs(L,y,1,0)
#        times1.append(clock()-t1)
#        t1 = clock()
#        x2 = cho_solve(L,y,False)
#        times2.append(clock()-t1)
#    import pylab as plt
#    plt.plot(Ns,times1,label='scipy.linalg.cho_solve')
#    plt.plot(Ns,times2,label='my choSolve')
#    plt.yscale('log')
#    plt.xscale('log')
#    plt.legend()
#    plt.show()
