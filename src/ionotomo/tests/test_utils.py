import numpy as np
from ionotomo.utils.cho_solver import *
from ionotomo.utils.gaussian_process import *

def test_level2_solve():
    np.random.seed(1234)
    K1 = SquaredExponential(2,l=0.29,sigma=3.7)
    #K1.fixed = 'l'
    #K1.fixed = 'sigma'
    K2 = Diagonal(2,sigma=1e-5)
    K2.fixed = 'sigma'
    K3 = RationalQuadratic(2,sigma=1.)
    K4 = MaternPIso(2,p=2)
    K6 = GammaExponential(2)
    K7 = PeriodicSep(2,0,l=0.5)
    K7.fixed = 'l'
    K8 = PeriodicSep(2,1,l=0.5)
    K8.fixed='l'
    K = K3 *K1+K2
    hp = K.hyperparams 
    x = np.random.uniform(size=[250,2])
    xstar = np.linspace(-1,2,100)
    Xstar,Ystar = np.meshgrid(xstar,xstar)
    xstar = np.array([Xstar.flatten(),Ystar.flatten()]).T
    y = np.sin(x[:,0]*2*np.pi/0.5) *np.cos( x[:,1]*np.pi/0.5*2.) + np.random.normal(size=x.shape[0])*0.1
    m_y = np.mean(y)
    y -= m_y
    sigma_y = 0.1
    hyperparams = level2_solve(x,y,sigma_y,K)
    K.hyperparams = hyperparams
    print(K)
    fstar,cov,log_mar_like = level1_solve(x,y,sigma_y,xstar,K)
    import pylab as plt
    vmin = np.min(y) + m_y
    vmax = np.max(y) + m_y
    plt.imshow(fstar.reshape(Xstar.shape)+m_y,extent=(-1,2,-1,2),origin='lower',vmin=vmin,vmax=vmax)
    plt.scatter(x[:,0],x[:,1],c=y+m_y)
    #plt.scatter(xstar[:,0],xstar[:,1],c=fstar,marker='+')
    plt.show()


def test_log_mar_like_func():
    K1 = SquaredExponential(2)
    K2 = Diagonal(2)
    K3 = RationalQuadratic(2)
    K4 = GammaExponential(2)
    K = K1 + K2 + K3 + K4
    print(K)
    hp = K.hyperparams 
    x = np.random.uniform(size=[100,2])
    y = x[:,0]**2 + x[:,1]**(1./3.) + np.random.normal(size=100)*0.1
    xstar = np.random.uniform(size=[100,2])
    sigma_y = 0.1
    lml,dlml = neg_log_mar_like_and_derivative(hp,x,y,sigma_y,K)
    lml_ = log_mar_like(hp,x,y,sigma_y,K)
    eps=1e-5
    grad = np.zeros(len(hp))
    for i in range(len(hp)):
        hp[i] += eps
        l_ = log_mar_like(hp,x,y,sigma_y,K)
        grad[i] = -(l_ + lml)/eps
        hp[i] -= eps
    print(grad,dlml)


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
