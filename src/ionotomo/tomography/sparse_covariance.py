import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import coo_matrix
import pylab as plt

def squared_exponential(x2,D=3):
    #x = np.reshape(x,(-1,D))
    return np.exp(-x2/2.)

def matern52(x2):
    x = np.sqrt(x2)
    res = x2
    res *= 5./3.
    res += np.sqrt(5) * x
    res += 1
    res *= np.exp((-np.sqrt(5))*x)
    return res


def sparse_covariance(cfun, points, sigma, corr,tol=0.1,upper_tri=True):
    N,D = points.shape
    if not isinstance(corr,np.ndarray):
        corr = np.ones(D)*corr
        
    #Get support
    if tol == 0.:
        isot = np.inf
    else:
        isot = 0.
        for dim in range(D):
            direction = (np.arange(D)==dim).astype(float)
            t = 0
            c0 = cfun(0)
            c = c0
            while c/c0 > tol:
                t += 0.1
                c = cfun(np.sum((t*direction/corr)**2))
            isot = max(isot,t/corr[dim])
    #print("isotropic support: {}".format(isot))
    kd = cKDTree(points/corr)

    if upper_tri:
        pairs = kd.query_pairs(isot,p=2,output_type='ndarray')
        pairs = np.concatenate([np.array([np.arange(N)]*2).T,pairs])

        x1 = points[pairs[:,0],:]
        x2 = points[pairs[:,1],:]
        dx = x1-x2
        dx /= corr
        dx *= dx
        dx = np.sum(dx,axis=1)

        cval = cfun(dx)
        csparse = coo_matrix((cval,(pairs[:,0],pairs[:,1])), shape=(N,N))
    else:
        X = kd.sparse_distance_matrix(kd,isot,output_type='coo_matrix')
        cval = cfun(X.data**2)
        csparse = coo_matrix((cval,(X.col,X.row)), shape=(N,N))
        
    return (sigma**2)*csparse

def dense_covariance(cfun, points, sigma, corr):
    N,D = points.shape
    if not isinstance(corr,np.ndarray):
        corr = np.ones(D)*corr
    points = points / corr
    X = squareform(pdist(points,metric='sqeuclidean'))
    return (sigma**2)*cfun(X)

def test_sparse_covariance():
    corr = np.array([0.2,0.5,0.1])

    xvec = np.linspace(0,1,50)
    yvec = np.linspace(0,1,10)
    zvec = np.linspace(0,1,10)

    X,Y,Z = np.meshgrid(xvec,yvec,zvec,indexing='ij')
    points = np.array([X.flatten(),Y.flatten(), Z.flatten()]).T

    #%timeit -n 2 cdense = dense_covariance(squared_exponential, points, None, corr)
    cdense = dense_covariance(matern52, points, 1., corr)
#    #print(cdense)
#    plt.imshow(cdense)
#    plt.colorbar()
#    plt.show()

    #%timeit -n 2 csparse = sparse_covariance(squared_exponential,points,None,corr,tol=0.1)
    csparse = sparse_covariance(matern52,points,1.,corr,tol=0,upper_tri=False)
    assert np.all(np.isclose(csparse.toarray(), cdense))
#    #print(csparse.toarray())
#    plt.imshow(csparse.toarray())
#    plt.colorbar()
#    plt.show()
#    plt.imshow(csparse.toarray() - cdense)
#    plt.colorbar()
#    plt.show()

    csparse = sparse_covariance(matern52,points,1.,corr,tol=0.1,upper_tri=True)

    print("upper triangle tol=0.1 -> saving: {}%".format(1-csparse.nonzero()[0].size/cdense.size))
    csparse = sparse_covariance(matern52,points,1.,corr,tol=0.01,upper_tri=True)

    print("upper triangle tol=0.01 -> saving: {}%".format(1-csparse.nonzero()[0].size/cdense.size))

def test_sparse_covariance_performance():
    corr = np.array([5.,5.,1.])

    xvec = np.linspace(-80,80,150)
    yvec = np.linspace(-80,80,150)
    zvec = np.linspace(0,1000,20)

    X,Y,Z = np.meshgrid(xvec,yvec,zvec,indexing='ij')
    points = np.array([X.flatten(),Y.flatten(), Z.flatten()]).T

    
    csparse = sparse_covariance(matern52,points,1.,corr,tol=0.1,upper_tri=True)

    print("upper triangle tol=0.1 -> saving: {}%".format(1-csparse.nonzero()[0].size/points.size**2))

if __name__=='__main__':
    test_sparse_covariance_performance()
