import numpy as np
from scipy.linalg import cho_solve

from scipy.integrate import simps
from scipy.spatial import cKDTree
import dask.array as da
from dask import delayed
import dask.array as da
import os

from ionotomo.plotting.plot_tools import plot_datapack
from ionotomo.utils.cho_solver import cho_back_substitution
import logging as log
TECU = 1e13 #TEC unit / km
speedoflight = 299792458. #m/s

def neg_log_like(g,dobs,CdCt,covariance,model,model_prior,tci,full=False):
    """Get the misfit for current model point.
    g : array
        theory data shape (Na,Nt,Nd,Nf)
    dobs : array
        observational data shape like g
    CdCt : array
        diagonal measurement + theory uncertainty 
    covariance : tuple (Covariance object, float)
        covariance for mu and clock
    model : tuple (array, array, array)
        the model for mu, clock, const
    tci : TriCubic 
        The interpolator
    """
    dd = dobs - g
    dd *= dd
    dd /= CdCt
#    print(CdCt)
#    print(CdCt == 0)
#    print(np.any(CdCt == 0))
    l2 = np.sum(dd)/2.

        
    if full:
        c_mu, c_clock = covariance
        mu,clock,const = model
        mu_prior, clock_prior, const_prior = model_prior

        dclock = clock - clock_prior
        dclock *= dclock
        dclock /= c_clock
        l2 += np.sum(dclock)/2.

        dmu = mu_prior - mu
        tci.M = dmu
        tci.M = c_mu.contract(tci.M)
        l2 += tci.inner(dmu, inplace = True)/2.

    return l2

def solve_sym_posdef(A,b):
    """Solve the system using cholesky decomposition if numerically pos def or else use pseudo inverse.
    A : array
        The LHS constraint matrix
    b : array
        the RHS constraint matrix
    """
    try:
        L = np.linalg.cholesky(A)
        x = cho_solve((L,True),b)
    except:
        log.debug("Matrix not pos-def, using pinv")
        x = np.linalg.pinv(A).dot(b)
    return x

def cholesky_inner(A,b):
    """calculate b.A^_1.b using cholesky or else pinv"""
    try:
        L = np.linalg.cholesky(A)
        x = cho_back_substitution(L,b,True,False)
        return x.T.dot(x)
    except:
        log.debug("Matrix not pos-def, using pinv")
        inv = np.linalg.pinv(A)
        return np.linalg.multi_dot([b.T,inv,b])



def forward_equation(model, tci, rays, freqs, K=1e11, i0 = 0):
    '''Calculates the phase from the given model point.
    model : tuple (mu, clock, const)
        mu is flattened model, clock is shape (Na, Nt), const is shape (Na,)
    tci : TriCubic
        interpolator for the region
    rays : array
        array of shape (Na, Nt, Nd, 4, Ns), the third dimension is x,y,z,s in that order
    freqs : array of shape (Nf,)
        the frequencies
    K : float (optional)
        the log-transform constant, default 1e11
    i0 : int (optional)
        the reference antenna index, default 0
    '''
    Na,Nt,Nd,_,Ns = rays.shape
    Nf = freqs.shape[0]
    mu,clock,const = model
    ne = np.exp(mu)
    ne *= K
    tci.M = ne
    g = np.einsum("i,j,k,l,i->ijkl",np.ones(Na),np.ones(Nt),np.ones(Nd),np.ones(Nf),const)
    ne_rays = tci.interp(rays[:,:,:,0,:],rays[:,:,:,1,:],rays[:,:,:,2,:])
    for l in range(Nf):
        a_ = 2*np.pi * freqs[l]
        dg = a_ * np.einsum("ij,k->ijk",clock,np.ones(Nd))
        n_p = 1.2404e-2*freqs[l]**2
        n_rays = ne_rays/(-n_p)
        n_rays += 1
        np.sqrt(n_rays,out=n_rays)
        #turn n_rays to 1 - n
        n_rays *= -1
        n_rays += 1
        phi_ion =  simps(n_rays,rays[:,:,:,3,:],axis=3)
        phi_ion -= phi_ion[i0,...]
        phi_ion *= (a_ / speedoflight)
        dg -= phi_ion
        g[:,:,:,l] += dg
#    print(g)
#    print(np.isnan(g))
#    print(np.any(np.isnan(g)))
    return g

def data_residuals(dobs,g):
    '''Do the unweighted data residuals
    dobs : array
        shape (Na, Nt, Nd, Nf) the phase
    g : array
        same shape as dobs, the model phase
    '''
    return dobs - g

def prior_penalty_mu(model, model_prior, tci, rays, freqs, K=1e11, i0 = 0):
    '''Calculates the regularisation penalty imposed by prior in mu
    model : tuple (mu, clock, const)
        mu is flattened model, clock is shape (Na, Nt), const is shape (Na,)
        Note: const is unused here.
    model : tuple (mu_prior, clock_prior)
        priors of the model, same shape.
        Note: const has no prior
    tci : TriCubic
        interpolator for the region
    rays : array
        array of shape (Na, Nt, Nd, 4, Ns), the third dimension is x,y,z,s in that order
    freqs : array of shape (Nf,)
        the frequencies
    K : float (optional)
        the log-transform constant, default 1e11
    i0 : int (optional)
        the reference antenna index, default 0
    '''
    Na,Nt,Nd,_,Ns = rays.shape
    Nf = freqs.shape[0]
    mu,clock,const = model
    mu_prior, clock_prior,const_prior = model_prior
    ne = np.exp(mu)
    ne *= K
    tci.M = ne
    r = np.zeros([Na,Nt,Nd,Nf],dtype=float)
    ne_rays = tci.interp(rays[:,:,:,0,:],rays[:,:,:,1,:],rays[:,:,:,2,:])
    #dm
    tci.M = mu_prior - mu
    dmu_rays = tci.interp(rays[:,:,:,0,:],rays[:,:,:,1,:],rays[:,:,:,2,:])
    for l in range(Nf):
        n_p = 1.2404e-2*freqs[l]**2
        a_ = 2*np.pi * freqs[l]
        b_ = a_ / (2 * n_p * speedoflight)
        #dr = a_ * np.einsum("ij,k->ijk",dclock,np.ones(Nd))
        n_rays = ne_rays/(-n_p)
        n_rays += 1
        np.sqrt(n_rays,out=n_rays)
        integrand = ne_rays / n_rays
        integrand *= dmu_rays
        ion =  simps(integrand,rays[:,:,:,3,:],axis=3)
        ion -= ion[i0,...]
        ion *= b_
        #dr -= ion
        r[:,:,:,l] -= ion
    return r

def signal(dd, prior_penalty):
    '''Calculate the signal.
    dd : array
        phase residuals (unweighted) shape (Na,Nt,Nd,Nf)
    prior_penalty: array
        penalty from the prior first order.
        shape same as dd
    '''
    return dd - prior_penalty

#@profile
def signal_covariance(model, covariance, tci, rays, freqs, CdCt, num_threads, K=1e11, i0 = 0):
    '''Calculates the phase from the given model point.
    model : tuple (mu, clock, const)
        mu is flattened model, clock is shape (Na, Nt), const is shape (Na,)
    covariance : tuple (Covariance, float)
        the Covariance object for mu, and float variance for clock
    tci : TriCubic
        interpolator for the region
    rays : array
        array of shape (Na, Nt, Nd, 4, Ns), the third dimension is x,y,z,s in that order
    freqs : array of shape (Nf,)
        the frequencies
    K : float (optional)
        the log-transform constant, default 1e11
    i0 : int (optional)
        the reference antenna index, default 0
    '''
    Na,Nt,Nd,_,Ns = rays.shape
    Nf = freqs.shape[0]
    def index(i,j,k,s, N2 = Nt, N3 = Nd, N4 = Ns):
        '''Get the C-order flattened index'''
        return np.ndarray.astype(s + N4*(k + N3*(j + N2*i)), int)

    def index_inv(h, N2 = Nt, N3 = Nd, N4 = Ns):
        '''Invert flattened index to the indices'''
        h = np.ndarray.astype(h,float)
        s = np.mod(h, float(N4))
        h -= s
        h /= float(N4)
        k = np.mod(h, float(N3))
        h -= k
        h /= float(N3)
        j = np.mod(h, float(N2))
        h -= j
        h /= float(N2)
        i = h
        return np.ndarray.astype(i,int), np.ndarray.astype(j,int), np.ndarray.astype(k,int), np.ndarray.astype(s,int)

    c_mu,c_clock = covariance
    mu,clock,const = model

    #kernel_size should be large enough that covariance is basically zero for greater distances
    kernel_size = 0.
    
    for d in [[0,0,1],[0,1,0],[1,0,0]]:
        c0_ = c_mu(np.zeros([1,3]),np.array([d])*0)[0,0]
        c_ = c_mu(np.zeros([1,3]),np.array([d])*kernel_size)[0,0]
        while c_/c0_ > 0.1:
            kernel_size += 0.5
            c_ = c_mu(np.zeros([1,3]),np.array([d])*kernel_size)[0,0]
    #kernel_size=10.
    log.info("Kernel_size is : {} km".format(kernel_size))
    
    #Use kd-tree to only calculate points within a kernel_size distance
    
    kdt = cKDTree(np.array([rays[:,:,:,0,:].flatten(),
        rays[:,:,:,1,:].flatten(),
        rays[:,:,:,2,:].flatten()]).T)
    pairs = kdt.query_pairs(r=kernel_size,eps=0.,output_type='ndarray')
    i1,j1,k1,s1 = index_inv(pairs[:,0])
    i2,j2,k2,s2 = index_inv(pairs[:,1])
    
#    i1,j1,k1,s1 = np.unravel_index(pairs[:,0],[Na,Nt,Nd,Ns])
#    i2,j2,k2,s2 = np.unravel_index(pairs[:,1],[Na,Nt,Nd,Ns])
    

    print("Number of pairs with kernel_size : {}".format(pairs.shape[0]))

    #ds at each point
    ds1 = rays[i1,j1,k1,3,s1]
    ds1 -= rays[i1,j1,k1,3,s1-1]
    mask = s1 == 0
    ds1[mask] = np.mean(ds1[s1 > 0]) 

    ds2 = rays[i2,j2,k2,3,s2]
    ds2 -= rays[i2,j2,k2,3,s2-1]
    mask = s2 == 0
    ds2[mask] = np.mean(ds2[s2 > 0])
    
    dsds = ds1*ds2


    #the distance between all pairs within kernel_size
    dist = rays[i1,j1,k1,0:3,s1] - rays[i2,j2,k2,0:3,s2]
    print("test")
    C_mu = c_mu(dist,np.zeros([1,3]))[:,0]#len(pairs) x 1 -> len(pairs)
    C_mu *= dsds

#    dist *= dist
#    dist = np.sum(dist,axis=3)
#    np.sqrt(dist,out=dist)

    #ne along path
    ne = np.exp(mu)
    ne *= K
    tci.M = ne
    ne_rays = tci.interp(rays[:,:,:,0,:],rays[:,:,:,1,:],rays[:,:,:,2,:])
    ne_rays1 = ne_rays[i1,j1,k1,s1]
    ne_rays2 = ne_rays[i2,j2,k2,s2]
#    ne_rays1 = tci.interp(rays[i1,j1,k1,0,s1],rays[i1,j1,k1,1,s1],rays[i1,j1,k1,2,s1])
#    ne_rays2 = tci.interp(rays[i2,j2,k2,1,s2],rays[i2,j2,k2,1,s2],rays[i2,j2,k2,2,s2])

    #result only store half then rearrange later
    S = np.zeros([Na*Nt*Nd*Nf,Na*Nt*Nd*Nf],dtype=float)

    mask_02 = i1 == i0
    mask_10 = i2 == i0
    mask_00 = mask_10*mask_02
    
    #S = np.diag(CdCt.flatten())
    def inner_loop(i1,j1,k1,s1,i2,j2,k2,s2,l,lp,freqs,speedoflight,d_,ne_rays2,Na,Nt,Nd,Nf,Ns,i0,mask_02,mask_10,mask_00):
        #S = np.zeros([Na,Nt,Nd,Nf,Na,Nt,Nd,Nf],dtype=float)
        S = np.zeros([Na*Nt*Nd*Nf,Na*Nt*Nd*Nf],dtype=float)
        a_2 = (2*np.pi * freqs[lp])
        n_p2 = 1.2404e-2*freqs[lp]**2
        b_2 = a_2/(2*n_p2*speedoflight)

        n_rays2 = ne_rays2/(-n_p2)
        n_rays2 += 1.
        np.sqrt(n_rays2,out=n_rays2)
        dmu_n2 = ne_rays2/n_rays2

        #dmu1 cm dmu2 dsds

#            S[:,:,:,l,:,:,:,lp] += a_1 * a_2 / c_clock
        f1 = np.ones(len(i1),dtype=int)*l
        f2 = np.ones(len(i2),dtype=int)*lp
    
        q = d_*dmu_n2
        
        q[mask_02] -= d_[mask_02]*dmu_n2[mask_02]
        q[mask_10] -= d_[mask_10]*dmu_n2[mask_10]
        q[mask_00] += d_[mask_00]*dmu_n2[mask_00]
        q *= b_1*b_2

        #np.add.at(S, (i1,j1,k1,f1,i2,j2,k2,f2), q)
        #np.add.at(S, (i2,j2,k2,f2,i1,j1,k1,f1), q)

        h1h2 = np.ravel_multi_index((i1,j1,k1,f1,i2,j2,k2,f2),[Na,Nt,Nd,Nf,Na,Nt,Nd,Nf])
        h2h1 = np.ravel_multi_index((i2,j2,k2,f2,i1,j1,k1,f1),[Na,Nt,Nd,Nf,Na,Nt,Nd,Nf])

        S += np.bincount(h1h2, q, minlength=S.size).reshape(S.shape)
        S += np.bincount(h2h1, q, minlength=S.size).reshape(S.shape)

        #np.add.at(S,I1I2,q)
        #np.add.at(S,I2I1,q)
        return S
#    client = Client()
    from dask.threaded import get
    #pair wise component i<j means need to do twice for swapped frequencies
    dsk = {}
    
    S_sum = []
    for l in range(Nf):
        a_1 = (2*np.pi * freqs[l])
        n_p1 = 1.2404e-2*freqs[l]**2
        b_1 = a_1/(2*n_p1*speedoflight)
        
        #Na Nt Nd Ns at l
        n_rays1 = ne_rays1/(-n_p1)
        n_rays1 += 1.
        np.sqrt(n_rays1,out=n_rays1)
        dmu_n1 = ne_rays1/n_rays1

        d_ = dmu_n1 * C_mu
        dsk['d_{}'.format(l)] = d_
        
        for lp in range(Nf):
            dsk['S_{}_{}'.format(l,lp)] = (inner_loop,i1,j1,k1,s1,i2,j2,k2,s2,l,lp,freqs,speedoflight,'d_{}'.format(l),ne_rays2,Na,Nt,Nd,Nf,Ns,i0,mask_02,mask_10,mask_00)
            S_sum.append('S_{}_{}'.format(l,lp))
    def merge(*S):
        res = S[0]
        for i in range(1,len(S)):
            res += S[i]
        return res
    dsk['S'] = (merge ,*S_sum)
    from dask.threaded import get
    S = get(dsk,'S',num_workers=num_threads)
 
    S_ = np.zeros([Na,Nt,Nd,Nf,Na,Nt,Nd,Nf],dtype=float)
    for l in range(Nf):
        a_1 = (2*np.pi * freqs[l])
        for lp in range(Nf):
            a_2 = (2*np.pi * freqs[lp])
            S_[:,:,:,l,:,:,:,lp] += a_1 * a_2 * c_clock
    S += S_.reshape([Na*Nt*Nd*Nf,Na*Nt*Nd*Nf])
    S += np.diag(CdCt.flatten())
    return S

def update_direction(snr, dd,  m_n, covariance, tci, rays, freqs, CdCt, num_threads,K=1e11, i0 = 0):
    '''Compute the update direction Cm G^t snr
    model : tuple (mu, clock, const)
        mu is flattened model, clock is shape (Na, Nt), const is shape (Na,)
    covariance : tuple (Covariance, float)
        the Covariance object for mu, and float variance for clock
    tci : TriCubic
        interpolator for the region
    rays : array
        array of shape (Na, Nt, Nd, 4, Ns), the third dimension is x,y,z,s in that order
    freqs : array of shape (Nf,)
        the frequencies
    num_threads : int
        number of threads allowed to use
    K : float (optional)
        the log-transform constant, default 1e11
    i0 : int (optional)
        the reference antenna index, default 0
    '''
    Na,Nt,Nd,_,Ns = rays.shape
    Nf = freqs.shape[0]

    def index(i,j,k,s, N2 = Nt, N3 = Nd, N4 = Ns):
        '''Get the C-order flattened index'''
        return np.ndarray.astype(s + N4*(k + N3*(j + N2*i)), int)

    def index_inv(h, N2 = Nt, N3 = Nd, N4 = Ns):
        '''Invert flattened index to the indices'''
        h = np.ndarray.astype(h,float)
        s = np.mod(h, float(N4))
        h -= s
        h /= float(N4)
        k = np.mod(h, float(N3))
        h -= k
        h /= float(N3)
        j = np.mod(h, float(N2))
        h -= j
        h /= float(N2)
        i = h
        return np.ndarray.astype(i,int), np.ndarray.astype(j,int), np.ndarray.astype(k,int), np.ndarray.astype(s,int)


    mu,clock,const = m_n
    c_mu, c_clock = covariance

    ne = np.exp(mu)
    ne *= K
    tci.M = ne
    ne_rays = tci.interp(rays[:,:,:,0,:],rays[:,:,:,1,:],rays[:,:,:,2,:])

    X,Y,Z = tci.get_model_coordinates()#nx*ny*nz
    #kernel_size should be large enough that covariance is basically zero for greater distances
    kernel_size = 0.
    
    for d in [[0,0,1],[0,1,0],[1,0,0]]:
        c0_ = c_mu(np.zeros([1,3]),np.array([d])*0)[0,0]
        c_ = c_mu(np.zeros([1,3]),np.array([d])*kernel_size)[0,0]
        while c_/c0_ > 0.5:
            kernel_size += 0.5
            c_ = c_mu(np.zeros([1,3]),np.array([d])*kernel_size)[0,0]
    log.info("Kernel_size is : {} km".format(kernel_size))

    kdt1 = cKDTree(np.array([rays[:,:,:,0,:].flatten(),
        rays[:,:,:,1,:].flatten(),
        rays[:,:,:,2,:].flatten()]).T)
    kdt2 = cKDTree(np.array([X,Y,Z]).T)

    pairs_ = kdt1.query_ball_tree(kdt2,r=kernel_size,eps=0.)
    pairs = []
    for i,pair in enumerate(pairs_):
        for j in pair:
            pairs.append([i,j])
    pairs = np.array(pairs)
    num_pairs = pairs.shape[0]
    print("Number of pairs within kernel_size : {}".format(num_pairs))
    
    i,j,k,s = index_inv(pairs[:,0],N4=Ns)

    ds = rays[i,j,k,3,s]
    ds -= rays[i,j,k,3,s-1]
    mask = s == 0
    ds[mask] = np.mean(ds[s > 0])

    dist = np.array([rays[i,j,k,0,s] - X[pairs[:,1]],
        rays[i,j,k,1,s] - Y[pairs[:,1]],
        rays[i,j,k,2,s] - Z[pairs[:,1]]]).T

    C_mu = c_mu(dist,np.zeros([1,3]))[:,0]*ds
    
    

    def inner_loop(l,freq,speedoflight,to_pair,ne_rays,ne_rays0,snr,C_mu,out_shape):

        phi = np.zeros(out_shape,dtype=float)
        a_ = (2*np.pi * freq)
        n_p = 1.2404e-2*freq**2
        b_ = a_/(2*n_p*speedoflight)
                
        #Na Nt Nd Ns at l
        n_rays = ne_rays/(-n_p)
        n_rays += 1.
        np.sqrt(n_rays,out=n_rays)
        dmu_n = ne_rays/n_rays
        dmu_n *= b_

        #Na Nt Nd Ns at l
        n_rays0 = ne_rays0/(-n_p)
        n_rays0 += 1.
        np.sqrt(n_rays0,out=n_rays0)
        dmu_n0 = ne_rays0/n_rays0
        dmu_n0 *= b_
                
        q = snr*C_mu*(dmu_n - dmu_n0)
        phi[to_pair] += q#nx*ny*nz x 1
        return phi

    dsk = {}
    phi_sum = []
    for l in range(Nf):
        indices = range(l,num_pairs,Nf)
        i_ = i[indices]
        j_ = j[indices]
        k_ = k[indices]
        s_ = s[indices]
        snr_ = snr[i_,j_,k_,l]
        ne_rays_ = ne_rays[i_,j_,k_,s_]
        ne_rays_0 = ne_rays[i0,j_,k_,s_]
        C_mu_ = C_mu[indices]
        output_shape = X.shape
        dsk["CmG_{}".format(l)] = (inner_loop,l,freqs[l],speedoflight,
                pairs[indices,1],ne_rays_,ne_rays_0,snr_,C_mu_,output_shape)
        phi_sum.append("CmG_{}".format(l))
    def merge(*phi):
        res = phi[0]
        for i in range(1,len(phi)):
            res += phi[i]
        return res
    dsk['phi'] = (merge ,*phi_sum)
    from dask.threaded import get
    phi = get(dsk,'phi',num_workers=num_threads)

    phi_clock = np.zeros([Na,Nt],dtype=float)

    for i in range(Na):
        for j in range(Nt):
            for k in range(Nd):
                for l in range(Nf):
                    a_ = (2*np.pi * freqs[l])
                    phi_clock[i,j] += snr[i,j,k,l] * a_ * c_clock

    #const term with no a priori G=1 is 
    dconst = np.sum(np.sum(np.sum(dd,axis=3),axis=2),axis=1)

    return phi.reshape(mu.shape), phi_clock, dconst
            

def linesearch(dd,g,phi,m_n, covariance, tci, rays, freqs, CdCt, num_threads,K=1e11, i0 = 0):
    """Perform line search for step size."""
    eps = 1e-4
    mu,clock,const = m_n
    dmu, dclock, dconst = phi

    Gdm = (forward_equation((mu+eps*dmu,clock+eps*dclock,const+eps*dconst), 
        tci, rays, freqs,K=K, i0 = i0) - g)/eps

    ep_0 = np.sum(Gdm*dd/CdCt)/np.sum(Gdm**2/CdCt)
    print("ep_0: {}".format(ep_0))

    return ep_0

def calc_I_11(model, covariance, tci, rays, freqs, num_threads, K=1e11, i0 = 0):
    '''Calculates the phase from the given model point.
    model : tuple (mu, clock, const)
        mu is flattened model, clock is shape (Na, Nt), const is shape (Na,)
    covariance : tuple (Covariance, float)
        the Covariance object for mu, and float variance for clock
    tci : TriCubic
        interpolator for the region
    rays : array
        array of shape (Na, Nt, Nd, 4, Ns), the third dimension is x,y,z,s in that order
    freqs : array of shape (Nf,)
        the frequencies
    num_threads: int or None
        the number of threads allowed for calculating I_11, None is all.
    K : float (optional)
        the log-transform constant, default 1e11
    i0 : int (optional)
        the reference antenna index, default 0
    '''
    Na,Nt,Nd,_,Ns = rays.shape
    Nf = freqs.shape[0]
    def index(i,j,k,s, N2 = Nt, N3 = Nd, N4 = Ns):
        '''Get the C-order flattened index'''
        return np.ndarray.astype(s + N4*(k + N3*(j + N2*i)), int)

    def index_inv(h, N2 = Nt, N3 = Nd, N4 = Ns):
        '''Invert flattened index to the indices'''
        h = np.ndarray.astype(h,float)
        s = np.mod(h, float(N4))
        h -= s
        h /= float(N4)
        k = np.mod(h, float(N3))
        h -= k
        h /= float(N3)
        j = np.mod(h, float(N2))
        h -= j
        h /= float(N2)
        i = h
        return np.ndarray.astype(i,int), np.ndarray.astype(j,int), np.ndarray.astype(k,int), np.ndarray.astype(s,int)

    c_mu,c_clock = covariance
    mu,clock,const = model

    #kernel_size should be large enough that covariance is basically zero for greater distances
    kernel_size = 0.
    
    for d in [[0,0,1],[0,1,0],[1,0,0]]:
        c0_ = c_mu(np.zeros([1,3]),np.array([d])*0)[0,0]
        c_ = c_mu(np.zeros([1,3]),np.array([d])*kernel_size)[0,0]
        while c_/c0_ > 0.1:
            kernel_size += 0.5
            c_ = c_mu(np.zeros([1,3]),np.array([d])*kernel_size)[0,0]
    #kernel_size=10.
    log.info("Kernel_size is : {} km".format(kernel_size))
    
    #Use kd-tree to only calculate points within a kernel_size distance
    
    kdt = cKDTree(np.array([rays[:,:,:,0,:].flatten(),
        rays[:,:,:,1,:].flatten(),
        rays[:,:,:,2,:].flatten()]).T)
    pairs = kdt.query_pairs(r=kernel_size,eps=0.,output_type='ndarray')
    i1,j1,k1,s1 = index_inv(pairs[:,0])
    i2,j2,k2,s2 = index_inv(pairs[:,1])
    
#    i1,j1,k1,s1 = np.unravel_index(pairs[:,0],[Na,Nt,Nd,Ns])
#    i2,j2,k2,s2 = np.unravel_index(pairs[:,1],[Na,Nt,Nd,Ns])
    

    log.info("Number of pairs within kernel_size : {}".format(pairs.shape[0]))

    #ds at each point
    ds1 = rays[i1,j1,k1,3,s1]
    ds1 -= rays[i1,j1,k1,3,s1-1]
    mask = s1 == 0
    ds1[mask] = np.mean(ds1[s1 > 0]) 

    ds2 = rays[i2,j2,k2,3,s2]
    ds2 -= rays[i2,j2,k2,3,s2-1]
    mask = s2 == 0
    ds2[mask] = np.mean(ds2[s2 > 0])
    
    dsds = ds1*ds2


    #the distance between all pairs within kernel_size
    dist = rays[i1,j1,k1,0:3,s1] - rays[i2,j2,k2,0:3,s2]

    C_mu = c_mu(dist,np.zeros([1,3]))[:,0]#len(pairs) x 1 -> len(pairs)
    C_mu *= dsds

#    dist *= dist
#    dist = np.sum(dist,axis=3)
#    np.sqrt(dist,out=dist)

    #ne along path
    ne = np.exp(mu)
    ne *= K
    tci.M = ne
    ne_rays = tci.interp(rays[:,:,:,0,:],rays[:,:,:,1,:],rays[:,:,:,2,:])
    ne_rays1 = ne_rays[i1,j1,k1,s1]
    ne_rays2 = ne_rays[i2,j2,k2,s2]
#    ne_rays1 = tci.interp(rays[i1,j1,k1,0,s1],rays[i1,j1,k1,1,s1],rays[i1,j1,k1,2,s1])
#    ne_rays2 = tci.interp(rays[i2,j2,k2,1,s2],rays[i2,j2,k2,1,s2],rays[i2,j2,k2,2,s2])

    #result only store half then rearrange later
    I11 = np.zeros([Na*Nt*Nd*Nf,Na*Nt*Nd*Nf],dtype=float)

    mask_02 = i1 == i0
    mask_10 = i2 == i0
    mask_00 = mask_10*mask_02
    
    #S = np.diag(CdCt.flatten())
    def inner_loop(i1,j1,k1,s1,i2,j2,k2,s2,l,lp,freqs,speedoflight,d_,ne_rays2,Na,Nt,Nd,Nf,Ns,i0,mask_02,mask_10,mask_00):
        #S = np.zeros([Na,Nt,Nd,Nf,Na,Nt,Nd,Nf],dtype=float)
        I11 = np.zeros([Na*Nt*Nd*Nf,Na*Nt*Nd*Nf],dtype=float)
        a_2 = (2*np.pi * freqs[lp])
        n_p2 = 1.2404e-2*freqs[lp]**2
        b_2 = a_2/(2*n_p2*speedoflight)

        n_rays2 = ne_rays2/(-n_p2)
        n_rays2 += 1.
        np.sqrt(n_rays2,out=n_rays2)
        dmu_n2 = ne_rays2/n_rays2

        #dmu1 cm dmu2 dsds

#            S[:,:,:,l,:,:,:,lp] += a_1 * a_2 / c_clock
        f1 = np.ones(len(i1),dtype=int)*l
        f2 = np.ones(len(i2),dtype=int)*lp
    
        q = d_*dmu_n2     
        q[mask_02] -= d_[mask_02]*dmu_n2[mask_02]
        q[mask_10] -= d_[mask_10]*dmu_n2[mask_10]
        q[mask_00] += d_[mask_00]*dmu_n2[mask_00]
        q *= b_1*b_2

        h1h2 = np.ravel_multi_index((i1,j1,k1,f1,i2,j2,k2,f2),[Na,Nt,Nd,Nf,Na,Nt,Nd,Nf])
        h2h1 = np.ravel_multi_index((i2,j2,k2,f2,i1,j1,k1,f1),[Na,Nt,Nd,Nf,Na,Nt,Nd,Nf])

        I11 += np.bincount(h1h2, q, minlength=I11.size).reshape(I11.shape)
        I11 += np.bincount(h2h1, q, minlength=I11.size).reshape(I11.shape)

        return I11
#    client = Client()
    from dask.threaded import get
    #pair wise component i<j means need to do twice for swapped frequencies
    dsk = {}
    
    I11_sum = []
    for l in range(Nf):
        a_1 = (2*np.pi * freqs[l])
        n_p1 = 1.2404e-2*freqs[l]**2
        b_1 = a_1/(2*n_p1*speedoflight)
        
        #Na Nt Nd Ns at l
        n_rays1 = ne_rays1/(-n_p1)
        n_rays1 += 1.
        np.sqrt(n_rays1,out=n_rays1)
        dmu_n1 = ne_rays1/n_rays1

        d_ = dmu_n1 * C_mu
        dsk['d_{}'.format(l)] = d_
        
        for lp in range(Nf):
            dsk['I11_{}_{}'.format(l,lp)] = (inner_loop,i1,j1,k1,s1,i2,j2,k2,s2,l,lp,freqs,speedoflight,'d_{}'.format(l),ne_rays2,Na,Nt,Nd,Nf,Ns,i0,mask_02,mask_10,mask_00)
            I11_sum.append('I11_{}_{}'.format(l,lp))
    def merge(*I11):
        res = I11[0]
        for i in range(1,len(I11)):
            res += I11[i]
        return res
    dsk['I11'] = (merge ,*I11_sum)
    from dask.threaded import get
    I11 = get(dsk,'I11',num_workers=num_threads)

    return I11



#@profile
def iterative_newton_step(m_n,m_prior,rays,tci,covariance,CdCt, dobs, freqs,num_threads,K=1e11,i0=0):
    '''Do the IRLS step about the current model point.
    Return the next model point.

    g_ijk = (2 pi nu / c) [int_ijk (1 - n)ds - int_i0jk (1-n)ds] + (2 pi nu) tau_ij + c_i

    prioris on parameters:
    n = sqrt(1-ne/np), ne = K exp(m), m ~ N(m_p,C_m)
    t_ij ~ N(tau_p, (5 ns)^2)
    c_i ~ U(-inf,inf)
    
    m_n+1 = m_p + epsilon_n C_m G^t T ( ( d_obs - g(m_n)) - G (m_p - m_n) )

    params:
    m_n : tuple (log(ne/K), tau, c)
        the current model point
    m_prior : tuple (log(ne_prior/K), tau_ prior, c_prior)
        the prior means of model
    rays: array of shape (Na, Nt, Nd, 4, Ns)
        the ray trajectories
    tci : TriCubic
        the tricubic interpolator (here only tri linear)
    covariance : tuple(Covariance, Covariance, Covariance)
        the covariances of model
    CdCt : array of shape (Na, Nt, Nd, Nf)
        the measurement and modeling uncertainties diagonal
    dobs : array of shape (Na,Nt,Nd,Nf)
        the phases in radians measured in each ray at Nf frequencies
    K : float
        the constant for log transform of electron density
    i0 : int
        the reference antenna
    freqs : array of shape (Nf,)
        the frequencies of observation i Hz
    '''

    Na,Nt,Nd,_,Ns = rays.shape
    print("Rays shape : {}".format(rays.shape))
    _,_,_,Nf = dobs.shape

    Nray = Na*Nt*Nd*Nf

    mu_n,clock_n,const_n = m_n
    mu_prior,clock_prior,const_prior = m_prior

    c_mu,c_clock = covariance
    c_const = 0.1

    G_const = 1.
    G_clock = 2.*np.pi * freqs

    ###
    # dd
    ###

    print("dd = d_obs - g(m_n)")
    g_phase = forward_equation(m_n, tci, rays, freqs, K=K, i0 = i0)
    dd = dobs - g_phase

    ###
    # Cd^-1 dd (assume invertible, non zero uncertainty)
    ###

    # Na Nt Nd Nf
    wdd = dd/CdCt

    ###
    # k2
    ###

    # Na Nt
    k2_clock = np.einsum("m,n,l,ijkl->mn",np.ones(Na),np.ones(Nt),G_clock,wdd,optimize=True)
    k2_clock += (clock_prior - clock_n)/c_clock

    # Na
    k2_const = np.einsum("m,ijkl->m",np.ones(Na),wdd,optimize=True)
    k2_const += (const_prior - const_n)/c_const

    ###
    # r_mu = - G_mu dm_mu
    ###

    print("r_mu_n = G_mu(mu_p-mu_n)")
    r_mu = prior_penalty_mu(m_n, m_prior, tci, rays, freqs, K=K, i0 = i0)

    ###
    # signal_mu
    ###

    print("signal_mu = dd-r_mu")
    signal_mu = dd - r_mu
    
    ###
    # I_11 = G_mu C_mu G_mu^t
    ###
    
    I_11 = calc_I_11(m_n, covariance, tci, rays, freqs, num_threads, K=K, i0 = i0).reshape([Nray,Nray])

    ###
    # C_d + I_11
    ###
    print("S_11 = C_d + I_11")

    S_11 = np.diag(CdCt.flatten()) + I_11

    ###
    # V_11 = I_11 * (Cd + I_11)^-1
    ###
    print("V_11 =I_11 * (Cd + I_11)^-1")

    V_11 = solve_sym_posdef(S_11,I_11).T   

    ###
    # VAU = G_2^t * (C_d + I_11)^-1 G_2
    ###
    print("VAU = G_2^t * (C_d + I_11)^-1 G_2")

    # Nrays, Na*Nt
    G_clock_ = np.einsum("m,n,l,i,j,k->ijklmn",np.ones(Na),np.ones(Nt),G_clock, np.ones(Na),np.ones(Nt), np.ones(Nd), optimize=True).reshape([Na*Nt*Nd*Nf,Na*Nt])
    # Nrays, Na
    G_const_ = np.einsum("m,l,i,j,k->ijklm",np.ones(Na),np.ones(Nf), np.ones(Na),np.ones(Nt), np.ones(Nd), optimize=True).reshape([Na*Nt*Nd*Nf,Na])
    VAU_11 = cholesky_inner(S_11,G_clock_)
    VAU_22 = cholesky_inner(S_11,G_const_)
    VAU_12 = G_clock_.T.dot(solve_sym_posdef(S_11,G_const_))
    VAU_21 = VAU_12.T

    ###
    # CVAU = C^-1 - VA^-1U
    ###
    print('CVAU = C^-1 - VA^-1U')

    CVAU = np.zeros([Na*Nt + Na,Na*Nt+Na])
    
    CVAU[:Na*Nt,:Na*Nt] = VAU_11 +  np.eye(Na*Nt)/c_clock
    CVAU[:Na*Nt,Na*Nt:] = VAU_12
    CVAU[Na*Nt:,:Na*Nt] = VAU_21
    CVAU[:Na*Nt,:Na*Nt] = VAU_22 +  np.eye(Na*Nt)/c_const#no const prior

    ###
    # R_2 = -(G_2^t Cd^-1 V_11 r_mu + k_2)
    ###
    print("R_2 = -(G_2^t Cd^-1 V_11 r_mu + k_2)")
    R_2_clock  = np.einsum("m,n,l,i,j,k,ijkl,ijklqwer,qwer->mn",
            np.ones(Na), np.ones(Nt),G_clock,np.ones(Na),np.ones(Nt),
            np.ones(Nd),1./CdCt,V_11.reshape([Na,Nt,Nd,Nf,Na,Nt,Nd,Nf]), r_mu,optimize=True)
    R_2_clock += k2_clock
    R_2_clock *= -1.

    R_2_const  = np.einsum("m,l,i,j,k,ijkl,ijklqwer,qwer->m",
            np.ones(Na), np.ones(Nf),np.ones(Na),np.ones(Nt),
            np.ones(Nd),1./CdCt,V_11.reshape([Na,Nt,Nd,Nf,Na,Nt,Nd,Nf]), r_mu,optimize=True)
    R_2_const += k2_const
    R_2_const *= -1.
    R_2 = np.concatenate([R_2_clock.flatten(),R_2_const.flatten()])

    ###
    # dm2 = CVAU^-1 R_2
    ###
    print("dm2 = CVAU^-1 R_2")
    
    dm2 = solve_sym_posdef(CVAU,R_2)
    print(dm2)

    ###
    # R_1 = r_mu - G2 dm2
    ###

    G2 = np.concatenate([np.einsum("m,n,l,i,j,k->mnijkl",
            np.ones(Na), np.ones(Nt),G_clock, np.ones(Na), np.ones(Nt),
            np.ones(Nd),optimize=True).reshape([Na*Nt,Na*Nt*Nd*Nf]), 
            np.einsum("m,l,i,j,k->mijkl",
            np.ones(Na), np.ones(Nf), np.ones(Na), np.ones(Nt),
            np.ones(Nd),optimize=True).reshape([Na,Na*Nt*Nd*Nf])])

    R_1 = -np.einsum("mr,m->r",G2,dm2)
    R_1 += r_mu.flatten()
    
    print(R_1)
    



    




    print("S = Cd + GCmG^t")

    S = signal_covariance(m_n, covariance, tci, rays, freqs, CdCt, num_threads,K=K, i0 = i0)

    print("snr = S^-1 signal")
    snr = solve_sym_posdef(S,signal.flatten()).reshape([Na,Nt,Nd,Nf])

    print("phi = Cm.G.snr")
    phi_n = update_direction(snr,dd,m_n, covariance, tci, rays, freqs, CdCt, num_threads,K=K, i0 = i0)

    print("linesearch")
    ep_n = linesearch(dd,g_phase,phi_n,m_n, covariance, tci, rays, freqs, CdCt, num_threads,K=K, i0 = i0)
    dmu_n = - ep_n*mu_n + ep_n*mu_prior + ep_n*phi_n[0]
    dclock_n = -ep_n*clock_n + ep_n*clock_prior + ep_n*phi_n[1]
    dconst_n = ep_n*phi_n[2]

    print('diagnostics')

    import pylab as plt
    f,(ax1,ax2,ax3) = plt.subplots(3,1)
    ax1.plot(dmu_n.flatten())
    ax2.plot(dclock_n.flatten())
    ax3.plot(dconst_n.flatten())
    plt.show()
    mu_n += dmu_n
    clock_n += dclock_n
    const_n += dconst_n

    return mu_n,clock_n,const_n
    

def iterative_newton_solve(model_0,model_prior,rays,freqs, covariance,CdCt, dobs, num_threads, diagnostic_folder, K=1e11, i0=0):
    '''solve using iterative iterative newton method.
    ne_0 is a Solution object,
    ne_prior is a Solution object'''
    #convergence conditions
    factr = 1e7
    pgtol = 1e-2
    eps = np.finfo(float).eps
    max_iter = 20
    #ne_0 and ne_p might share same address
    ne_0, clock_0, const_0 = model_0
    ne_p, clock_p, const_p = model_prior
    clock_0 *= 0
    const_0 *= 0
    clock_p *= 0
    const_p *= 0

    ne_p.save(os.path.join(diagnostic_folder,"ne_prior.hdf5"))
    
    mu_0 = ne_0.M/K
    np.log(mu_0,out=mu_0)


    
    mu_p = ne_p.M/K
    np.log(mu_p,out=mu_p)

    solution = ne_0.copy() 
    model_n = (mu_0,clock_0.copy(),const_0.copy())
    model_prior = (mu_p,clock_p,const_p)

    g = forward_equation(model_n, solution, rays, freqs, K=K, i0 = i0)
    S_n = neg_log_like(g,dobs,CdCt,covariance,model_n,model_prior,solution,full=False)
    log.info("Initial negative log likelihood : {}".format(S_n))
    #initial conditions
    model_np1 = model_n

    S_np1 = S_n
    iter = 0
    while ((S_n - S_np1)/max(np.abs(S_n),np.abs(S_np1),1.) > factr*eps and np.max(np.abs(model_n[0] - model_np1[0])) > pgtol and iter < max_iter) or iter < 5:
        model_n = model_np1
        S_n = S_np1
        #step
        model_np1 = iterative_newton_step(model_n,model_prior,rays,solution,covariance,CdCt, dobs,freqs,num_threads,K=K,i0=i0)
        solution.M = model_np1[0]
        solution.save(os.path.join(diagnostic_folder,"solution_iter{}.hdf5".format(iter)))
        #L2 neg log like
        g = forward_equation(model_np1, solution, rays, freqs, K=K, i0 = i0)
        S_np1 = neg_log_like(g,dobs,CdCt,covariance,model_np1,model_prior,solution,full=False)
        print("Iter {} : negative log likelihood : {}".format(iter,S_n))
        iter += 1
    
    if ((S_n - S_np1)/max(np.abs(S_n),np.abs(S_np1),1.) > factr*eps and np.max(np.abs(model_n - model_np1)) > pgtol):
        convergence = False
        log.info("Iterative Newton solve did not converge")
    else:
        convergence = True
        log.info("Iterative Newton solve converged")
    assert convergence,"Did not converge"
    
    ne_sol = np.exp(model_np1[0])
    ne_sol *= K
    solution.M = ne_sol
    return (solution, model_np1[1], model_np1[2])

