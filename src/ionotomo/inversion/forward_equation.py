
# coding: utf-8

# In[1]:

import numpy as np
from scipy.integrate import simps
import dask.array as da
from dask import delayed
from dask.multiprocessing import get
from ionotomo.geometry.ray_dirac import get_ray_dirac
TECU=1e13
def do_forward_equation(rays,ne_tci):
    #ray_dirac,midpoints = get_ray_dirac(rays,ne_tci)
    #ne = ne_tci.interp(midpoints[0,...],midpoints[1,...],midpoints[2,...])
    #g_mod = np.einsum("ijklm,klm->ij",ray_dirac,ne_tci.M)
    N1,N2, _ , Ns = rays.shape
    g = np.zeros([N1,N2],dtype=np.double)
    j = 0
    while j < N1:
        k = 0
        while k < N2:
            x = rays[j,k,0,:]
            y = rays[j,k,1,:]
            z = rays[j,k,2,:]
            s = rays[j,k,3,:]
            nevec = ne_tci.interp(x,y,z)           
            g[j,k] += simps(nevec,s)
            k += 1
        j += 1
#    print(g,g_mod)
#    print(g-g_mod)
    return g
                

def forward_equation(rays,K_ne,m_tci,i0):
    '''For each ray do the forward equation using ref antenna i0'''
    

    Na,Nt,Nd, _, Ns = rays.shape
    ne_tci = m_tci.copy()
    np.exp(ne_tci.M,out=ne_tci.M)
    ne_tci.M *= K_ne/TECU
    if Na < Nd:
        #do over antennas
        tec = np.stack([do_forward_equation(rays[i,:,:,:,:],ne_tci) for i in range(Na)],axis=0)
    else:
        #do over directions
        tec = np.stack([do_forward_equation(rays[:,:,k,:,:],ne_tci) for k in range(Nd)],axis=2)
    dtec = tec - tec[i0,:,:]
    return dtec

def forward_equation_dask(rays,K_ne,m_tci,i0):
    '''For each ray do the forward equation using ref antenna i0'''
    Na,Nt,Nd, _, Ns = rays.shape
    ne_tci = m_tci.copy()
    np.exp(ne_tci.M,out=ne_tci.M)
    ne_tci.M *= K_ne/TECU

    if Na < Nd:
        #do over antennas
        tec = da.stack([da.from_delayed(delayed(do_forward_equation)(rays[i,:,:,:,:],ne_tci),(Nt,Nd),dtype=np.double) for i in range(Na)],axis=0)
    else:
        #do over directions
        tec = da.stack([da.from_delayed(delayed(do_forward_equation)(rays[:,:,k,:,:],ne_tci),(Na,Nt),dtype=np.double) for k in range(Nd)],axis=2)
    dtec = tec - tec[i0,:,:]
    return dtec.compute(get=get)

