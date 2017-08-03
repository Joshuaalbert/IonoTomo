
# coding: utf-8

# In[1]:

import numpy as np
from scipy.integrate import simps
import dask.array as da
from dask import delayed
from dask.multiprocessing import get

def do_forward_equation(rays,K_ne,m_tci):
    N1,N2, _ , Ns = rays.shape
    #nevec = np.zeros(Ns,dtype=np.double)
    g = np.zeros([N1,N2],dtype=np.double)
    j = 0
    while j < N1:
        k = 0
        while k < N2:
            x = rays[j,k,0,:]
            y = rays[j,k,1,:]
            z = rays[j,k,2,:]
            s = rays[j,k,3,:]
            nevec = m_tci.interp(x,y,z)
#            nevec *= 0
#            idx = 0
#            while idx < Ns:
#                nevec[idx] += m_tci.interp(x[idx],y[idx],z[idx])
#                idx += 1
#           
            #print(nevec) 
            np.exp(nevec,out=nevec)
            nevec *= K_ne/1e13
            #if np.any(np.isnan(nevec)):
           
            g[j,k] += simps(nevec,s)
            k += 1
        j += 1
    return g
                

def forward_equation(rays,K_ne,m_tci,i0):
    '''For each ray do the forward equation using ref antenna i0'''
    Na,Nt,Nd, _, Ns = rays.shape
    if Na < Nd:
        #do over antennas
        tec = np.stack([do_forward_equation(rays[i,:,:,:,:],K_ne,m_tci) for i in range(Na)],axis=0)
    else:
        #do over directions
        tec = np.stack([do_forward_equation(rays[:,:,k,:,:],K_ne,m_tci) for k in range(Nd)],axis=2)
    dtec = tec - tec[i0,:,:]
    return dtec

def forward_equation_dask(rays,K_ne,m_tci,i0):
    '''For each ray do the forward equation using ref antenna i0'''
    Na,Nt,Nd, _, Ns = rays.shape
    if Na < Nd:
        #do over antennas
        tec = da.stack([da.from_delayed(delayed(do_forward_equation)(rays[i,:,:,:,:],K_ne,m_tci),(Nt,Nd),dtype=np.double) for i in range(Na)],axis=0)
    else:
        #do over directions
        tec = da.stack([da.from_delayed(delayed(do_forward_equation)(rays[:,:,k,:,:],K_ne,m_tci),(Na,Nt),dtype=np.double) for k in range(Nd)],axis=2)
    dtec = tec - tec[i0,:,:]
    return dtec.compute(get=get)

