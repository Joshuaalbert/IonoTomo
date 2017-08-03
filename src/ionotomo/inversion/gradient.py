
# coding: utf-8

# In[1]:

'''The gradient for steepest direction, i.e. <Cm, d/dm(-log(posterior))> 
is equal to Adjoint(G).(g(m) - d_obs) + (m - m_prior) = Cm.G^t.Cd^-1 .( g(m) - d_obs ) + (m - m_prior)'''

from ionotomo.geometry.tri_cubic import bisection 
import numpy as np
from scipy.integrate import simps
import dask.array as da
from dask import delayed
from dask.multiprocessing import get
from ionotomo.ionosphere.covariance import Covariance

def do_adjoint(rays, dd, K_ne, m_tci, sigma_m, Nkernel, size_cell, i0):
    #print("Doing gradient")
    L_m = Nkernel*size_cell
    #if antennas parallelization Nt,Nd
    #if directions parallelization Na,Nd
    N1,N2,_,Ns = rays.shape
    m_shape = [N1,N2,m_tci.nx,m_tci.ny,m_tci.nz]
    grad = np.zeros([m_tci.nx,m_tci.ny,m_tci.nz],dtype=np.double)
    
    mask = np.zeros(m_shape, dtype=np.bool)
    idx_min = np.ones(m_shape,dtype=np.int64)*Ns
    idx_max = np.ones(m_shape,dtype=np.int64)*-1
    nevec = np.zeros([N1,N2,Ns],dtype=np.double)
    #go through the mask
#     X,Y,Z = np.meshgrid(np.arange(m_tci.xvec.size),
#                         np.arange(m_tci.yvec.size),
#                         np.arange(m_tci.zvec.size),indexing='ij')
    j = 0
    while j < N1:
        k = 0
        while k < N2:
            x_ray = rays[j,k,0,:]
            y_ray = rays[j,k,1,:]
            z_ray = rays[j,k,2,:]
            s_ray = rays[j,k,3,:]
            nevec[j,k,:] = K_ne*np.exp(m_tci.interp(x_ray,y_ray,z_ray))/1e13

            idx = 0
            while idx < Ns:
                #nevec[j,k,idx] = K_ne*np.exp(m_tci.interp(x_ray[idx],y_ray[idx],z_ray[idx]))/1e13
                xi,yi,zi = bisection(m_tci.xvec,x_ray[idx]),bisection(m_tci.yvec,y_ray[idx]),bisection(m_tci.zvec,z_ray[idx])
                local_mask = (j,k,slice(max(0,xi - Nkernel), min(m_tci.nx - 1, xi + Nkernel + 1)),
                            slice(max(0,yi - Nkernel) , min(m_tci.ny - 1,yi + Nkernel + 1)),
                            slice(max(0, zi - Nkernel), min(m_tci.nz - 1, zi + Nkernel + 1)))
                mask[local_mask] = True
                shape = mask[local_mask].shape
                idx_max[local_mask] = np.max(np.stack([idx_max[local_mask],
                                                     np.ones(shape,dtype=np.int64)*idx],axis=-1),axis=-1)
                #print(idx_max[local_mask])
                idx_min[local_mask] = np.min(np.stack([idx_min[local_mask],
                                                     np.ones(shape,dtype=np.int64)*idx],axis=-1),axis=-1)
                idx += 1   
            k += 1
        j += 1
    
    sum_mask = np.sum(np.sum(mask,axis=0),axis=0)
    xi = 0
    while xi < m_tci.nx:
        yi = 0
        while yi < m_tci.ny:
            zi = 0
            while zi < m_tci.nz:
                if not sum_mask[xi,yi,zi]:
                    zi += 1
                    continue
                x,y,z = m_tci.xvec[xi],m_tci.yvec[yi],m_tci.zvec[zi]
                j = 0
                while j < N2:
                    i = 0
                    while i < N1:
                        x_ray = rays[i,j,0,:]
                        y_ray = rays[i,j,1,:]
                        z_ray = rays[i,j,2,:]
                        s_ray = rays[i,j,3,:]
                        ne = nevec[i,j,:]
                        if mask[i,j,xi,yi,zi]:
                            segment_mask = (slice(idx_min[i,j,xi,yi,zi],idx_max[i,j,xi,yi,zi]+1),)
                            dx = x - x_ray[segment_mask]
                            dy = y - y_ray[segment_mask]
                            dz = z - z_ray[segment_mask]
                            Cm = dx**2
                            dy *= dy
                            dz *= dz
                            Cm += dy
                            Cm += dz
                            #np.sqrt(Cm,out=Cm)
                            Cm /= -2.*L_m**2
                            np.exp(Cm,out=Cm)
                            Cm *= sigma_m**2
                            Cm *= ne[segment_mask]
                            comp = simps(Cm*dd[i,j],s_ray[segment_mask])
                            grad[xi,yi,zi] += comp
#                            if i == i0:
#                                grad[xi,yi,zi] -= N1*comp
                            
                        i += 1
                    j += 1
                zi += 1
            yi += 1
        xi += 1
    grad[:,:,:] -= grad[i0,:,:]
    return grad
     
          
def compute_adjoint_dask(rays, g, dobs, i0, K_ne, m_tci, m_prior, CdCt, sigma_m, Nkernel, size_cell):
    L_m = Nkernel*size_cell
#     #i not eq i0 mask
#     mask = np.ones(rays.shape[0],dtype=np.bool)
#     mask[i0] = False
#     rays = rays[mask,:,:,:,:]
#     g = g[mask,:,:]
#     dobs = dobs[mask,:,:]
#     CdCt = CdCt[mask,:,:]
    #residuals
    #g.shape, dobs.shape [Na,Nt,Nd]
    dd = g - dobs
    #weighted residuals
    #Cd.shape [Na,Nt,Nd] i.e. diagonal
    #CdCt^-1 = 1./CdCt
    dd /= (CdCt + 1e-15)
    #get ray info
    Na, Nt, Nd, _ ,Ns = rays.shape
    #parallelize over directions
    gradient = da.sum(da.stack([da.from_delayed(delayed(do_adjoint)(rays[:,:,d,:,:], dd[:,:,d], K_ne, m_tci, 
                                      sigma_m, Nkernel, size_cell, i0),(m_tci.nx,m_tci.ny,m_tci.nz),dtype=np.double) for d in range(Nd)],axis=-1),axis=-1)
    gradient = gradient.compute(get=get)
    gradient += m_tci.M
    gradient -= m_prior
    
    return gradient

def compute_adjoint(rays, g, dobs, i0, K_ne, m_tci, m_prior, CdCt, sigma_m, Nkernel, size_cell):
    L_m = Nkernel*size_cell
#     #i not eq i0 mask
#     mask = np.ones(rays.shape[0],dtype=np.bool)
#     mask[i0] = False
#     rays = rays[mask,:,:,:,:]
#     g = g[mask,:,:]
#     dobs = dobs[mask,:,:]
#     CdCt = CdCt[mask,:,:]
    #residuals
    #g.shape, dobs.shape [Na,Nt,Nd]
    dd = g - dobs
    #weighted residuals
    #Cd.shape [Na,Nt,Nd] i.e. diagonal
    #CdCt^-1 = 1./CdCt
    dd /= (CdCt + 1e-15)
    #get ray info
    Na, Nt, Nd, _ ,Ns = rays.shape
#     if Na < Nd:
#         #parallelize over antennas
#         gradient = np.sum(np.stack([do_gradient(rays[i,:,:,:,:], dd[i,:,:], K_ne, m_tci, 
#                                           sigma_m, Nkernel, size_cell) for i in range(Na)],axis=-1),axis=-1)
#     else:
#         #parallelize over directions
#         gradient = np.sum(np.stack([do_gradient(rays[:,:,d,:,:], dd[:,:,d], K_ne, m_tci, 
#                                          sigma_m, Nkernel, size_cell) for d in range(Nd)],axis=-1),axis=-1)

    #parallelize over directions
    gradient = np.sum(np.stack([do_adjoint(rays[:,:,d,:,:], dd[:,:,d], K_ne, m_tci, 
                                           sigma_m, Nkernel, size_cell,i0) for d in range(Nd)],axis=-1),axis=-1)
              
    gradient += m_tci.M
    gradient -= m_prior
    
    return gradient

def do_gradient(rays, dd, K_ne, m_tci, sigma_m, Nkernel, size_cell, i0):
    '''Gradient of S is G^t.CdCt^-1.(g-dobs) + Cm^-1.(m - mprior)'''

    adjoint = do_adjoint(rays, dd, K_ne, m_tci, sigma_m, Nkernel, size_cell, i0)
    
#    Nkernel=0
#    #print("Doing gradient")
#    L_m = Nkernel*size_cell
#    #if antennas parallelization Nt,Nd
#    #if directions parallelization Na,Nd
#    N1,N2,_,Ns = rays.shape
#    m_shape = [N1,N2,m_tci.nx,m_tci.ny,m_tci.nz]
#    grad = np.zeros([m_tci.nx,m_tci.ny,m_tci.nz],dtype=np.double)
#    
#    mask = np.zeros(m_shape, dtype=np.bool)
#    #idx_min = np.ones(m_shape,dtype=np.int64)*Ns
#    #idx_max = np.ones(m_shape,dtype=np.int64)*-1
#    #nevec = np.zeros([N1,N2,Ns],dtype=np.double)
#    #go through the mask
#    j = 0
#    while j < N1:
#        k = 0
#        while k < N2:
#            x_ray = rays[j,k,0,:]
#            y_ray = rays[j,k,1,:]
#            z_ray = rays[j,k,2,:]
#            s_ray = rays[j,k,3,:]
#            idx = 0
#            while idx < Ns:
#                #nevec[j,k,idx] = K_ne*np.exp(m_tci.interp(x_ray[idx],y_ray[idx],z_ray[idx]))/1e16
#                xi,yi,zi = bisection(m_tci.xvec,x_ray[idx]),bisection(m_tci.yvec,y_ray[idx]),bisection(m_tci.zvec,z_ray[idx])
#                local_mask = (j,k,slice(max(0,xi - Nkernel), min(m_tci.nx - 1, xi + Nkernel + 1)),
#                            slice(max(0,yi - Nkernel) , min(m_tci.ny - 1,yi + Nkernel + 1)),
#                            slice(max(0, zi - Nkernel), min(m_tci.nz - 1, zi + Nkernel + 1)))
#                mask[local_mask] = True
#                shape = mask[local_mask].shape
##                 idx_max[local_mask] = np.max(np.stack([idx_max[local_mask],
##                                                      np.ones(shape,dtype=np.int64)*idx],axis=-1),axis=-1)
##                 #print(idx_max[local_mask])
##                 idx_min[local_mask] = np.min(np.stack([idx_min[local_mask],
##                                                      np.ones(shape,dtype=np.int64)*idx],axis=-1),axis=-1)
#                idx += 1   
#            k += 1
#        j += 1
#    
#    #Cm^-1 (m-mprior) 
#    dmpart = np.zeros([m_tci.nx,m_tci.ny,m_tci.nz],dtype=np.double)
#    sum_mask = np.sum(np.sum(mask,axis=0),axis=0)#is there any ray in the cell at all?
#    xi = 0
#    while xi < m_tci.nx:
#        yi = 0
#        while yi < m_tci.ny:
#            zi = 0
#            while zi < m_tci.nz:
#                if not sum_mask[xi,yi,zi]:
#                    zi += 1
#                    continue
#                x,y,z = m_tci.xvec[xi],m_tci.yvec[yi],m_tci.zvec[zi]
#                j = 0
#                while j < N2:
#                    i = 0
#                    while i < N1:
#                        paircomp = 0.
#                        if mask[i,j,xi,yi,zi]:
#                            paircomp = 1.
#                        if mask[i0,j,xi,yi,zi]:
#                            paircomp -= 1.
#                        grad[xi,yi,zi] += dd[i,j]*paircomp*K_ne*np.exp(m_tci.interp(m_tci.xvec[xi],
#                                                                           m_tci.yvec[yi],
#                                                                           m_tci.zvec[zi]))/1e12
#                            
#                            
#                        i += 1
#                    j += 1
#                zi += 1
#            yi += 1
#        xi += 1
#    return grad
 

def compute_gradient_dask(rays, g, dobs, i0, K_ne, m_tci, m_prior, CdCt, sigma_m, Nkernel, size_cell, cov_obj=None):
    L_m = Nkernel*size_cell
#     #i not eq i0 mask
#     mask = np.ones(rays.shape[0],dtype=np.bool)
#     mask[i0] = False
#     rays = rays[mask,:,:,:,:]
#     g = g[mask,:,:]
#     dobs = dobs[mask,:,:]
#     CdCt = CdCt[mask,:,:]
    #residuals
    #g.shape, dobs.shape [Na,Nt,Nd]
    dd = g - dobs
    #weighted residuals
    #Cd.shape [Na,Nt,Nd] i.e. diagonal
    #CdCt^-1 = 1./CdCt
    dd /= (CdCt + 1e-15)
    #get ray info
    Na, Nt, Nd, _ ,Ns = rays.shape
#     if Na < Nd:
#         #parallelize over antennas
#         gradient = da.sum(da.stack([da.from_delayed(delayed(do_gradient)(rays[i,:,:,:,:], dd[i,:,:], K_ne, m_tci, 
#                                          sigma_m, Nkernel, size_cell),(m_tci.nx,m_tci.ny,m_tci.nz),dtype=np.double) for i in range(Na)],axis=-1),axis=-1)
#     else:
#         #parallelize over directions
#         gradient = da.sum(da.stack([da.from_delayed(delayed(do_gradient)(rays[:,:,d,:,:], dd[:,:,d], K_ne, m_tci, 
#                                           sigma_m, Nkernel, size_cell),(m_tci.nx,m_tci.ny,m_tci.nz),dtype=np.double) for d in range(Nd)],axis=-1),axis=-1)
        #parallelize over directions
    gradient = da.sum(da.stack([da.from_delayed(delayed(do_gradient)(rays[:,:,d,:,:], dd[:,:,d], K_ne, m_tci, 
                                      sigma_m, Nkernel, size_cell, i0),(m_tci.nx,m_tci.ny,m_tci.nz),dtype=np.double) for d in range(Nd)],axis=-1),axis=-1)
    gradient = gradient.compute(get=get)
    if cov_obj is not None:
        dm = m_tci.M - m_prior
        gradient + cov_obj.contract(dm)
    #gradient += m_tci.M
    #gradient -= m_prior
    
    return gradient



def compute_gradient(rays, g, dobs, i0, K_ne, m_tci, m_prior, CdCt, sigma_m, Nkernel, size_cell, cov_obj=None):
    L_m = Nkernel*size_cell
#     #i not eq i0 mask
#     mask = np.ones(rays.shape[0],dtype=np.bool)
#     mask[i0] = False
#     rays = rays[mask,:,:,:,:]
#     g = g[mask,:,:]
#     dobs = dobs[mask,:,:]
#     CdCt = CdCt[mask,:,:]
    #residuals
    #g.shape, dobs.shape [Na,Nt,Nd]
    dd = g - dobs
    #weighted residuals
    #Cd.shape [Na,Nt,Nd] i.e. diagonal
    #CdCt^-1 = 1./CdCt
    dd /= (CdCt + 1e-15)
    #get ray info
    Na, Nt, Nd, _ ,Ns = rays.shape
#     if Na < Nd:
#         #parallelize over antennas
#         gradient = np.sum(np.stack([do_gradient(rays[i,:,:,:,:], dd[i,:,:], K_ne, m_tci, 
#                                           sigma_m, Nkernel, size_cell) for i in range(Na)],axis=-1),axis=-1)
#     else:
#         #parallelize over directions
#         gradient = np.sum(np.stack([do_gradient(rays[:,:,d,:,:], dd[:,:,d], K_ne, m_tci, 
#                                          sigma_m, Nkernel, size_cell) for d in range(Nd)],axis=-1),axis=-1)

    #parallelize over directions
    gradient = np.sum(np.stack([do_gradient(rays[:,:,d,:,:], dd[:,:,d], K_ne, m_tci, 
                                           sigma_m, Nkernel, size_cell,i0) for d in range(Nd)],axis=-1),axis=-1)
    if cov_obj is not None:
        dm = m_tci.M - m_prior
        gradient + cov_obj.contract(dm)
    #gradient += m_tci.M
    #gradient -= m_prior
    return gradient


