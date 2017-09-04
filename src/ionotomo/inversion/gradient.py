'''The gradient for steepest direction, i.e. <Cm, d/dm(-log(posterior))> 
is equal to Adjoint(G).(g(m) - d_obs) + (m - m_prior) = Cm.G^t.Cd^-1 .( g(m) - d_obs ) + (m - m_prior)'''

from ionotomo.geometry.tri_cubic import bisection 
import numpy as np
from scipy.integrate import simps
import dask.array as da
from dask import delayed
from dask.multiprocessing import get
from ionotomo.ionosphere.covariance import Covariance

from ionotomo.geometry.ray_dirac import get_ray_dirac

TECU = 1e13#with km factor
def do_gradient(rays, dd, ne_tci, sigma_m, Nkernel, size_cell, i0):
    '''Gradient of S is K*exp(m) sum_ijk (s_ijk - s_i0jk) * dd_ijk / CdCt_ijk'''
    ray_dirac,midpoints = get_ray_dirac(rays,ne_tci)
    #ray_dirac -= ray_dirac[i0,...]
    G = np.einsum("ijklm,klm,ij->klm",ray_dirac,ne_tci.M,dd,optimize=True)
    return G  

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
    ne_tci = m_tci.copy()
    np.exp(ne_tci.M,out=ne_tci.M)
    ne_tci.M *= K_ne/TECU
    gradient = da.sum(da.stack([da.from_delayed(delayed(do_gradient)(rays[:,:,d,:,:], dd[:,:,d], ne_tci, 
                                      sigma_m, Nkernel, size_cell, i0),(m_tci.nx,m_tci.ny,m_tci.nz),dtype=np.double) for d in range(Nd)],axis=-1),axis=-1)
    gradient = gradient.compute(get=get)
    gradient -= gradient[i0,...]
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
    gradient = np.sum(np.stack([do_gradient(rays[:,:,d,:,:], dd[:,:,d], K_ne, m_tci, sigma_m, Nkernel, size_cell,i0) for d in range(Nd)],axis=-1),axis=-1)
    if cov_obj is not None:
        dm = m_tci.M - m_prior
        gradient + cov_obj.contract(dm)
    #gradient += m_tci.M
    #gradient -= m_prior
    return gradient


