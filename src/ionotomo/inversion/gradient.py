
# coding: utf-8

# In[1]:

'''The gradient for steepest direction, i.e. <Cm, d/dm(-log(posterior))> 
is equal to Adjoint(G).(g(m) - d_obs) + (m - m_prior) = Cm.G^t.Cd^-1 .( g(m) - d_obs ) + (m - m_prior)'''

from Bisect import bisect
import numpy as np
from scipy.integrate import simps
import dask.array as da
from dask import delayed
from dask.multiprocessing import get
from Covariance import Covariance

def do_adjoint(rays, dd, K_ne, m_tci, sigma_m, Nkernel, size_cell, i0):
    #print("Doing gradient")
    L_m = Nkernel*size_cell
    #if antennas parallelization Nt,Nd
    #if directions parallelization Na,Nd
    N1,N2,_,Ns = rays.shape
    mShape = [N1,N2,m_tci.nx,m_tci.ny,m_tci.nz]
    grad = np.zeros([m_tci.nx,m_tci.ny,m_tci.nz],dtype=np.double)
    
    mask = np.zeros(mShape, dtype=np.bool)
    idx_min = np.ones(mShape,dtype=np.int64)*Ns
    idxMax = np.ones(mShape,dtype=np.int64)*-1
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
            idx = 0
            while idx < Ns:
                nevec[j,k,idx] = K_ne*np.exp(m_tci.interp(x_ray[idx],y_ray[idx],z_ray[idx]))/1e13
                xi,yi,zi = bisect(m_tci.xvec,x_ray[idx]),bisect(m_tci.yvec,y_ray[idx]),bisect(m_tci.zvec,z_ray[idx])
                localMask = (j,k,slice(max(0,xi - Nkernel), min(m_tci.nx - 1, xi + Nkernel + 1)),
                            slice(max(0,yi - Nkernel) , min(m_tci.ny - 1,yi + Nkernel + 1)),
                            slice(max(0, zi - Nkernel), min(m_tci.nz - 1, zi + Nkernel + 1)))
                mask[localMask] = True
                shape = mask[localMask].shape
                idxMax[localMask] = np.max(np.stack([idxMax[localMask],
                                                     np.ones(shape,dtype=np.int64)*idx],axis=-1),axis=-1)
                #print(idxMax[localMask])
                idx_min[localMask] = np.min(np.stack([idx_min[localMask],
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
                            segment_mask = (slice(idx_min[i,j,xi,yi,zi],idxMax[i,j,xi,yi,zi]+1),)
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
                            if i == i0:
                                grad[xi,yi,zi] -= N1*comp
                            
                        i += 1
                    j += 1
                zi += 1
            yi += 1
        xi += 1
    return grad
 
    
                
          
def computeAdjoint_dask(rays, g, dobs, i0, K_ne, m_tci, mPrior, CdCt, sigma_m, Nkernel, size_cell):
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
    gradient += m_tci.get_shaped_array()
    gradient -= mPrior
    
    return gradient

def computeAdjoint(rays, g, dobs, i0, K_ne, m_tci, mPrior, CdCt, sigma_m, Nkernel, size_cell):
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
              
    gradient += m_tci.get_shaped_array()
    gradient -= mPrior
    
    return gradient

def do_gradient(rays, dd, K_ne, m_tci, sigma_m, Nkernel, size_cell, i0):
    '''Gradient of S is G^t.CdCt^-1.(g-dobs) + Cm^-1.(m - mprior)'''
    Nkernel=0
    #print("Doing gradient")
    L_m = Nkernel*size_cell
    #if antennas parallelization Nt,Nd
    #if directions parallelization Na,Nd
    N1,N2,_,Ns = rays.shape
    mShape = [N1,N2,m_tci.nx,m_tci.ny,m_tci.nz]
    grad = np.zeros([m_tci.nx,m_tci.ny,m_tci.nz],dtype=np.double)
    
    mask = np.zeros(mShape, dtype=np.bool)
    #idx_min = np.ones(mShape,dtype=np.int64)*Ns
    #idxMax = np.ones(mShape,dtype=np.int64)*-1
    #nevec = np.zeros([N1,N2,Ns],dtype=np.double)
    #go through the mask
    j = 0
    while j < N1:
        k = 0
        while k < N2:
            x_ray = rays[j,k,0,:]
            y_ray = rays[j,k,1,:]
            z_ray = rays[j,k,2,:]
            s_ray = rays[j,k,3,:]
            idx = 0
            while idx < Ns:
                #nevec[j,k,idx] = K_ne*np.exp(m_tci.interp(x_ray[idx],y_ray[idx],z_ray[idx]))/1e16
                xi,yi,zi = bisect(m_tci.xvec,x_ray[idx]),bisect(m_tci.yvec,y_ray[idx]),bisect(m_tci.zvec,z_ray[idx])
                localMask = (j,k,slice(max(0,xi - Nkernel), min(m_tci.nx - 1, xi + Nkernel + 1)),
                            slice(max(0,yi - Nkernel) , min(m_tci.ny - 1,yi + Nkernel + 1)),
                            slice(max(0, zi - Nkernel), min(m_tci.nz - 1, zi + Nkernel + 1)))
                mask[localMask] = True
                shape = mask[localMask].shape
#                 idxMax[localMask] = np.max(np.stack([idxMax[localMask],
#                                                      np.ones(shape,dtype=np.int64)*idx],axis=-1),axis=-1)
#                 #print(idxMax[localMask])
#                 idx_min[localMask] = np.min(np.stack([idx_min[localMask],
#                                                      np.ones(shape,dtype=np.int64)*idx],axis=-1),axis=-1)
                idx += 1   
            k += 1
        j += 1
    
    #Cm^-1 (m-mprior) 
    dmpart = np.zeros([m_tci.nx,m_tci.ny,m_tci.nz],dtype=np.double)
    sum_mask = np.sum(np.sum(mask,axis=0),axis=0)#is there any ray in the cell at all?
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
                        paircomp = 0.
                        if mask[i,j,xi,yi,zi]:
                            paircomp = 1.
                        if mask[i0,j,xi,yi,zi]:
                            paircomp -= 1.
                        grad[xi,yi,zi] += dd[i,j]*paircomp*K_ne*np.exp(m_tci.interp(m_tci.xvec[xi],
                                                                           m_tci.yvec[yi],
                                                                           m_tci.zvec[zi]))/1e12
                            
                            
                        i += 1
                    j += 1
                zi += 1
            yi += 1
        xi += 1
    return grad
 

def compute_gradient_dask(rays, g, dobs, i0, K_ne, m_tci, mPrior, CdCt, sigma_m, Nkernel, size_cell, cov_obj=None):
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
        dm = m_tci.get_shaped_array() - mPrior
        gradient + cov_obj.contract(dm)
    #gradient += m_tci.get_shaped_array()
    #gradient -= mPrior
    
    return gradient



def compute_gradient(rays, g, dobs, i0, K_ne, m_tci, mPrior, CdCt, sigma_m, Nkernel, size_cell, cov_obj=None):
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
        dm = m_tci.get_shaped_array() - mPrior
        gradient + cov_obj.contract(dm)
    #gradient += m_tci.get_shaped_array()
    #gradient -= mPrior
    return gradient

def test_compute_gradient():
    from real_data import DataPack
    from tri_cubic import TriCubic
    from CalcRays import calcRays
    from ForwardEquation import forward_equation
    from InitialModel import create_initial_model
    i0 = 0
    #datapack = DataPack(filename="output/test/datapack_sim.hdf5").clone()
    datapack = DataPack(filename="output/test/simulate/simulate_0/datapack_sim.hdf5").clone()
    antennas,antenna_labels = datapack.get_antennas(ant_idx = -1)
    patches, patch_names = datapack.get_directions(dir_idx=-1)
    times,timestamps = datapack.get_times(time_idx=[0])
    datapack.set_reference_antenna(antenna_labels[i0])
    ne_tci = create_initial_model(datapack,ant_idx = -1, time_idx = [0], dir_idx = -1, zmax = 1000.)
    cov_obj = Covariance(ne_tci,np.log(5),50.,7./2.)
    dobs = datapack.get_dtec(ant_idx = -1, time_idx = [0], dir_idx = -1)
    CdCt = (0.15*np.mean(dobs))**2
    Na = len(antennas)
    Nt = len(times)
    Nd = len(patches)  
    fixtime = times[Nt>>1]
    phase = datapack.get_center_direction()
    array_center = datapack.radio_array.get_center()
    print("Calculating rays...")
    rays = calcRays(antennas,patches,times, array_center, fixtime, phase, ne_tci, datapack.radio_array.frequency, True, 1000., 100)
    m_tci = ne_tci.copy()
    K_ne = np.mean(m_tci.m)
    m_tci.m /= K_ne
    np.log(m_tci.m,out=m_tci.m)
    #print(ne_tci.m)
    g0 = forward_equation(rays,K_ne,m_tci,i0)
    #gradient = compute_gradient(rays, g, dobs, 0, K_ne, m_tci, m_tci.get_shaped_array(), CdCt, 1, 3, 5.)
    gradient = compute_gradient_dask(rays, g0, dobs,  i0, K_ne, m_tci, m_tci.get_shaped_array(), CdCt, 1, 4, 5., cov_obj)
    ### random gradient numerical check
    print("Doing random numerical check")
    S0 = np.sum((g0-dobs)**2/(CdCt+1e-15))/2.
    i = 0
    Ncheck = 10
    while i < Ncheck:
        xi,yi,zi = np.random.randint(ne_tci.nx),np.random.randint(ne_tci.ny),np.random.randint(ne_tci.nz)
        while gradient[xi,yi,zi] == 0:
            xi,yi,zi = np.random.randint(ne_tci.nx),np.random.randint(ne_tci.ny),np.random.randint(ne_tci.nz)
        m_tci.m[m_tci.index(xi,yi,zi)] += 1e-3
        g = forward_equation(rays,K_ne,m_tci,i0)
        S = np.sum((g-dobs)**2/(CdCt+1e-15)/2.)
        gradNum = (S - S0)/1e-3
        m_tci.m[m_tci.index(xi,yi,zi)] -= 1e-3
        print("Numerical gradient[{},{},{}] = {}, calculated = {}".format(xi,yi,zi,gradNum,gradient[xi,yi,zi]))
        i += 1
    import pylab as plt
    plt.hist(gradient.flatten())
    plt.yscale('log')
    plt.show()
    plt.plot(gradient.flatten())
    plt.show()
    from PlotTools import animate_tci_slices
    gradTCI = TriCubic(m_tci.xvec,m_tci.yvec,m_tci.zvec,gradient)
    animate_tci_slices(gradTCI,"output/test/gradient",num_seconds=20.)
    
def test_computeAdjoint():
    from real_data import DataPack
    from tri_cubic import TriCubic
    from CalcRays import calcRays
    from ForwardEquation import forward_equation
    from InitialModel import create_initial_model
    i0 = 0
    #datapack = DataPack(filename="output/test/datapack_sim.hdf5").clone()
    datapack = DataPack(filename="output/test/simulate/simulate_0/datapack_sim.hdf5").clone()
    antennas,antenna_labels = datapack.get_antennas(ant_idx = -1)
    patches, patch_names = datapack.get_directions(dir_idx=-1)
    times,timestamps = datapack.get_times(time_idx=[0])
    datapack.set_reference_antenna(antenna_labels[i0])
    ne_tci = create_initial_model(datapack,ant_idx = -1, time_idx = [0], dir_idx = -1, zmax = 1000.)
#    cov_obj = Covariance(ne_tci,np.log(5),50.,7./2.)
    dobs = datapack.get_dtec(ant_idx = -1, time_idx = [0], dir_idx = -1)
    CdCt = (0.15*np.mean(dobs))**2
    Na = len(antennas)
    Nt = len(times)
    Nd = len(patches)  
    fixtime = times[Nt>>1]
    phase = datapack.get_center_direction()
    array_center = datapack.radio_array.get_center()
    print("Calculating rays...")
    rays = calcRays(antennas,patches,times, array_center, fixtime, phase, ne_tci, datapack.radio_array.frequency, True, 1000., 100)
    m_tci = ne_tci.copy()
    K_ne = np.mean(m_tci.m)
    m_tci.m /= K_ne
    np.log(m_tci.m,out=m_tci.m)
    #print(ne_tci.m)
    g0 = forward_equation(rays,K_ne,m_tci,i0)
    #gradient = compute_gradient(rays, g, dobs, 0, K_ne, m_tci, m_tci.get_shaped_array(), CdCt, 1, 3, 5.)
    gradient = computeAdjoint_dask(rays, g0, dobs,  i0, K_ne, m_tci, m_tci.get_shaped_array(), CdCt, 1, 4, 5.)
    ### random gradient numerical check
    print("Doing random numerical check")
    S0 = np.sum((g0-dobs)**2/(CdCt+1e-15))/2.
    i = 0
    Ncheck = 10
    while i < Ncheck:
        xi,yi,zi = np.random.randint(ne_tci.nx),np.random.randint(ne_tci.ny),np.random.randint(ne_tci.nz)
        while gradient[xi,yi,zi] == 0:
            xi,yi,zi = np.random.randint(ne_tci.nx),np.random.randint(ne_tci.ny),np.random.randint(ne_tci.nz)
        m_tci.m[m_tci.index(xi,yi,zi)] += 1e-3
        g = forward_equation(rays,K_ne,m_tci,i0)
        S = np.sum((g-dobs)**2/(CdCt+1e-15)/2.)
        gradNum = (S - S0)/1e-3
        m_tci.m[m_tci.index(xi,yi,zi)] -= 1e-3
        print("Numerical gradient[{},{},{}] = {}, calculated = {}".format(xi,yi,zi,gradNum,gradient[xi,yi,zi]))
        i += 1
    import pylab as plt
    plt.hist(gradient.flatten())
    plt.yscale('log')
    plt.show()
    plt.plot(gradient.flatten())
    plt.show()
    from PlotTools import animate_tci_slices
    gradTCI = TriCubic(m_tci.xvec,m_tci.yvec,m_tci.zvec,gradient)
    animate_tci_slices(gradTCI,"output/test/adjoint",num_seconds=20.)
    
if __name__ == '__main__':
    #from PlotTools import make_animation
    #make_animation("output/test/gradient",prefix='fig',fps=6)
    
    #test_compute_gradient()
    test_computeAdjoint()


# In[ ]:



