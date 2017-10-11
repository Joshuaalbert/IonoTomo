import numpy as np
import pylab as plt
from ionotomo import *

from ionotomo.inversion.gradient import *


def test_scipy_bfgs_inversion():
    return
    datapack = generate_example_datapack()
    antennas,antenna_labels = datapack.get_antennas(ant_idx = -1)
    patches, patch_names = datapack.get_directions(dir_idx = -1)
    times,timestamps = datapack.get_times(time_idx=-1)
    dobs = datapack.get_dtec(ant_idx = -1, time_idx = -1, dir_idx = -1)
    Na = len(antennas)
    Nt = len(times)
    Nd = len(patches)  
    fixtime = times[Nt>>1]
    phase = datapack.get_center_direction()
    array_center = datapack.radio_array.get_center()
    ne_tci = create_initial_model(datapack)
    rays = calc_rays(antennas,patches,times, array_center, fixtime, phase, ne_tci, datapack.radio_array.frequency, True, 1000, ne_tci.nz) 
    m_tci = ne_tci.copy()
    K_ne = np.median(m_tci.M)
    m_tci.M = np.log(m_tci.M/K_ne)
    i0 = 0
    d = forward_equation(rays,K_ne,m_tci,i0)
    CdCt = (0.01*np.ones(dobs.shape))**2

    def func_and_gradient(M,K_ne,m_tci,rays,i0,dobs,CdCt):
        m_tci.M = M.copy().reshape(m_tci.M.shape)
        g = forward_equation(rays,K_ne,m_tci,i0)
        grad = compute_gradient_dask(rays, d, dobs,  i0, K_ne, m_tci, m_tci.M, CdCt, 1, 4, 5., None)
        S = np.sum((g-dobs)**2/(CdCt+1e-15))/2.
        print(S,grad)
        return S,grad.flatten()
    from scipy.optimize import fmin_l_bfgs_b
    m0 = m_tci.M.flatten().copy()
    res = fmin_l_bfgs_b(func_and_gradient,m0,fprime=None,args=(K_ne,m_tci,rays,i0,dobs,CdCt))#,m=1000)

def test_gradient():
    return
    datapack = generate_example_datapack()
    antennas,antenna_labels = datapack.get_antennas(ant_idx = -1)
    patches, patch_names = datapack.get_directions(dir_idx = -1)
    times,timestamps = datapack.get_times(time_idx=-1)
    dobs = datapack.get_dtec(ant_idx = -1, time_idx = -1, dir_idx = -1)
    Na = len(antennas)
    Nt = len(times)
    Nd = len(patches)  
    fixtime = times[Nt>>1]
    phase = datapack.get_center_direction()
    array_center = datapack.radio_array.get_center()
    ne_tci = create_initial_model(datapack)
    rays = calc_rays(antennas,patches,times, array_center, fixtime, phase, ne_tci, datapack.radio_array.frequency, True, 1000, ne_tci.nz) 
    m_tci = ne_tci.copy()
    K_ne = np.median(m_tci.M)
    m_tci.M = np.log(m_tci.M/K_ne)
    #print(m_tci.M)
    i0 = 0
    d = forward_equation(rays,K_ne,m_tci,i0)
    
    cov_obj = Covariance(ne_tci,np.log(10),20.,2./3.)
    cov_obj=None
    CdCt = (0.01*np.ones(dobs.shape))**2

#    t1 = clock()
#    res = [compute_gradient(rays, d, dobs,  i0, K_ne, m_tci, m_tci.M, CdCt, 1, 4, 5., cov_obj) for i in range(2)]
#    print("Mean time for gradient (serial): {}s".format((clock() - t1)/2.))
    res = compute_gradient_dask(rays, d, dobs,  i0, K_ne, m_tci, m_tci.M, CdCt, 1, 4, 5., cov_obj) 
    ### random gradient numerical check
    print("Doing random numerical check")
    gradient = res
    S0 = np.sum((d-dobs)**2/(CdCt+1e-15))/2.
    i = 0
    Ncheck = 20
    while i < Ncheck:
        xi,yi,zi = np.random.randint(ne_tci.nx),np.random.randint(ne_tci.ny),np.random.randint(ne_tci.nz)
        while gradient[xi,yi,zi] == 0:
            xi,yi,zi = np.random.randint(ne_tci.nx),np.random.randint(ne_tci.ny),np.random.randint(ne_tci.nz)
        m_tci.M[xi,yi,zi] += 1e-7
        g = forward_equation(rays,K_ne,m_tci,i0)
        S = np.sum((g-dobs)**2/(CdCt+1e-15))/2.
        grad_num = (S - S0)/1e-7
        m_tci.M[xi,yi,zi] -= 1e-7
        print("Numerical gradient[{},{},{}] = {}, calculated = {}".format(xi,yi,zi,grad_num,gradient[xi,yi,zi]))
        i += 1   
