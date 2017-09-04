import numpy as np
import pylab as plt
from ionotomo.astro.real_data import generate_example_datapack
#from ionotomo.geometry.tri_cubic import TriCubic
from ionotomo.geometry.calc_rays import *
#from ionotomo.inversion.forward_equation import *
from ionotomo.inversion.initial_model import *
from ionotomo.inversion.gradient import *
from ionotomo.inversion.inversion_pipeline import *
from ionotomo.inversion.lbfgs_solver import LBFGSSolver
from ionotomo.inversion.forward_equation import *
from time import clock

def test_inversion_pipeline():
    return
    datapack = generate_example_datapack()
    p = InversionPipeline(datapack,LBFGSSolver)
    p.run()

def test_initial_model():
    return
    datapack = generate_example_datapack()
    ne_tci = create_initial_model(datapack)
    pert_tci = create_turbulent_model(datapack)
    #pert_tci.M -= ne_tci.M
    pert_tci.save("pert_tci.hdf5")
    #import pylab as plt
    #plt.hist(pert_tci.M.flatten(),bins = int(np.sqrt(np.size(pert_tci.M))))
    #plt.show()

def test_calc_rays():
    return
    datapack = generate_example_datapack()
    ne_tci = create_initial_model(datapack)
    antennas,antenna_labels = datapack.get_antennas(ant_idx = -1)
    patches, patch_names = datapack.get_directions(dir_idx=-1)
    times,timestamps = datapack.get_times(time_idx=-1)
    Na = len(antennas)
    Nt = len(times)
    Nd = len(patches)  
    fixtime = times[Nt>>1]
    phase = datapack.get_center_direction()
    array_center = datapack.radio_array.get_center()
    t1 = clock()
    rays1 = calc_rays(antennas,patches,times, array_center, fixtime, phase, ne_tci, 120e6, True, 1000, 1000)
    #print(clock() - t1)
    t1 = clock()
    rays2 = calc_rays_dask(antennas,patches,times, array_center, fixtime, phase, ne_tci, 120e6, True, 1000, 1000)
    #print(clock() - t1)
    print("Num of rays calculated: {}".format(rays1.shape[0]*rays1.shape[1]*rays1.shape[2]))
    assert np.all(rays1==rays2),"Not same result"
    assert rays1.shape[0] == Na and rays1.shape[1] == Nt and rays1.shape[2] == Nd and rays1.shape[3] == 4 and rays1.shape[4] == 1000
  
def test_forward_equation():
    return
    datapack = generate_example_datapack()
    antennas,antenna_labels = datapack.get_antennas(ant_idx = -1)
    patches, patch_names = datapack.get_directions(dir_idx = -1)
    times,timestamps = datapack.get_times(time_idx=-1)
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
    #print (np.any(np.isnan(m_tci.M)))
    assert d.shape[0] == Na and d.shape[1] == Nt and d.shape[2] == Nd
    assert not np.any(np.isnan(d))
    d_dask = forward_equation_dask(rays,K_ne, m_tci,i0)
    assert np.all(d==d_dask)
    t1 = clock()
    res = [forward_equation(rays,K_ne,m_tci,i0) for i in range(10)]
    print("Average time (serial) {}s".format((clock() - t1)/10.))
    t1 = clock()
    res = [forward_equation_dask(rays,K_ne,m_tci,i0) for i in range(10)]
    print("Average time (dask) {}s".format((clock() - t1)/10.))

    
    
def test_scipy_bfgs_inversion():
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
