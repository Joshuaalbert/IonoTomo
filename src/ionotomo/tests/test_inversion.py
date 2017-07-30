import numpy as np
import pylab as plt
from ionotomo.astro.real_data import generate_example_datapack
from ionotomo.geometry.tri_cubic import TriCubic
from ionotomo.inversion.calc_rays import *
from ionotomo.inversion.forward_equation import *
from ionotomo.inversion.initial_model import *
from ionotomo.inversion.gradient import *

def test_gradient():

    i0 = 0
    #datapack = DataPack(filename="output/test/datapack_sim.hdf5").clone()
    datapack = generate_example_datapack()
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

    plt.hist(gradient.flatten())

    plt.yscale('log')

    plt.show()

    plt.plot(gradient.flatten())
    plt.show()

    
