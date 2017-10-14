import numpy as np
import pylab as plt
from ionotomo import *

def test_forward_equation():
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
    

