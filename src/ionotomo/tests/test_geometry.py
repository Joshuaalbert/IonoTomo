import numpy as np
from ionotomo import *
import os


def test_tri_cubic():
    xvec = np.linspace(-0.1,1.1,100)
    yvec = np.linspace(-0.1,1.1,100)
    zvec = np.linspace(-0.1,1.11,100)
    M = np.random.uniform(size=[100,100,100])
    x,y,z = np.meshgrid(xvec,yvec,zvec,indexing='ij')
    M = x*y*z + x - y - 2*z + x**2
    tci = TriCubic(xvec,yvec,zvec,M)
    
    #print(tci.interp(0.2,0.2,0.2), tci.interp(0.2,0.2,0.2))
    #save test
    tci.save("test.hdf5")
    tci2 = TriCubic(filename="test.hdf5").copy()#test copy also
    assert np.all(tci2.M == tci.M)
    os.system('rm test.hdf5')
    t1 = clock()
    res1 = np.array([tci.interp(1./i,1./i,1./i) for i in range(1,1000)])
    print("Serial 1000 items time: {}".format(clock() - t1))
    x = np.array([1./i for i in range(1,1000)])
    t1 = clock()
    res2 = tci.interp(x,x,x)
    print("Parallel 1000 items time: {}".format(clock() - t1))
    assert np.all(res2==res1)

    #test extrapolation
    a = tci.extrapolate(1.1,1.1,1.1)
    b = tci.extrapolate(2,2,2)
    print(a,b)
    print(2*2*2 + 2 - 2 - 2*2 + 2**2)

def test_calc_rays():
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
    print('time serial',clock() - t1)
    t1 = clock()
    rays2 = calc_rays_dask(antennas,patches,times, array_center, fixtime, phase, ne_tci, 120e6, True, 1000, 1000)
    print('time parallel',clock() - t1)
    print("Num of rays calculated: {}".format(rays1.shape[0]*rays1.shape[1]*rays1.shape[2]))
    assert np.all(rays1==rays2),"Not same result"
    assert rays1.shape[0] == Na and rays1.shape[1] == Nt and rays1.shape[2] == Nd and rays1.shape[3] == 4 and rays1.shape[4] == 1000


