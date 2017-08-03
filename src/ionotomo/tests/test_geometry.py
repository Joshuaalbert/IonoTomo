import numpy as np
from ionotomo.geometry.tri_cubic import *
from ionotomo.geometry.calc_rays import *
from ionotomo.astro.read_data import generate_example_datapack
import os
from time import clock

def test_tri_cubic():
    xvec = np.linspace(-0.1,1.1,100)
    yvec = np.linspace(-0.1,1.1,100)
    zvec = np.linspace(-0.1,1.11,100)
    M = np.random.uniform(size=[100,100,100])
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

def test_calc_rays():
    datapack = generate_example_datapack(Ntime=2)
    antennas,antenna_labels = datapack.get_antennas(ant_idx = ant_idx)
    patches, patch_names = datapack.get_directions(dir_idx=dir_idx)
    times,timestamps = datapack.get_times(time_idx=time_idx)
    Na = len(antennas)
    Nt = len(times)
    Nd = len(patches)  
    #Setting up ionosphere to use
    array_center = datapack.radio_array.get_center()
    phase = datapack.get_center_direction()
    fixtime = times[Nt>>1]
    tci = create_turbulent_model(datapack,ant_idx = -1, time_idx = -1, dir_idx = -1, zmax = 1000., spacing=5.)
    rays = calc_rays(antennas,patches,times,array_center,fixtime,phase,tci,datapack.radio_array.frequency,True,1000.)
    print(rays)
