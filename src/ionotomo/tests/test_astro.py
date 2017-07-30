import numpy as np
from ionotomo.astro.antenna_facet_selection import *
from ionotomo.astro.real_data import *
from ionotomo.astro.radio_array import *
import os

def test_radio_array():
    print("Test radio array")
    radio_array = generate_example_radio_array(Nant=10)
    print(radio_array)
    assert radio_array.Nantenna == 10
    assert os.path.isfile(RadioArray.lofar_array)
    radio_array = RadioArray(RadioArray.lofar_array)
    print(radio_array)
     
    
def test_real_data():
    print("Test read data")
    datapack = generate_example_datapack(Nant=12, Ntime = 10, Ndir = 12) 
    assert datapack.Na == 12
    assert datapack.Nt == 10
    assert datapack.Nd == 12
    patches, patch_names = datapack.get_directions(dir_idx=-1)
    antennas,antenna_labels = datapack.get_antennas(ant_idx=-1)
    times,timestamps = datapack.get_times(time_idx = -1)
    datapack.flag_antennas([antenna_labels[0]])
    datapack.flag_times([timestamps[0]])
    datapack.flag_directions([patch_names[0]])
    assert datapack.Na == 11
    assert datapack.Nt == 9
    assert datapack.Nd == 11

    
def test_antenna_facet_selection():
    print("Test antenna facet selection")
    datapack = generate_example_datapack(Nant=10, Ndir = 12)
    datapack_sel = select_antennas(10,datapack,-1, time_idx = [0])
    antennas,antenna_labels = datapack_sel.get_antennas(ant_idx = -1)
    times,timestamps = datapack_sel.get_times(time_idx=[0])
    Na = len(antennas)
    Nt = len(times)
    fixtime = times[Nt>>1]
    #print("Fixing frame at {}".format(fixtime.isot))
    phase = datapack_sel.get_center_direction()
    uvw = UVW(location = datapack_sel.radio_array.get_center().earth_location,obstime = fixtime,phase = phase)
    #
    
    ants = antennas.transform_to(uvw).cartesian.xyz.to(au.km).value.transpose()
    center = np.mean(ants,axis=0)
    dist = ants-center
    mag = np.linalg.norm(dist,axis=1)
    #plt.scatter(ants[:,0],ants[:,1])
    #plt.show()

    datapack_sel = select_facets(8,datapack,-1, time_idx = [0])
    patches, patch_names = datapack_sel.get_directions(dir_idx = -1)
    times,timestamps = datapack_sel.get_times(time_idx=[0])
    Nd = len(patches)
    Nt = len(times)
    fixtime = times[Nt>>1]
    #print("Fixing frame at {}".format(fixtime.isot))
    phase = datapack_sel.get_center_direction()
    uvw = UVW(location = datapack_sel.radio_array.get_center().earth_location,obstime = fixtime,phase = phase)
    #
    
    dirs = patches.transform_to(uvw).cartesian.xyz.value.transpose()
    center = np.mean(dirs,axis=0)
    mag = np.arccos(np.dot(dirs,center))
    #plt.scatter(dirs[:,0],dirs[:,1])
    #plt.show()


