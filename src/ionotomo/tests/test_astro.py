import numpy as np
from ionotomo import *
import os



def test_radio_array():
    radio_array = generate_example_radio_array(Nant=10)
    assert radio_array.Nantenna == 10
    assert os.path.isfile(RadioArray.lofar_array)
    radio_array = RadioArray(RadioArray.lofar_array)
    print(radio_array)
     
    
def test_real_data():
    print("Test read data")
    datapack = generate_example_datapack(Nant=12, Ntime = 10, Ndir = 12,fov = 4., alt = 90., az=0., time = None, radio_array=None) 
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
    datapack_screen = phase_screen_datapack(10,Nant = 10, Ntime = 1, fov = 4., alt = 90., az=0., time = None, radio_array=None,datapack=datapack)
    assert datapack_screen.radio_array == datapack.radio_array
    assert datapack_screen.Na == datapack.Na
    assert datapack_screen.Nt == datapack.Nt
    assert datapack_screen.Nd == 10**2

    
def test_antenna_facet_selection():
    print("Test antenna facet selection")
    datapack = generate_example_datapack(Nant=12, Ndir = 12)
    datapack_sel = select_antennas(10,datapack,-1, time_idx = [0])
    antennas,antenna_labels = datapack_sel.get_antennas(ant_idx = -1)
    times,timestamps = datapack_sel.get_times(time_idx=[0])
    Na = len(antennas)
    assert Na == 10
   
    datapack_sel = select_facets(8,datapack,-1, time_idx = [0])
    patches, patch_names = datapack_sel.get_directions(dir_idx = -1)
    times,timestamps = datapack_sel.get_times(time_idx=[0])
    Nd = len(patches)
    assert Nd == 8
