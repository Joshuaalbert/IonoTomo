from ionotomo.geometry.tri_cubic import TriCubic
from ionotomo.plotting.mayavi_tools import plot_tci
import numpy as np
#import pylab as plt
#from ionotomo.astro.real_data import generate_example_datapack
#from ionotomo.geometry.calc_rays import *
#from ionotomo.inversion.initial_model import *

def test_plot_tci():
#    datapack = generate_example_datapack()
#    ne_tci = create_initial_model(datapack)
#    nt_pert = create_turbulent_model(datapack)
#
#    antennas,antenna_labels = datapack.get_antennas(ant_idx = -1)
#    patches, patch_names = datapack.get_directions(dir_idx=-1)
#    times,timestamps = datapack.get_times(time_idx=-1)
#    Na = len(antennas)
#    Nt = len(times)
#    Nd = len(patches)  
#    fixtime = times[Nt>>1]
#    phase = datapack.get_center_direction()
#    array_center = datapack.radio_array.get_center()
#    rays = calc_rays(antennas,patches,times, array_center, fixtime, phase, ne_tci, 120e6, True, 1000, 1000)
    pert_tci = TriCubic(filename='pert_tci.hdf5')
    plot_tci(pert_tci,None,show=True)
