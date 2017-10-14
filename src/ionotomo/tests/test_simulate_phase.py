import numpy as np
import pylab as plt
from ionotomo import *

def test_simulate_phase():
    datapack = generate_example_datapack(Ntime = 1)
    datapack = simulate_phase(datapack,num_threads=1,datafolder='turbulent_simulation',
                     ant_idx=-1,time_idx=-1,dir_idx=-1,freq_idx=-1,do_plot_datapack=True)

