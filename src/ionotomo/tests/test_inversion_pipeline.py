import numpy as np
import pylab as plt
from ionotomo import *

def test_inversion_pipeline():
    datapack = generate_example_datapack(Ntime = 1)
    datapack = simulate_phase(datapack,num_threads=1,datafolder=None,   ant_idx=-1,time_idx=-1,dir_idx=-1,freq_idx=-1,do_plot_datapack=False)
    p = InversionPipeline(datapack,coherence_time=16.,stateful=True,num_threads_per_solve=4)
    p.preprocess()
    assert len(p.datapack.timestamps)==p.datapack.phase.shape[1]
    p.run()

if __name__ == '__main__':
    test_inversion_pipeline()
