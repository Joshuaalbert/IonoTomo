from ionotomo.astro.real_data import generate_example_datapack, phase_screen_datapack
from ionotomo.astro.radio_array import generate_example_radio_array
from ionotomo.astro.antenna_facet_selection import select_antennas
from ionotomo.astro.simulate_observables import *
from ionotomo.inversion.initial_model import *
import logging as log
import os
from time import clock
from dask.threaded import get
from functools import partial

def run(output_folder):
    output_folder = os.path.join(os.getcwd(),output_folder)

    try:
        os.makedirs(output_folder)
    except:
        pass

    log.basicConfig(filename=os.path.join(output_folder,"log"),format='%(asctime)s %(levelname)s:%(message)s', level=log.DEBUG)
    log.info("Using output folder {}".format(output_folder))
    datapack_folder = os.path.join(output_folder,"datapacks")

    try:
        os.makedirs(datapack_folder)
    except:
        pass
    datapack_ = DataPack(filename="../rvw_data_analysis/rvw_datapack.hdf5")
    datapack_screen_ = phase_screen_datapack(10,datapack=datapack_)
    log.info("Generating a number of ionospheres")
#    #using lofar configuration generate for a number of random pointings and times
#    radio_array = generate_example_radio_array(config='lofar')
    info_file = os.path.join(output_folder,"info")
    if os.path.exists(info_file) and os.path.isfile(info_file):
        info = open(info_file,'a')
    else:
        info = open(info_file,'w')
        info.write("#time_idx timestamp factor corr\n")
    times,timestamps = datapack_.get_times(time_idx=range(100))

    dsk = {}
   
    Nt = len(times)
    for factor in [2.,4.,8.,16.]:
        for corr in [10.,20.,40.,70.]:
            datapack = datapack_.clone()
            datapack_screen = datapack_screen_.clone()
            dsk['datapack_{}_{}_{}'.format(factor,corr,-1)] = datapack
            dsk['datapack_screen_{}_{}_{}'.format(factor,corr,-1)] = datapack_screen
            for j in range(Nt):
                seed = j+1000
                log.info("Simulating turbulent ionosphere: factor {} corr {} time {}\n".format(factor,corr,timestamps[j]))
#                ne_tci = create_turbulent_model(datapack,factor=factor,corr=corr,seed=seed,ant_idx = -1, time_idx = [j], dir_idx = -1, zmax = 1000.,spacing=5.,padding=20)
                dsk['ne_tci_{}_{}_{}'.format(factor,corr,j)] = (partial(create_turbulent_model,factor=factor,corr=corr,seed=seed,ant_idx = -1, time_idx = [j], dir_idx = -1, zmax = 1000.,spacing=5.,padding=20),datapack)

#                datapack = simulate_phase(datapack,ne_tci=ne_tci,num_threads=1,datafolder=None,
#                     ant_idx=-1,time_idx=[j],dir_idx=-1,freq_idx=-1,do_plot_datapack=False,flag_remaining=False)
                dsk['datapack_{}_{}_{}'.format(factor,corr,j)] = (lambda datapack, ne_tci : simulate_phase(datapack, ne_tci=ne_tci, num_threads=1, datafolder=None,
                     ant_idx=-1, time_idx=[j],dir_idx=-1,freq_idx=-1,do_plot_datapack=False,flag_remaining=False), 'datapack_{}_{}_{}'.format(factor,corr,j-1), 'ne_tci_{}_{}_{}'.format(factor,corr,j))
#                datapack_screen = simulate_phase(datapack_screen,ne_tci=ne_tci,num_threads=1,datafolder=None,
#                     ant_idx=-1,time_idx=[j],dir_idx=-1,freq_idx=-1,do_plot_datapack=False,flag_remaining=False)
                dsk['datapack_screen_{}_{}_{}'.format(factor,corr,j)] = (lambda datapack, ne_tci : simulate_phase(datapack,ne_tci=ne_tci,num_threads=1,datafolder=None,
                     ant_idx=-1,time_idx=[j],dir_idx=-1,freq_idx=-1,do_plot_datapack=False,flag_remaining=False), 'datapack_screen_{}_{}_{}'.format(factor,corr,j-1), 'ne_tci_{}_{}_{}'.format(factor,corr,j))

                info.write("{} {} {} {}\n".format(j, timestamps[j], factor, corr))
            objectives = ['datapack_{}_{}_{}'.format(factor,corr,Nt-1),'datapack_screen_{}_{}_{}'.format(factor,corr,Nt-1)]
            from dask.callbacks import Callback
            class PrintKeys(Callback):
                def _pretask(self, key, dask, state):
                    """Print the key of every task as it's started"""
                    print("Computing: {0}!".format(repr(key)))
            with PrintKeys():
                datapack,datapack_screen = get(dsk,objectives,num_workers=None)
            datapack.save(os.path.join(datapack_folder,"datapack_factr{:d}_corr{:d}.hdf5".format(int(factor),int(corr))))
            datapack_screen.save(os.path.join(datapack_folder,"datapack_screen_factr{:d}_corr{:d}.hdf5".format(int(factor),int(corr))))
    info.close()

if __name__=='__main__':
    run("output")
