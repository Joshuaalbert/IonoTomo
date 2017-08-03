import numpy as np
from ionotomo.geometry.tri_cubic import TriCubic
from ionotomo.astro.real_data import DataPack
from ionotomo.plotting.plot_tools import plot_datapack
from ionotomo.geometry.calc_rays import calc_rays
from ionotomo.inversion.forward_equation import forward_equation
import numpy as np
import os
import logging as log
from dask import delayed
from dask.distributed import Client

def run(output_folder):
    output_folder = os.path.join(os.getcwd(),output_folder)
    tci_folder = os.path.join(output_folder,"ionospheres")

    log.basicConfig(filename=os.path.join(output_folder,"log"),format='%(asctime)s %(levelname)s:%(message)s', level=log.DEBUG)
    log.info("Using output folder {}".format(output_folder))
    plot_folder = os.path.join(output_folder,'datapack_sim')
    try:
        os.makedirs(plot_folder)
    except:
        pass
    log.info("Plotting generated ionospheres")
    datapack_files = np.genfromtxt(os.path.join(output_folder,"info"),dtype='str',usecols=[4])
    tci_files = np.genfromtxt(os.path.join(output_folder,"info"),dtype='str',usecols=[5])
    plot_dsk = []
    for datapack_file,tci_file in zip(datapack_files,tci_files):
    #    file = str(file,'utf-8')
        ne_tci = TriCubic(filename=tci_file)
        datapack = DataPack(filename=datapack_file)
        antennas,antenna_labels = datapack.get_antennas(ant_idx = -1)
        patches, patch_names = datapack.get_directions(dir_idx = -1)
        times,timestamps = datapack.get_times(time_idx=-1)
        Na = len(antennas)
        Nt = len(times)
        Nd = len(patches)  
        fixtime = times[Nt>>1]
        phase = datapack.get_center_direction()
        array_center = datapack.radio_array.get_center()
        rays = calc_rays(antennas,patches,times, array_center, fixtime, phase, ne_tci, datapack.radio_array.frequency, True, 1000, 250)
        K_ne = np.mean(ne_tci.M)
        ne_tci.M = np.log(ne_tci.M/K_ne)
        m_tci = ne_tci
        i0 = 0
        g = forward_equation(rays,K_ne,m_tci,i0)
        datapack.set_dtec(g,ant_idx=-1,time_idx=-1,dir_idx=-1)
        datapack_plot = os.path.join(plot_folder,tci_file.replace("ionosphere","datapack").split(os.sep)[-1].split('.hdf5')[0])
        plot_datapack(datapack,ant_idx=-1,time_idx=-1,dir_idx=-1,figname=datapack_plot)
        datapack.save(datapack_file)

if __name__=='__main__':
    run('output')
