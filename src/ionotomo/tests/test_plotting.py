from ionotomo.geometry.tri_cubic import TriCubic
import numpy as np
import os

from ionotomo.plotting import *

def test_plot_tci():
    vec = np.linspace(-1,1,100)
    x,y,z = np.meshgrid(vec,vec,vec,indexing='ij')
    r2 = x**2 + y**2 + z**2
    M = np.exp(-r2/0.1**2/2.)
    tci = TriCubic(vec,vec,vec,M)
    plot_tci(tci,show=True)

def test_plot_datapack():
    datapack = DataPack(filename="output/test/datapack_obs.hdf5")
    try:
        os.makedirs('output/test/plot_datapack')
    except:
        pass
    plot_datapack(datapack,ant_idx=-1,time_idx=[0,1,2,3], dir_idx=-1,figname=None)#'output/test/plot_datapack/fig')

def test_transfer_patch_data():
    if os.path.isfile('test_data/WendysBootes.npz'):
        transfer_patch_data(infoFile='test_data/WendysBootes.npz', 
                      data_folder='test_data/', 
                      hdf5Out='test_data/dtecData.hdf5')

def test_prepare_datapack():
    
    datapack = prepare_datapack('test_data/dtecData.hdf5',timeStart=0,timeEnd=-1,
                           array_file='arrays/lofar.hba.antenna.cfg')
    datapack.flag_antennas(['CS007HBA1','CS007HBA0','CS013HBA0','CS013HBA1'])
    datapack.set_reference_antenna(datapack.antenna_labels[0])
    #'CS501HBA1'
    datapack.save("output/test/datapack_obs.hdf5")


