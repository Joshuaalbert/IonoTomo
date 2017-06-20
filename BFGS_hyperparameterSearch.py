
# coding: utf-8

# In[ ]:

from BFGS_dask import createBFGSDask
from RealData import DataPack
from AntennaFacetSelection import selectAntennaFacets
from dask import get
import pylab as plt
import numpy as np
import h5py

if __name__=='__main__':
    i0 = 0
    datapack = DataPack(filename="output/test/simulate/simulate_3/datapackSim.hdf5")
    datapackSel = selectAntennaFacets(25, datapack, antIdx=-1, dirIdx=-1, timeIdx = np.arange(1))
    sizeCell = 5.
    niter = 30
    S = []
    for L_ne in [10.,20.,30.,40.,50.,60.,70.,80.,90.,100.]:
        outputfolder = 'output/search/L_ne/{}'.format(L_ne)
        dsk = createBFGSDask(False,outputfolder,niter,datapack,L_ne,sizeCell,i0, antIdx=-1, dirIdx=-1, timeIdx = np.arange(1))
        get(dsk,['pull_m{:d}'.format(niter)])
        with h5py.File("{}/state".format(outputfolder),'r') as state:
            assert '/{}/S'.format(niter) in state, "Failed to get result"
            S.append(state['/{}/S'.format(niter)])
    plt.plot([10.,20.,30.,40.,50.,60.,70.,80.,90.,100.], S)
    plt.show()
        

