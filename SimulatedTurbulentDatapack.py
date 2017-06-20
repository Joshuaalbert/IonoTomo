
# coding: utf-8

# In[2]:

import numpy as np
from ForwardEquation import forwardEquation, forwardEquation_dask
from CalcRays import calcRays,calcRays_dask

def simulateDatapack(templateDatapack,neModelTurbulentTCI,i0, antIdx = -1, timeIdx=[0], dirIdx=-1):
    antennas,antennaLabels = templateDatapack.get_antennas(antIdx = antIdx)
    patches, patchNames = templateDatapack.get_directions(dirIdx = dirIdx)
    times,timestamps = templateDatapack.get_times(timeIdx=timeIdx)
    Na = len(antennas)
    Nt = len(times)
    Nd = len(patches)  
    fixtime = times[Nt>>1]
    phase = templateDatapack.getCenterDirection()
    arrayCenter = templateDatapack.radioArray.getCenter()
    rays = calcRays(antennas,patches,times, arrayCenter, fixtime, phase, neModelTurbulentTCI, templateDatapack.radioArray.frequency, True, 1000, 1000)
    mTCI = neModelTurbulentTCI.copy()
    K_ne = np.mean(mTCI.m)
    mTCI.m /= K_ne
    np.log(mTCI.m,out=mTCI.m)
    g = forwardEquation_dask(rays,K_ne,mTCI,i0)
    datapack = templateDatapack.clone()
    datapack.set_dtec(g,antIdx = antIdx, timeIdx=timeIdx, dirIdx = dirIdx, refAnt=antennaLabels[i0])
    return datapack

def test_simulateDatapack():
    from InitialModel import createTurbulentlModel, createInitialModel
    from RealData import DataPack
    from AntennaFacetSelection import selectAntennaFacets
    from RealData import plotDataPack
    from PlotTools import animateTCISlices
    import os
    datapack = DataPack(filename="output/test/datapackObs.hdf5")
    for i in range(1,5):
        try:
            os.makedirs("output/test/simulate/simulate_{}/fig".format(i))
        except:
            pass
        N = 15
        #datapackSel = selectAntennaFacets(N, datapack, antIdx=-1, dirIdx=-1, timeIdx = [0])
        turbTCI = createTurbulentlModel(datapack,antIdx = -1, timeIdx = [0], dirIdx = -1, zmax = 1000.,spacing=5.)
        nePriorTCI = createInitialModel(datapack,antIdx = -1, timeIdx = [0], dirIdx = -1, zmax = 1000.,spacing=5.)
        turbTCI.save("output/test/simulate/simulate_{}/neModel.hdf5".format(i))
        datapackSim = simulateDatapack(datapack,turbTCI,0, antIdx = -1, timeIdx=[0], dirIdx=-1)
        datapackSim.save("output/test/simulate/simulate_{}/datapackSim.hdf5".format(i))
        plotDataPack(datapackSim,antIdx=-1,timeIdx=[0], dirIdx=-1,figname="output/test/simulate/simulate_{}/dobs".format(i))
        turbTCI.m -= nePriorTCI.m
        animateTCISlices(turbTCI,"output/test/simulate/simulate_{}/fig".format(i))

if __name__ == '__main__':
    test_simulateDatapack()


# In[ ]:



