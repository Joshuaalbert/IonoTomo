
# coding: utf-8

# In[2]:

import numpy as np
from InfoCompleteness import precondition
from TricubicInterpolation import TriCubic
from RealData import DataPack
from AntennaFacetSelection import selectAntennaFacets
from CalcRays import calcRays
from LineSearch import lineSearch
from InitialModel import createInitialModel, createTurbulentlModel
from ForwardEquation import forwardEquation, forwardEquation_dask
from UVWFrame import UVW
from Gradient import computeGradient_dask,computeGradient

import astropy.units as au
    
def bfgs(datapack,L_ne,sizeCell,i0):
    antennas,antennaLabels = datapack.get_antennas(antIdx = -1)
    patches, patchNames = datapack.get_directions(dirIdx = -1)
    times,timestamps = datapack.get_times(timeIdx=[0])
    datapack.setReferenceAntenna(antennaLabels[i0])
    dobs = datapack.get_dtec(antIdx = -1, timeIdx = [0], dirIdx = -1)
    CdCt = (0.15*np.abs(dobs))**2
    Na = len(antennas)
    Nt = len(times)
    Nd = len(patches)  
    fixtime = times[Nt>>1]
    phase = datapack.getCenterDirection()
    arrayCenter = datapack.radioArray.getCenter()
    neTCI = createInitialModel(datapack,antIdx = -1, timeIdx = -1, dirIdx = -1, zmax = 1000.,spacing=sizeCell)
    rays = calcRays(antennas,patches,times, arrayCenter, fixtime, phase, neTCI, datapack.radioArray.frequency, True, 1000., neTCI.nz)
    K_ne = np.mean(neTCI.m)
    mTCI = neTCI.copy()
    mTCI.m /= K_ne
    np.log(mTCI.m,out=mTCI.m)
    g = forwardEquation_dask(rays,K_ne,mTCI,i0)
    Nkernel = int(float(L_ne*3)/sizeCell)
    uvw = UVW(location = datapack.radioArray.getCenter().earth_location,obstime = fixtime,phase = phase)
    ants_uvw = antennas.transform_to(uvw).cartesian.xyz.to(au.km).value.transpose()
    dirs_uvw = patches.transform_to(uvw).cartesian.xyz.value.transpose()
    F0 = np.ones([mTCI.nx,mTCI.ny,mTCI.nz])#precondition(ants_uvw,dirs_uvw,neTCI,L_ne=L_ne)
    F = F0.copy()
    mPrior = mTCI.getShapedArray()
    iter = 0
    ## persistent variables
    beta = []
    dm = []#m_n+1 - m_n
    gamma = []
    v = []#Fn
    while iter < 10:
        grad = computeGradient_dask(rays, g, dobs, i0, K_ne, mTCI, mPrior, CdCt, 1, Nkernel, sizeCell)
        phi = F*grad
        epsilon_n = lineSearch(rays,K_ne,mTCI,i0,phi,g,dobs,CdCt,plot=True)
        dm = epsilon_n*phi
        mTCI = TriCubic(mTCI.xvec,mTCI.yvec,mTCI.zvec,mTCI.getShapedArray() - dm)
        g = forwardEquation_dask(rays,K_ne,mTCI,i0)
        print("mean abs(dm): {}".format(np.mean(np.abs(dm))))
        ## update F not
        iter += 1
    return mTCI
        
def test_bfgs():
    i0 = 0
    datapack = DataPack(filename="output/test/datapackObs.hdf5")
    datapackSel = selectAntennaFacets(10, datapack, antIdx=-1, dirIdx=-1, timeIdx = [0])
    pertTCI = createTurbulentlModel(datapackSel,antIdx = -1, timeIdx = -1, dirIdx = -1, zmax = 1000.)
    L_ne = 15.
    sizeCell = 5.
    bfgs(datapackSel,L_ne,sizeCell,i0)
    
if __name__ == '__main__':
    test_bfgs()


# In[20]:




# In[ ]:



