
# coding: utf-8

# In[33]:

'''Calculate rays efficiently. Output is array of shape 
[Na, Nt, Nd, N, N ,N, N] where N is integration resolution.
these correspond to antIdx, timeIdx, dirIdx, x,y,z,s'''

import dask.array as da
from FermatClass import Fermat
import numpy as np
from dask import delayed
from PointingFrame import Pointing
from RealData import DataPack
from TricubicInterpolation import TriCubic

import astropy.units as au
import astropy.coordinates as ac
import astropy.time as at

from dask.multiprocessing import get


#@delayed
def splitPatches(antennas,times, patchDir, arrayCenter, fixtime, phase):
    '''get origins and directions in shape [Na, Nt, 3]'''
    Na = len(antennas)
    Nt = len(times)
    origins = np.zeros([Na,Nt,3],dtype=np.double)
    directions = np.zeros([Na,Nt,3],dtype=np.double)
    j = 0
    while j < Nt:
        time = times[j]
        pointing = Pointing(location = arrayCenter.earth_location,
                            obstime = time, fixtime = fixtime, phase = phase)
        direction = patchDir.transform_to(pointing).cartesian.xyz.value.transpose()
        ants = antennas.transform_to(pointing).cartesian.xyz.to(au.km).value.transpose()
        origins[:,j,:] += ants#Na x 3 + Na x 3
        directions[:,j,:] += direction#Na x 3 + 1 x 3
        j += 1
    return origins, directions

#@delayed
def splitAntennas(times, patches, antenna, arrayCenter, fixtime, phase):
    '''get origins and directions in shape [Nt, Nd, 3]'''
    Nd = len(patches)
    Nt = len(times)
    origins = np.zeros([Nt, Nd,3],dtype=np.double)
    directions = np.zeros([Nt, Nd,3],dtype=np.double)
    j = 0
    while j < Nt:
        time = times[j]
        pointing = Pointing(location = arrayCenter.earth_location,
                            obstime = time, fixtime = fixtime, phase = phase)
        dirs = patches.transform_to(pointing).cartesian.xyz.value.transpose()
        origin = antenna.transform_to(pointing).cartesian.xyz.to(au.km).value.transpose()
        origins[j,:, :] += origin#Nd x 3 + 1 x 3
        directions[j, :, :] += dirs#Nd x 3 + Nd x 3
        j += 1
    return origins, directions
    
#@delayed
def castRay(batch, fermat, tmax, N):
    '''Calculates TEC for all given rays.
    ``origins`` is an array with coordinates in prefered frame
    ``diretions`` is an array with coordinates in prefered frame
    ``tmax`` is the length of rays to use.
    ``neTCI`` is the tri cubic interpolator
    return list of ray trajectories'''
    origins, directions = batch
    #origins.shape = [Na,Nt,3] or [Nt,Nd,3]
    #add x,y,z,s
    shape = list(origins.shape[:-1]) + [4,N]
    rays = np.zeros(shape,dtype=np.double)
    #fermat = Fermat(neTCI=neTCI,frequency = frequency,type='z',straightLineApprox=straightLineApprox)
    #print("Casting {} rays".format(shape[0]*shape[1]))
    i = 0
    while i < shape[0]:
        j = 0
        while j < shape[1]:
            origin = origins[i,j,:]
            direction = directions[i,j,:]
            x,y,z,s = fermat.integrateRay(origin,direction,tmax,N=N)
            rays[i,j,0,:] = x
            rays[i,j,1,:] = y
            rays[i,j,2,:] = z
            rays[i,j,3,:] = s
            j += 1
        i += 1
    return rays

def mergeRays(*rayBundles):
    #print("Merging")
    out = []
    for rays in rayBundles:
        out.append(rays)
    return out

def calcRays_dask(antennas,patches,times, arrayCenter, fixtime, phase, neTCI, frequency,  straightLineApprox,tmax, N= None):
    '''Do rays in parallel processes batch by directions'''
    #from dask.distributed import Client
    #client = Client()
    if N is None:
        N = neTCI.nz
    Na = len(antennas)
    Nt = len(times)
    Nd = len(patches)
    print("Casting rays: {}".format(Na*Nt*Nd))
    #rays = np.zeros([Na, Nt, Nd, 4, N], dtype= np.double)
    #split over smaller to make largest workloads
    if Na < Nd:
        print("spliting over antennas")
        batches = [delayed(splitAntennas)(times, patches, antenna, arrayCenter, fixtime, phase) for antenna in antennas]
    else:
        print("splitting over directions")
        batches = [delayed(splitPatches)(antennas,times, patchDir, arrayCenter, fixtime, phase) for patchDir in patches]
    fermat = Fermat(neTCI=neTCI,frequency = frequency,type='z',straightLineApprox=straightLineApprox)
    if Na < Nd:
        #[Nt,Nd,4,N]
        rays = da.stack([da.from_delayed(delayed(castRay)(batch, fermat, tmax, N),
                                         (Nt,Nd,4,N),dtype=np.double) for batch in batches],axis=0)
    else:
        #[Na,Nt,4,N]
        rays = da.stack([da.from_delayed(delayed(castRay)(batch, fermat, tmax, N),
                                         (Na,Nt,4,N),dtype=np.double) for batch in batches],axis=2)
    rays = rays.compute()
    #print(rays)
    return rays

def calcRays(antennas,patches,times, arrayCenter, fixtime, phase, neTCI, frequency,  straightLineApprox,tmax, N=None):
    '''Do rays in parallel processes batch by directions'''
    if N is None:
        N = neTCI.nz
    Na = len(antennas)
    Nt = len(times)
    Nd = len(patches)
    print("Casting rays: {}".format(Na*Nt*Nd))
    #rays = np.zeros([Na, Nt, Nd, 4, N], dtype= np.double)
    #split over smaller to make largest workloads
    if Na < Nd:
        print("spliting over antennas")
        batches = [splitAntennas(times, patches, antenna, arrayCenter, fixtime, phase) for antenna in antennas]
    else:
        print("splitting over directions")
        batches = [splitPatches(antennas,times, patchDir, arrayCenter, fixtime, phase) for patchDir in patches]
    fermat = Fermat(neTCI=neTCI,frequency = frequency,type='z',straightLineApprox=straightLineApprox)
    if Na < Nd:
        #[Nt,Nd,4,N]
        rays = np.stack([castRay(batch, fermat, tmax, N) for batch in batches],axis=0)
    else:
        #[Na,Nt,4,N]
        rays = np.stack([castRay(batch, fermat, tmax, N) for batch in batches],axis=2)
    #print(rays)
    return rays

#def plotRays(antIdx=-1,timeIdx=-1, dirIdx = -1)
  
def test_calcRays():
    datapack = DataPack(filename="output/simulated/dataobs.hdf5").clone()
    neTCI = TriCubic(filename="output/simulated/neModel-0.hdf5").copy()
    antennas,antennaLabels = datapack.get_antennas(antIdx = -1)
    patches, patchNames = datapack.get_directions(dirIdx=np.arange(8))
    times,timestamps = datapack.get_times(timeIdx=[0])
    Na = len(antennas)
    Nt = len(times)
    Nd = len(patches)  
    fixtime = times[Nt>>1]
    phase = datapack.getCenterDirection()
    arrayCenter = datapack.radioArray.getCenter()
    print("Calculating rays...")
    rays1 = calcRays(antennas,patches,times, arrayCenter, fixtime, phase, neTCI, 120e6, True, 1000, 1000)
    rays2 = calcRays_dask(antennas,patches,times, arrayCenter, fixtime, phase, neTCI, 120e6, True, 1000, 1000)
    assert np.all(rays1==rays2),"Not same result"
    
if __name__ == '__main__':
    #at the moment only dask is faster for 80+ directions, unless parallelize over antennas
    test_calcRays()
    
    
    

