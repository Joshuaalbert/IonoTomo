
# coding: utf-8

# In[33]:

'''Calculate rays efficiently. Output is array of shape 
[Na, Nt, Nd, N, N ,N, N] where N is integration resolution.
these correspond to ant_idx, time_idx, dir_idx, x,y,z,s'''

import dask.array as da
from fermat import Fermat
import numpy as np
from dask import delayed
from pointing_frame import Pointing
from real_data import DataPack
from tri_cubic import TriCubic

import astropy.units as au
import astropy.coordinates as ac
import astropy.time as at

from dask.multiprocessing import get


#@delayed
def split_patches(antennas,times, patch_dir, array_center, fixtime, phase):
    '''get origins and directions in shape [Na, Nt, 3]'''
    Na = len(antennas)
    Nt = len(times)
    origins = np.zeros([Na,Nt,3],dtype=np.double)
    directions = np.zeros([Na,Nt,3],dtype=np.double)
    j = 0
    while j < Nt:
        time = times[j]
        pointing = Pointing(location = array_center.earth_location,
                            obstime = time, fixtime = fixtime, phase = phase)
        direction = patch_dir.transform_to(pointing).cartesian.xyz.value.transpose()
        ants = antennas.transform_to(pointing).cartesian.xyz.to(au.km).value.transpose()
        origins[:,j,:] += ants#Na x 3 + Na x 3
        directions[:,j,:] += direction#Na x 3 + 1 x 3
        j += 1
    return origins, directions

#@delayed
def split_antennas(times, patches, antenna, array_center, fixtime, phase):
    '''get origins and directions in shape [Nt, Nd, 3]'''
    Nd = len(patches)
    Nt = len(times)
    origins = np.zeros([Nt, Nd,3],dtype=np.double)
    directions = np.zeros([Nt, Nd,3],dtype=np.double)
    j = 0
    while j < Nt:
        time = times[j]
        pointing = Pointing(location = array_center.earth_location,
                            obstime = time, fixtime = fixtime, phase = phase)
        dirs = patches.transform_to(pointing).cartesian.xyz.value.transpose()
        origin = antenna.transform_to(pointing).cartesian.xyz.to(au.km).value.transpose()
        origins[j,:, :] += origin#Nd x 3 + 1 x 3
        directions[j, :, :] += dirs#Nd x 3 + Nd x 3
        j += 1
    return origins, directions
    
#@delayed
def cast_ray(batch, fermat, tmax, N):
    '''Calculates TEC for all given rays.
    ``origins`` is an array with coordinates in prefered frame
    ``diretions`` is an array with coordinates in prefered frame
    ``tmax`` is the length of rays to use.
    ``ne_tci`` is the tri cubic interpolator
    return list of ray trajectories'''
    origins, directions = batch
    #origins.shape = [Na,Nt,3] or [Nt,Nd,3]
    #add x,y,z,s
    shape = list(origins.shape[:-1]) + [4,N]
    rays = np.zeros(shape,dtype=np.double)
    #fermat = Fermat(ne_tci=ne_tci,frequency = frequency,type='z',straight_line_approx=straight_line_approx)
    #print("Casting {} rays".format(shape[0]*shape[1]))
    i = 0
    while i < shape[0]:
        j = 0
        while j < shape[1]:
            origin = origins[i,j,:]
            direction = directions[i,j,:]
            x,y,z,s = fermat.integrate_ray(origin,direction,tmax,N=N)
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

def calcRays_dask(antennas,patches,times, array_center, fixtime, phase, ne_tci, frequency,  straight_line_approx,tmax, N= None):
    '''Do rays in parallel processes batch by directions'''
    #from dask.distributed import Client
    #client = Client()
    if N is None:
        N = ne_tci.nz
    Na = len(antennas)
    Nt = len(times)
    Nd = len(patches)
    print("Casting rays: {}".format(Na*Nt*Nd))
    #rays = np.zeros([Na, Nt, Nd, 4, N], dtype= np.double)
    #split over smaller to make largest workloads
    if Na < Nd:
        print("spliting over antennas")
        batches = [delayed(split_antennas)(times, patches, antenna, array_center, fixtime, phase) for antenna in antennas]
    else:
        print("splitting over directions")
        batches = [delayed(split_patches)(antennas,times, patch_dir, array_center, fixtime, phase) for patch_dir in patches]
    fermat = Fermat(ne_tci=ne_tci,frequency = frequency,type='z',straight_line_approx=straight_line_approx)
    if Na < Nd:
        #[Nt,Nd,4,N]
        rays = da.stack([da.from_delayed(delayed(cast_ray)(batch, fermat, tmax, N),
                                         (Nt,Nd,4,N),dtype=np.double) for batch in batches],axis=0)
    else:
        #[Na,Nt,4,N]
        rays = da.stack([da.from_delayed(delayed(cast_ray)(batch, fermat, tmax, N),
                                         (Na,Nt,4,N),dtype=np.double) for batch in batches],axis=2)
    rays = rays.compute()
    #print(rays)
    return rays

def calcRays(antennas,patches,times, array_center, fixtime, phase, ne_tci, frequency,  straight_line_approx,tmax, N=None):
    '''Do rays in parallel processes batch by directions'''
    if N is None:
        N = ne_tci.nz
    Na = len(antennas)
    Nt = len(times)
    Nd = len(patches)
    print("Casting rays: {}".format(Na*Nt*Nd))
    #rays = np.zeros([Na, Nt, Nd, 4, N], dtype= np.double)
    #split over smaller to make largest workloads
    if Na < Nd:
        print("spliting over antennas")
        batches = [split_antennas(times, patches, antenna, array_center, fixtime, phase) for antenna in antennas]
    else:
        print("splitting over directions")
        batches = [split_patches(antennas,times, patch_dir, array_center, fixtime, phase) for patch_dir in patches]
    fermat = Fermat(ne_tci=ne_tci,frequency = frequency,type='z',straight_line_approx=straight_line_approx)
    if Na < Nd:
        #[Nt,Nd,4,N]
        rays = np.stack([cast_ray(batch, fermat, tmax, N) for batch in batches],axis=0)
    else:
        #[Na,Nt,4,N]
        rays = np.stack([cast_ray(batch, fermat, tmax, N) for batch in batches],axis=2)
    #print(rays)
    return rays

#def plotRays(ant_idx=-1,time_idx=-1, dir_idx = -1)
  
def test_calcRays():
    datapack = DataPack(filename="output/simulated/dataobs.hdf5").clone()
    ne_tci = TriCubic(filename="output/simulated/neModel-0.hdf5").copy()
    antennas,antenna_labels = datapack.get_antennas(ant_idx = -1)
    patches, patch_names = datapack.get_directions(dir_idx=np.arange(8))
    times,timestamps = datapack.get_times(time_idx=[0])
    Na = len(antennas)
    Nt = len(times)
    Nd = len(patches)  
    fixtime = times[Nt>>1]
    phase = datapack.get_center_direction()
    array_center = datapack.radio_array.get_center()
    print("Calculating rays...")
    rays1 = calcRays(antennas,patches,times, array_center, fixtime, phase, ne_tci, 120e6, True, 1000, 1000)
    rays2 = calcRays_dask(antennas,patches,times, array_center, fixtime, phase, ne_tci, 120e6, True, 1000, 1000)
    assert np.all(rays1==rays2),"Not same result"
    
if __name__ == '__main__':
    #at the moment only dask is faster for 80+ directions, unless parallelize over antennas
    test_calcRays()
    
    
    

