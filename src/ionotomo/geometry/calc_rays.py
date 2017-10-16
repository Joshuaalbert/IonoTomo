'''Calculate rays efficiently. Output is array of shape 
[Na, Nt, Nd, N, N ,N, N] where N is integration resolution.
these correspond to ant_idx, time_idx, dir_idx, x,y,z,s'''
import numpy as np
from dask import delayed
import dask.array as da

from ionotomo.inversion.fermat import Fermat
from ionotomo.astro.frames.pointing_frame import Pointing
from ionotomo.astro.real_data import DataPack
from ionotomo.geometry.tri_cubic import TriCubic
from ionotomo.inversion.solution import Solution
import astropy.units as au
import astropy.coordinates as ac
import astropy.time as at

from dask.distributed import Client

import logging as log



def split_patches(antennas,times, patch_dir, array_center, fixtime, phase):
    '''get origins and directions in shape [Na, Nt, 3]'''
    Na = len(antennas)
    Nt = len(times)
    origins = np.zeros([Na,Nt,3],dtype=float)
    directions = np.zeros([Na,Nt,3],dtype=float)
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
    

def cast_ray(batch, fermat, tmax, N):
    '''Calculates ray trajectories.
    batch : list
        [origins, directions] where origins and directions are of shape 
        (Na, Nt, Nd, 3) in the frame of model.
    fermat : Fermat class object
        Contains the machinery to integrate rays in 3d space.
    tmax : float
        The distance from origin to place plane of sky
    N : int
        The number of segments equi partitioned along the direction of 
        observation to calculate ray points.
    '''
    origins, directions = batch
    Na, Nt, Nd, _ = origins.shape
    #origins.shape = [Na,Nt,3] or [Nt,Nd,3]
    #add x,y,z,s
    shape = (Na, Nt, Nd, 4, N)
    rays = np.zeros(shape, dtype=float)
    i = 0
    while i < shape[0]:
        j = 0
        while j < shape[1]:
            k = 0
            while k < shape[2]:
                origin = origins[i,j,k,:]
                direction = directions[i,j,k,:]
                x,y,z,s = fermat.integrate_ray(origin,direction,tmax,N=N)
                rays[i,j,k,0,:] = x
                rays[i,j,k,1,:] = y
                rays[i,j,k,2,:] = z
                rays[i,j,k,3,:] = s
                k += 1
            j += 1
        i += 1
    return rays

def merge_rays(*ray_bundles):
    #log.info("Merging")
    out = []
    for rays in ray_bundles:
        out.append(rays)
    return out

def calc_rays_dask(antennas,patches,times, array_center, fixtime, phase, ne_tci, frequency,  straight_line_approx,tmax, N= None, get=None):
    '''Do rays in parallel processes batch by directions'''
    pass 

def calc_rays(antennas,patches,times, array_center, fixtime, phase, ne_tci, frequency,  straight_line_approx,tmax, N=None):
    '''Do rays in parallel processes batch by directions'''
    if N is None:
        N = ne_tci.nz
    #print(antennas,patches,times, array_center, fixtime, phase, ne_tci, frequency,  straight_line_approx,tmax, N)
    Na = len(antennas)
    Nt = len(times)
    Nd = len(patches)
    log.info("Casting rays: {}".format(Na*Nt*Nd))

    #rays = np.zeros([Na, Nt, Nd, 4, N], dtype= np.double)
    #split over smaller to make largest workloads

    origins = np.zeros([Na, Nt, Nd,3],dtype=np.double)
    directions = np.zeros([Na, Nt, Nd,3],dtype=np.double)
    j = 0
    while j < Nt:
        time = times[j]
        pointing = Pointing(location = array_center.earth_location,
                            obstime = time, fixtime = fixtime, phase = phase)
        # Nd x 3
        dirs = patches.transform_to(pointing).cartesian.xyz.value.transpose()
        # Na x 3
        origin = antennas.transform_to(pointing).cartesian.xyz.to(au.km).value.transpose()
        # Na x Nd x 3
        # Na x 1  x 3
        origins[:,j,:, :] += np.expand_dims(origin,1)
        # Na x Nd x 3
        #      Nd x 3
        directions[:,j, :, :] += dirs
        j += 1
    batch = (origins,directions)
    
    fermat = Fermat(ne_tci=ne_tci,frequency = frequency,type='z',straight_line_approx=straight_line_approx)
    rays = cast_ray(batch, fermat, tmax, N)
    
    return rays

