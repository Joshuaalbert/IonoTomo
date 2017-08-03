
# coding: utf-8

# In[1]:

'''Get the number of pierce points within 1/2 L_ne of each voxel.
Heuristic: 5 is enough.
'''
from real_data import DataPack
import numpy as np
import astropy.units as au
from uvw_frame import UVW
from tri_cubic import TriCubic
from dask import delayed
import dask.array as da

import astropy.units as au
import astropy.time as at
import astropy.coordinates as ac

def precondition(ne_tci, datapack,ant_idx=-1, dir_idx=-1, time_idx = [0]):
    '''given ants, dirs in uvw frame an ne_tci give precondition array shaped like ne_tci.M'''
    antennas,antenna_labels = datapack.get_antennas(ant_idx = ant_idx)
    patches, patch_names = datapack.get_directions(dir_idx = dir_idx)
    times,timestamps = datapack.get_times(time_idx=time_idx)
    Na = len(antennas)
    Nt = len(times)
    Nd = len(patches) 
    fixtime = times[Nt>>1]
    phase = datapack.get_center_direction()
    array_center = datapack.radio_array.get_center()
    uvw = UVW(location = array_center.earth_location, obstime=fixtime,phase = phase)
    X,Y,Z = np.meshgrid(ne_tci.xvec,ne_tci.yvec,ne_tci.zvec,indexing='ij')
    coords = ac.SkyCoord(X.flatten()*au.km,Y.flatten()*au.km,Z.flatten()*au.km,frame=uvw).transform_to('itrs').earth_location.to_geodetic('WGS84')
    heights = coords[2].to(au.km).value.reshape(X.shape)#height in geodetic
    M = ne_tci.get_shaped_array()
    M *= 0
    M += 1.
    M[heights > 450.] *= 0.01
    M[heights < 60.] *= 0.01
    out = TriCubic(ne_tci.xvec,ne_tci.yvec, ne_tci.zvec, M)
    return out
    L_pre = L_ne/2.
    xvec = ne_tci.xvec
    yvec = ne_tci.yvec
    zvec = ne_tci.zvec
    X,Y = np.meshgrid(xvec,yvec,indexing='ij')
    N = np.size(X)
    X_ = da.from_array(X.flatten(),chunks=(N>>2,))
    Y_ = da.from_array(Y.flatten(),chunks=(N>>2,))
    ants = da.from_array(ants_uvw,chunks=(ants_uvw.shape[0]>>1,1))
    dirs = da.from_array(dirs_uvw,chunks=(dirs_uvw.shape[0]>>1,1))
    @delayed
    def do_plane(ants,dirs,z,X_,Y_,L_pre,shape):
        scale = z/dirs[:,2]
        pir_u = da.add.outer(ants[:,0],dirs[:,0]*scale).flatten()
        pir_v = da.add.outer(ants[:,1],dirs[:,1]*scale).flatten()
        out = da.sum(da.exp(-(da.subtract.outer(X_,pir_u)**2 + da.subtract.outer(Y_,pir_v)**2)/L_pre**2/2.),axis=1).reshape(shape)
        return out
    planes = []
    i = 0
    while i < len(zvec):
        planes.append(do_plane(ants_uvw,dirs_uvw,zvec[i],X_,Y_,L_pre,X.shape))
        i += 1
    arrays = [da.from_delayed(plane,(xvec.size,yvec.size),dtype=np.double) for plane in planes]
    F0 = da.stack(arrays,axis=-1)
    #F0 = F0 / da.max(F0)
    F0[F0 > 5.] = 5.
    F0 /= 5.
    return F0.compute()

def test_precondition():
    from time import clock
    datapack = DataPack(filename="output/simulated/dataobs.hdf5").clone()
    ne_tci = TriCubic(filename="output/simulated/ne_model-0.hdf5").copy()
    antennas,antenna_labels = datapack.get_antennas(ant_idx = -1)
    patches, patch_names = datapack.get_directions(dir_idx=-1)
    times,timestamps = datapack.get_times(time_idx=[0])
    Na = len(antennas)
    Nt = len(times)
    Nd = len(patches)  
    fixtime = times[Nt>>1]
    #print("Fixing frame at {}".format(fixtime.isot))
    phase = datapack.get_center_direction()
    uvw = UVW(location = datapack.radio_array.get_center().earth_location,obstime = fixtime,phase = phase)
    ants_uvw = antennas.transform_to(uvw).cartesian.xyz.to(au.km).value.transpose()
    dirs_uvw = patches.transform_to(uvw).cartesian.xyz.value.transpose()
    t1 =clock()
    F0 = precondition(ants_uvw,dirs_uvw,ne_tci,L_ne=15.)
    print("Time for preconditioning build {}".format(clock()-t1))
    from PlotTools import animate_tci_slices
    F0TCI = TriCubic(ne_tci.xvec,ne_tci.yvec,ne_tci.zvec,F0)
    animate_tci_slices(F0TCI,"output/test/infomatrix",num_seconds=10.)

if __name__ == '__main__':
    test_precondition()


# In[ ]:



