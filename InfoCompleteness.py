
# coding: utf-8

# In[1]:

'''Get the number of pierce points within 1/2 L_ne of each voxel.
Heuristic: 5 is enough.
'''
from RealData import DataPack
import numpy as np
import astropy.units as au
from UVWFrame import UVW
from TricubicInterpolation import TriCubic
from dask import delayed
import dask.array as da

import astropy.units as au
import astropy.time as at
import astropy.coordinates as ac

def precondition(neTCI, datapack,antIdx=-1, dirIdx=-1, timeIdx = [0]):
    '''given ants, dirs in uvw frame an neTCI give precondition array shaped like neTCI.M'''
    antennas,antennaLabels = datapack.get_antennas(antIdx = antIdx)
    patches, patchNames = datapack.get_directions(dirIdx = dirIdx)
    times,timestamps = datapack.get_times(timeIdx=timeIdx)
    Na = len(antennas)
    Nt = len(times)
    Nd = len(patches) 
    fixtime = times[Nt>>1]
    phase = datapack.getCenterDirection()
    arrayCenter = datapack.radioArray.getCenter()
    uvw = UVW(location = arrayCenter.earth_location, obstime=fixtime,phase = phase)
    X,Y,Z = np.meshgrid(neTCI.xvec,neTCI.yvec,neTCI.zvec,indexing='ij')
    coords = ac.SkyCoord(X.flatten()*au.km,Y.flatten()*au.km,Z.flatten()*au.km,frame=uvw).transform_to('itrs').earth_location.to_geodetic('WGS84')
    heights = coords[2].to(au.km).value.reshape(X.shape)#height in geodetic
    M = neTCI.getShapedArray()
    M *= 0
    M += 1.
    M[heights > 450.] *= 0.01
    M[heights < 60.] *= 0.01
    out = TriCubic(neTCI.xvec,neTCI.yvec, neTCI.zvec, M)
    return out
    L_pre = L_ne/2.
    xvec = neTCI.xvec
    yvec = neTCI.yvec
    zvec = neTCI.zvec
    X,Y = np.meshgrid(xvec,yvec,indexing='ij')
    N = np.size(X)
    X_ = da.from_array(X.flatten(),chunks=(N>>2,))
    Y_ = da.from_array(Y.flatten(),chunks=(N>>2,))
    ants = da.from_array(ants_uvw,chunks=(ants_uvw.shape[0]>>1,1))
    dirs = da.from_array(dirs_uvw,chunks=(dirs_uvw.shape[0]>>1,1))
    @delayed
    def doPlane(ants,dirs,z,X_,Y_,L_pre,shape):
        scale = z/dirs[:,2]
        pir_u = da.add.outer(ants[:,0],dirs[:,0]*scale).flatten()
        pir_v = da.add.outer(ants[:,1],dirs[:,1]*scale).flatten()
        out = da.sum(da.exp(-(da.subtract.outer(X_,pir_u)**2 + da.subtract.outer(Y_,pir_v)**2)/L_pre**2/2.),axis=1).reshape(shape)
        return out
    planes = []
    i = 0
    while i < len(zvec):
        planes.append(doPlane(ants_uvw,dirs_uvw,zvec[i],X_,Y_,L_pre,X.shape))
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
    neTCI = TriCubic(filename="output/simulated/neModel-0.hdf5").copy()
    antennas,antennaLabels = datapack.get_antennas(antIdx = -1)
    patches, patchNames = datapack.get_directions(dirIdx=-1)
    times,timestamps = datapack.get_times(timeIdx=[0])
    Na = len(antennas)
    Nt = len(times)
    Nd = len(patches)  
    fixtime = times[Nt>>1]
    #print("Fixing frame at {}".format(fixtime.isot))
    phase = datapack.getCenterDirection()
    uvw = UVW(location = datapack.radioArray.getCenter().earth_location,obstime = fixtime,phase = phase)
    ants_uvw = antennas.transform_to(uvw).cartesian.xyz.to(au.km).value.transpose()
    dirs_uvw = patches.transform_to(uvw).cartesian.xyz.value.transpose()
    t1 =clock()
    F0 = precondition(ants_uvw,dirs_uvw,neTCI,L_ne=15.)
    print("Time for preconditioning build {}".format(clock()-t1))
    from PlotTools import animateTCISlices
    F0TCI = TriCubic(neTCI.xvec,neTCI.yvec,neTCI.zvec,F0)
    animateTCISlices(F0TCI,"output/test/infomatrix",numSeconds=10.)

if __name__ == '__main__':
    test_precondition()


# In[ ]:



