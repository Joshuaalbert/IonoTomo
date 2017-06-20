
# coding: utf-8

# In[1]:

import astropy.units as au
import astropy.time as at
import astropy.coordinates as ac
import numpy as np
from scipy.special import gamma
from UVWFrame import UVW
from IRI import aPrioriModel
from TricubicInterpolation import TriCubic
from Covariance import CovarianceClass

def determineInversionDomain(spacing,antennas, directions, pointing, zmax, padding = 20):
    '''Determine the domain of the inversion'''
    ants = antennas.transform_to(pointing).cartesian.xyz.to(au.km).value.transpose()
    dirs = directions.transform_to(pointing).cartesian.xyz.value.transpose()
    #old
    uend = np.add.outer(ants[:,0],dirs[:,0]*zmax/dirs[:,2])
    vend = np.add.outer(ants[:,1],dirs[:,1]*zmax/dirs[:,2])
    wend = np.add.outer(ants[:,2],dirs[:,2]*zmax/dirs[:,2])
    
    
    umin = min(np.min(ants[:,0]),np.min(uend.flatten()))-spacing*padding
    umax = max(np.max(ants[:,0]),np.max(uend.flatten()))+spacing*padding
    vmin = min(np.min(ants[:,1]),np.min(vend.flatten()))-spacing*padding
    vmax = max(np.max(ants[:,1]),np.max(vend.flatten()))+spacing*padding
    wmin = min(np.min(ants[:,2]),np.min(wend.flatten()))-spacing*padding
    wmax = max(np.max(ants[:,2]),np.max(wend.flatten()))+spacing*padding
    Nu = np.ceil((umax-umin)/spacing)
    Nv = np.ceil((vmax-vmin)/spacing)
    Nw = np.ceil((wmax-wmin)/spacing)
    uvec = np.linspace(umin,umax,int(Nu))
    vvec = np.linspace(vmin,vmax,int(Nv))
    wvec = np.linspace(wmin,wmax,int(Nw))
    print("Found domain u in {}..{}, v in {}..{}, w in {}..{}".format(umin,umax,vmin,vmax,wmin,wmax))
    return uvec,vvec,wvec

def turbulentPerturbation(TCI,sigma = 3.,corr = 20., nu = 5./2.):    
    covC = CovarianceClass(TCI,sigma,corr,nu)
    B = covC.realization()
    return B
    

def createInitialModel(datapack,antIdx = -1, timeIdx = -1, dirIdx = -1, zmax = 1000.,spacing=5.,padding=20):
    antennas,antennaLabels = datapack.get_antennas(antIdx = antIdx)
    patches, patchNames = datapack.get_directions(dirIdx=dirIdx)
    times,timestamps = datapack.get_times(timeIdx=timeIdx)
    Na = len(antennas)
    Nt = len(times)
    Nd = len(patches)  
    #Setting up ionosphere to use
    print("Using radio array {}".format(datapack.radioArray))
    phase = datapack.getCenterDirection()
    print("Using phase center {} {}".format(phase.ra,phase.dec))
    fixtime = times[Nt>>1]
    print("Fixing frame at {}".format(fixtime.isot))
    uvw = UVW(location = datapack.radioArray.getCenter().earth_location,obstime = fixtime,phase = phase)
    print("Elevation is {}".format(uvw.elevation))
    zenith = datapack.radioArray.getSunZenithAngle(fixtime)
    print("Sun at zenith angle {}".format(zenith))
    print("Creating ionosphere model...")
    xvec,yvec,zvec = determineInversionDomain(spacing,antennas, patches,uvw, zmax, padding = padding)
    X,Y,Z = np.meshgrid(xvec,yvec,zvec,indexing='ij')
    print("Nx={} Ny={} Nz={} number of cells: {}".format(len(xvec),len(yvec),len(zvec),np.size(X)))
    coords = ac.SkyCoord(X.flatten()*au.km,Y.flatten()*au.km,Z.flatten()*au.km,frame=uvw).transform_to('itrs').earth_location.to_geodetic('WGS84')
    heights = coords[2].to(au.km).value#height in geodetic
    neModel = aPrioriModel(heights,zenith).reshape(X.shape)
    neModel[neModel<4e7] = 4e7
    return TriCubic(xvec,yvec,zvec,neModel)

def createTurbulentlModel(datapack,antIdx = -1, timeIdx = -1, dirIdx = -1, zmax = 1000., spacing=5.):
    neTCI = createInitialModel(datapack,antIdx = antIdx, timeIdx = timeIdx, dirIdx = dirIdx, zmax = zmax, spacing= spacing)
    dM = turbulentPerturbation(neTCI,sigma=np.log(5.),corr=25.,nu=7./2.)
    pertTCI = TriCubic(neTCI.xvec,neTCI.yvec,neTCI.zvec,neTCI.getShapedArray()*np.exp(dM))
    return pertTCI
    
def test_createInitialModel():
    from RealData import DataPack
    datapack = DataPack(filename="output/test/datapackObs.hdf5")
    neTCI = createInitialModel(datapack,antIdx = -1, timeIdx = -1, dirIdx = -1, zmax = 1000.)
    neTCI.save("output/test/neModel.hdf5")
    
def test_createTurbulentModel():
    from RealData import DataPack
    from PlotTools import animateTCISlices
    import os
    datapack = DataPack(filename="output/test/datapackObs.hdf5")
    for i in range(1):
        neTCI = createTurbulentlModel(datapack,antIdx = -1, timeIdx = [0], dirIdx = -1, zmax = 1000.)
        try:
            os.makedirs("output/test/InitialModel/turbulent-{}/fig".format(i))
        except:
            pass
        #neTCI.save("output/test/InitialModel/turbulent-{}/neModelTurbulent.hdf5".format(i))
        #animateTCISlices(neTCI,"output/test/InitialModel/turbulent-{}/fig".format(i))
    
if __name__ == '__main__':
    #test_createInitialModel()
    test_createTurbulentModel()


# In[ ]:



