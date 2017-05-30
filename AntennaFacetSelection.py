
# coding: utf-8

# In[ ]:

'''Choose optimal facet and antenna layout from a datapack object'''
from RealData import DataPack
import numpy as np
import astropy.units as au
from UVWFrame import UVW

def selectFacets(N,datapack,dirIdx=-1,timeIdx=[0]):
    '''Will select N uniform assembly of antennas and return a datapack
    with the rest flagged'''
    assert N <= datapack.Nd, "Requested number of directions {} to large {}".format(N,datapack.Na)
    patches, patchNames = datapack.get_directions(dirIdx=dirIdx)
    times,timestamps = datapack.get_times(timeIdx=timeIdx)
    Nd = len(patches)
    Nt = len(times)
    fixtime = times[Nt>>1]
    #print("Fixing frame at {}".format(fixtime.isot))
    phase = datapack.getCenterDirection()
    uvw = UVW(location = datapack.radioArray.getCenter().earth_location,obstime = fixtime,phase = phase)
    #
    dirs = patches.transform_to(uvw).cartesian.xyz.value.transpose()
    center = phase.transform_to(uvw).cartesian.xyz.value.transpose()
    mag = np.arccos(np.dot(dirs, center))
    #mag and get first
    argsort = np.argsort(mag)
    outdirs = [dirs[argsort[0],:]]
    outidx = [argsort[0]]
    i = 1
    while i < N:
        center = np.mean(outdirs,axis=0)
        mag = np.arccos(np.dot(dirs,center))
        argsort = np.argsort(mag)
        j = len(argsort) - 1
        while j >= 0:
            if argsort[j] not in outidx:
                outidx.append(argsort[j])
                outdirs.append(dirs[argsort[j],:])
                break
            j -= 1
        i += 1
    # flag all others
    flag = []
    i = 0
    while i < len(patchNames):
        if i not in outidx:
            flag.append(patchNames[i])
        i += 1
    outDatapack = datapack.clone()
    outDatapack.flagPatches(flag)
    print("flagged {}".format(flag))
    return outDatapack

def selectAntennas(N,datapack,antIdx=-1,timeIdx=[0]):
    '''Will select N uniform assembly of antennas and return a datapack
    with the rest flagged'''
    assert N <= datapack.Na, "Requested number of antennas {} to large {}".format(N,datapack.Na)
    antennas,antennaLabels = datapack.get_antennas(antIdx = antIdx)
    times,timestamps = datapack.get_times(timeIdx=timeIdx)
    Na = len(antennas)
    Nt = len(times)
    fixtime = times[Nt>>1]
    #print("Fixing frame at {}".format(fixtime.isot))
    phase = datapack.getCenterDirection()
    uvw = UVW(location = datapack.radioArray.getCenter().earth_location,obstime = fixtime,phase = phase)
    #
    center = datapack.radioArray.getCenter().transform_to(uvw).cartesian.xyz.to(au.km).value.transpose()
    ants = antennas.transform_to(uvw).cartesian.xyz.to(au.km).value.transpose()
    dist = ants - center
    #mag and get first
    mag = np.linalg.norm(dist,axis=1)
    argsort = np.argsort(mag)
    outants = [ants[argsort[0],:]]
    outidx = [argsort[0]]
    i = 1
    while i < N:
        center = np.mean(outants,axis=0)
        dist = np.subtract(ants,center)
        mag = np.linalg.norm(dist,axis=1)
        argsort = np.argsort(mag)
        j = len(argsort) - 1
        while j >= 0:
            if argsort[j] not in outidx:
                outidx.append(argsort[j])
                outants.append(ants[argsort[j],:])
                break
            j -= 1
        i += 1
    # flag all others
    flag = []
    i = 0
    while i < len(antennaLabels):
        if i not in outidx:
            flag.append(antennaLabels[i])
        i += 1
    outDatapack = datapack.clone()
    outDatapack.flagAntennas(flag)
    print("flagged {}".format(flag))
    outDatapack.setReferenceAntenna(antennaLabels[outidx[0]])
    return outDatapack

def selectAntennaFacets(N,datapack,antIdx=-1,dirIdx=-1,timeIdx=[0]):
    datapack = selectAntennas(N,datapack,antIdx=-1,timeIdx=timeIdx)
    datapack = selectFacets(N,datapack,dirIdx=-1,timeIdx=timeIdx)
    return datapack

def test_selectAntennas():
    datapack = DataPack(filename="datapackObs.hdf5").clone()
    datapackSel = selectAntennas(10,datapack,-1, timeIdx = [0])
    antennas,antennaLabels = datapackSel.get_antennas(antIdx = -1)
    times,timestamps = datapackSel.get_times(timeIdx=[0])
    Na = len(antennas)
    Nt = len(times)
    fixtime = times[Nt>>1]
    #print("Fixing frame at {}".format(fixtime.isot))
    phase = datapackSel.getCenterDirection()
    uvw = UVW(location = datapackSel.radioArray.getCenter().earth_location,obstime = fixtime,phase = phase)
    #
    
    ants = antennas.transform_to(uvw).cartesian.xyz.to(au.km).value.transpose()
    center = np.mean(ants,axis=0)
    dist = ants-center
    mag = np.linalg.norm(dist,axis=1)
    import pylab as plt
    plt.hist(mag,bins=5)
    plt.show()
    print(mag)
    plt.scatter(ants[:,0],ants[:,1])
    plt.show()
    
def test_selectFacets():
    datapack = DataPack(filename="datapackObs.hdf5").clone()
    datapackSel = selectFacets(8,datapack,-1, timeIdx = [0])
    patches, patchNames = datapackSel.get_directions(dirIdx = -1)
    times,timestamps = datapackSel.get_times(timeIdx=[0])
    Nd = len(patches)
    Nt = len(times)
    fixtime = times[Nt>>1]
    #print("Fixing frame at {}".format(fixtime.isot))
    phase = datapackSel.getCenterDirection()
    uvw = UVW(location = datapackSel.radioArray.getCenter().earth_location,obstime = fixtime,phase = phase)
    #
    
    dirs = patches.transform_to(uvw).cartesian.xyz.value.transpose()
    center = np.mean(dirs,axis=0)
    mag = np.arccos(np.dot(dirs,center))
    import pylab as plt
    plt.hist(mag,bins=5)
    plt.show()
    print(mag)
    plt.scatter(dirs[:,0],dirs[:,1])
    plt.show()
    
    
if __name__ == '__main__':
    test_selectAntennas()
    test_selectFacets()

