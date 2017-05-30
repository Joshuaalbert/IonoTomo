
# coding: utf-8

# In[ ]:

import glob
import numpy as np
from scipy.interpolate import interp2d
import astropy.units as au
import astropy.time as at
import astropy.coordinates as ac
import pp
import h5py
import os
import pylab as plt

from RadioArray import RadioArray
from UVWFrame import UVW
from PointingFrame import Pointing


def getDatumIdx(antIdx,timeIdx,dirIdx,numAnt,numTimes):
    '''standarizes indexing'''
    idx = antIdx + numAnt*(timeIdx + numTimes*dirIdx)
    return idx

def getDatum(datumIdx,numAnt,numTimes):
    antIdx = datumIdx % numAnt
    timeIdx = (datumIdx - antIdx)/numAnt % numTimes
    dirIdx = (datumIdx - antIdx - numAnt*timeIdx)/numAnt/numTimes
    return antIdx,timeIdx,dirIdx

class DataPack(object):
    '''dataDict = {'radioArray':radioArray,'antennas':outAntennas,'antennaLabels':outAntennaLabels,
                    'times':outTimes,'timestamps':outTimestamps,
                    'directions':outDirections,'patchNames':outPatchNames,'dtec':outDtec}
    '''
    def __init__(self,dataDict=None,filename=None):
        '''get the astropy object defining rays and then also the dtec data'''
        if dataDict is not None:
            self.addDataDict(**dataDict)
        else:
            if filename is not None:
                self.load(filename)
                return
        self.refAnt = None
        print("Loaded {0} antennas, {1} times, {2} directions".format(self.Na,self.Nt,self.Nd))
    
    def __repr__(self):
        return "DataPack: numAntennas = {}, numTimes = {}, numDirections = {}\nReference Antenna = {}".format(self.Na,self.Nt,self.Nd,self.refAnt)
    def clone(self):
        dataPack = DataPack({'radioArray':self.radioArray, 'antennas':self.antennas, 'antennaLabels':self.antennaLabels,
                        'times':self.times, 'timestamps':self.timestamps, 'directions':self.directions,
                         'patchNames' : self.patchNames, 'dtec':self.dtec})
        dataPack.setReferenceAntenna(self.refAnt)
        return dataPack
    
    def save(self,filename):
        dt = h5py.special_dtype(vlen=str)
        f = h5py.File(filename,'w')
        antennaLabels = f.create_dataset("datapack/antennas/labels",(self.Na,),dtype=dt)
        f["datapack/antennas"].attrs['frequency'] = self.radioArray.frequency
        antennas = f.create_dataset("datapack/antennas/locs",(self.Na,3),dtype=np.double)
        antennaLabels[...] = self.antennaLabels
        antennas[:,:] = self.antennas.cartesian.xyz.to(au.m).value.transpose()#to Nax3 in m
        patchNames = f.create_dataset("datapack/directions/patchnames",(self.Nd,),dtype=dt)
        ra = f.create_dataset("datapack/directions/ra",(self.Nd,),dtype=np.double)
        dec = f.create_dataset("datapack/directions/dec",(self.Nd,),dtype=np.double)
        patchNames[...] = self.patchNames
        ra[...] = self.directions.ra.deg
        dec[...] = self.directions.dec.deg
        timestamps = f.create_dataset("datapack/times/timestamps",(self.Nt,),dtype=dt)
        gps = f.create_dataset("datapack/times/gps",(self.Nt,),dtype=np.double)
        timestamps[...] = self.timestamps
        gps[...] = self.times.gps
        dtec = f.create_dataset("datapack/dtec",(self.Na,self.Nt,self.Nd),dtype=np.double)
        dtec[:,:,:] = self.dtec
        dtec.attrs['refAnt'] = str(self.refAnt)
        f.close()
        
    def load(self,filename):
        f = h5py.File(filename,'r')
        self.antennaLabels = f["datapack/antennas/labels"][:].astype(str)
        antennas = f["datapack/antennas/locs"][:,:]
        frequency = f["datapack/antennas"].attrs['frequency']
        self.radioArray = RadioArray(antennaPos = antennas,frequency = frequency)
        self.antennas = ac.SkyCoord(antennas[:,0]*au.m,antennas[:,1]*au.m,antennas[:,2]*au.m,frame='itrs')
        self.patchNames = f["datapack/directions/patchnames"][:].astype(str)
        ra = f["datapack/directions/ra"][:]
        dec = f["datapack/directions/dec"][:]
        self.directions = ac.SkyCoord(ra*au.deg,dec*au.deg,frame='icrs')
        self.timestamps = f["datapack/times/timestamps"][:].astype(str)
        self.times = at.Time(self.timestamps,format='isot',scale='tai')
        self.dtec = f["datapack/dtec"][:,:,:]
        self.refAnt = np.array(f["datapack/dtec"].attrs['refAnt']).astype(str).item(0)
        self.Na = len(self.antennas)
        self.Nt = len(self.times)
        self.Nd = len(self.directions)
        self.setReferenceAntenna(self.refAnt)
        f.close()
        
    
    def addDataDict(self,**args):
        '''Set up variables here that will hold references throughout'''
        for attr in args.keys():
            try:
                setattr(self,attr,args[attr])
            except:
                print("Failed to set {0} to {1}".format(attr,args[attr]))
        self.Na = len(self.antennas)
        self.Nt = len(self.times)
        self.Nd = len(self.directions)
                
    def set_dtec(self,dtec,antIdx=[],timeIdx=[], dirIdx=[],refAnt=None):
        '''Set the specified dtec solutions corresponding to the requested indices.
        value of -1 means all.'''
        if antIdx is -1:
            antIdx = np.arange(self.Na)
        if timeIdx is -1:
            timeIdx = np.arange(self.Nt)
        if dirIdx is -1:
            dirIdx = np.arange(self.Nd)
        antIdx = np.sort(antIdx)
        timeIdx = np.sort(timeIdx)
        dirIdx = np.sort(dirIdx)
        Na = len(antIdx)
        Nt = len(timeIdx)
        Nd = len(dirIdx)
        i = 0
        while i < Na:
            j = 0
            while j < Nt:
                k = 0
                while k < Nd:
                    self.dtec[antIdx[i],timeIdx[j],dirIdx[k]] = dtec[i,j,k]
                    k += 1
                j += 1
            i += 1
        if refAnt is not None:
            self.setReferenceAntenna(refAnt)
        else:
            if self.refAnt is not None:
                self.setReferenceAntenna(self.refAnt)
                

    def get_dtec(self,antIdx=[],timeIdx=[], dirIdx=[]):
        '''Retrieve the specified dtec solutions corresponding to the requested indices.
        value of -1 means all.'''
        if antIdx is -1:
            antIdx = np.arange(self.Na)
        if timeIdx is -1:
            timeIdx = np.arange(self.Nt)
        if dirIdx is -1:
            dirIdx = np.arange(self.Nd)
        antIdx = np.sort(antIdx)
        timeIdx = np.sort(timeIdx)
        dirIdx = np.sort(dirIdx)
        Na = len(antIdx)
        Nt = len(timeIdx)
        Nd = len(dirIdx)
        output = np.zeros([Na,Nt,Nd],dtype=np.double)
        i = 0
        while i < Na:
            j = 0
            while j < Nt:
                k = 0
                while k < Nd:
                    output[i,j,k] = self.dtec[antIdx[i],timeIdx[j],dirIdx[k]]
                    k += 1
                j += 1
            i += 1
        return output
    
    def get_antennas(self,antIdx=[]):
        '''Get the list of antenna locations in itrs'''
        if antIdx is -1:
            antIdx = np.arange(self.Na)
        antIdx = np.sort(antIdx)
        output = self.antennas[antIdx]
        Na = len(antIdx)
        outputLabels = []
        i = 0
        while i < Na:
            outputLabels.append(self.antennaLabels[antIdx[i]])
            i += 1
        return output, outputLabels
    
    def get_times(self,timeIdx=[]):
        '''Get the gps times'''
        if timeIdx is -1:
            timeIdx = np.arange(self.Nt)
        timeIdx = np.sort(timeIdx)
        output = self.times[timeIdx]
        Nt = len(timeIdx)
        outputLabels = []
        j = 0
        while j < Nt:
            outputLabels.append(self.timestamps[timeIdx[j]])
            j += 1
        return output, outputLabels
    
    def get_directions(self, dirIdx=[]):
        '''Get the array of directions in itrs'''
        if dirIdx is -1:
            dirIdx = np.arange(self.Nd)
        dirIdx = np.sort(dirIdx)
        output = self.directions[dirIdx]
        Nd = len(dirIdx)
        outputLabels = []
        k = 0
        while k < Nd:
            outputLabels.append(self.patchNames[dirIdx[k]])
            k += 1
        return output, outputLabels
    
    def setReferenceAntenna(self,refAnt):
        if refAnt is None:
            return
        refAntIdx = None
        i = 0
        while i < self.Na:
            if self.antennaLabels[i] == refAnt:
                refAntIdx = i
                break
            i += 1          
        assert refAntIdx is not None, "{} is not a valid antenna. Choose from {}".format(refAnt,self.antennaLabels)
        print("Setting refAnt: {}".format(refAnt))
        self.refAnt = refAnt
        self.dtec = self.dtec - self.dtec[refAntIdx,:,:]
        
    def getCenterDirection(self):
        raMean = np.mean(self.directions.transform_to('icrs').ra)
        decMean = np.mean(self.directions.transform_to('icrs').dec)
        phase = ac.SkyCoord(raMean,decMean,frame='icrs')
        return phase

    def findFlaggedAntennas(self):
        '''Determine which antennas are flagged'''
        assert self.refAnt is not None, "Set a refAnt before finding flagged antennas"
        mask = np.sum(np.sum(self.dtec,axis=2),axis=1) == 0
        i = 0
        while i < self.Na:
            if self.antennaLabels[i] == self.refAnt:
                refAntIdx = i
                break
            i += 1   
        mask[refAntIdx] = False
        return list(self.antennaLabels[mask])
        
    def flagAntennas(self,antennaLabels):
        '''remove data corresponding to the given antenna names if it exists'''
        assert type(antennaLabels) == type([]), "{} is not a list of station names. Choose from {}".format(antennaLabels,self.antennaLabels)
        mask = np.ones(len(self.antennaLabels), dtype=bool)
        antennasFound = 0
        i = 0
        while i < self.Na:
            if self.antennaLabels[i] in antennaLabels:
                antennasFound += 1
                mask[i] = False
            i += 1
        #some flags may have not existed in data
        self.antennaLabels = self.antennaLabels[mask]
        self.antennas = self.antennas[mask]
        self.dtec = self.dtec[mask,:,:]
        self.Na = len(self.antennas)
        
    def flagPatches(self,patchNames):
        '''remove data corresponding to the given antenna names if it exists'''
        assert type(patchNames) == type([]), "{} is not a list of patch names. Choose from {}".format(antennaLabels,self.antennaLabels)
        mask = np.ones(len(self.patchNames), dtype=bool)
        patchesFound = 0
        i = 0
        while i < self.Nd:
            if self.patchNames[i] in patchNames:
                patchesFound += 1
                mask[i] = False
            i += 1
        #some flags may have not existed in data
        self.patchNames = self.patchNames[mask]
        self.directions = self.directions[mask]
        self.dtec = self.dtec[:,:,mask]
        self.Nd = len(self.directions)
        
def transferPatchData(infoFile, dataFolder, hdf5Out):
    '''transfer old numpy format to hdf5. Only run with python 2.7'''
    
    assert os.path.isdir(dataFolder), "{0} is not a directory".format(dataFolder)
    dt = h5py.special_dtype(vlen=str)
    f = h5py.File(hdf5Out,"w")
    
    info = np.load(infoFile)
    #these define the direction order
    patches = info['patches']#names
    radec = info['directions']#astrpy.icrs
    Nd = len(patches)
    print("Loading {} patches".format(Nd))
    namesds = f.create_dataset("dtecObservations/patchNames",(Nd,),dtype=dt)
    #rads = f.create_dataset("dtecObservations/patches/ra",(Nd,),dtype=np.double)
    #dec = f.create_dataset("dtecObservations/patches/dec",(Nd,),dtype=np.double)
    dset = f['dtecObservations']
    dset.attrs['frequency'] = 150e6
    namesds[...] = patches
    #rads[...] = radec.ra.deg
    #decds[...] = radec.dec.deg
    
    patchIdx = 0
    while patchIdx < Nd:
        patch = patches[patchIdx]
        #find the appropriate file (this will be standardized later)
        files = glob.glob("{0}/*_{1}_*.npz".format(dataFolder,patch))
        if len(files) == 1:
            patchFile = files[0]
        else:
            print('Too many files found. Could not find patch: {0}'.format(patch))
            patchIdx += 1
            continue
        try:
            d = np.load(patchFile)
            print("Loading data file: {0}".format(patchFile))
        except:
            print("Failed loading data file: {0}".format(patchFile))
            return  
        if "dtecObservations/antennaLabels" not in f:
            antennaLabels = d['antennas']#labels
            Na = len(antennaLabels)
            antennaLabelsds = f.create_dataset("dtecObservations/antennaLabels",(Na,),dtype=dt)
            antennaLabelsds[...] = antennaLabels
        if "dtecObservations/timestamps" not in f:
            times = d['times']#gps tai
            timestamps = at.Time(times,format='gps',scale='tai').isot
            Nt = len(times)
            print(len(timestamps[0]))
            timeds = f.create_dataset("dtecObservations/timestamps",(Nt,),dtype=dt)
            timeds[...] = timestamps
        patchds = f.create_dataset("dtecObservations/patches/{}".format(patch),(Nt,Na),dtype=np.double)
        patchds[...] = d['data']
        patchds.attrs['ra'] = radec[patchIdx].ra.deg
        patchds.attrs['dec'] = radec[patchIdx].dec.deg
        patchIdx += 1
    f.close()
    
        
def prepareDataPack(hdf5Datafile,timeStart=0,timeEnd=-1,arrayFile='arrays/lofar.hba.antenna.cfg'):
    '''Grab real data from soltions products. 
    Stores in a DataPack object.'''
    
    f = h5py.File(hdf5Datafile,'r')
    dset = f['dtecObservations']
    frequency = dset.attrs['frequency']
    print("Using radio array file: {}".format(arrayFile))
    #get array stations (they must be in the array file to be considered for data packaging)
    radioArray = RadioArray(arrayFile,frequency=frequency)#set frequency from solutions todo
    print("Created {}".format(radioArray))
    patchNames = f["dtecObservations/patchNames"][:].astype(str)
    Nd = len(patchNames)
    ra = np.zeros(Nd,dtype= np.double)
    dec = np.zeros(Nd,dtype=np.double)
    antennaLabels = f["dtecObservations/antennaLabels"][:].astype(str)
    Na = len(antennaLabels)
    antennas = np.zeros([3,Na],dtype=np.double)
    antIdx = 0#index in solution table
    while antIdx < Na:
        ant = antennaLabels[antIdx]
        labelIdx = radioArray.getAntennaIdx(ant)  
        if labelIdx is None:
            print("failed to find {} in {}".format(ant,radioArray.labels))
            return
        #ITRS WGS84
        stationLoc = radioArray.locs[labelIdx]
        antennas[:,antIdx] = stationLoc.cartesian.xyz.to(au.km).value.flatten()
        antIdx += 1
    antennas = ac.SkyCoord(antennas[0,:]*au.km,antennas[1,:]*au.km,
                          antennas[2,:]*au.km,frame='itrs')
    timestamps = f["dtecObservations/timestamps"][:].astype(str)
    times = at.Time(timestamps,format="isot",scale='tai')
    Nt = len(timestamps)
    dtec = np.zeros([Na,Nt,Nd],dtype=np.double)
    patchIdx = 0
    while patchIdx < Nd:
        patchName = patchNames[patchIdx]
        patchds = f["dtecObservations/patches/{}".format(patchName)]
        ra[patchIdx] = patchds.attrs['ra']
        dec[patchIdx] = patchds.attrs['dec']
        dtec[:,:,patchIdx] = patchds[:,:].transpose()#from NtxNa to NaxNt
        patchIdx += 1
    f.close()
    directions = ac.SkyCoord(ra*au.deg,dec*au.deg,frame='icrs')
    dataDict = {'radioArray':radioArray,'antennas':antennas,'antennaLabels':antennaLabels,
                'times':times,'timestamps':timestamps,
                'directions':directions,'patchNames':patchNames,'dtec':dtec}
    return DataPack(dataDict)

def interpNearest(x,y,z,x_,y_):
    dx = np.subtract.outer(x_,x)
    dy = np.subtract.outer(y_,y)
    r = dx**2
    dy *= dy
    r += dy
    np.sqrt(r,out=r)
    arg = np.argmin(r,axis=1)
    z_ = z[arg]
    return z_

def plotDataPack(datapack,antIdx=-1,timeIdx=[0], dirIdx=-1,figname=None,vmin=None,vmax=None):
    assert datapack.refAnt is not None, "set DataPack refAnt first"
    directions, patchNames = datapack.get_directions(dirIdx=dirIdx)
    antennas, antLabels = datapack.get_antennas(antIdx=antIdx)
    times,timestamps = datapack.get_times(timeIdx=timeIdx)
    dtec = np.stack([np.mean(datapack.get_dtec(antIdx = antIdx,dirIdx=dirIdx,timeIdx=timeIdx),axis=1)],axis=1)
    Na = len(antennas)
    Nt = len(times)
    Nd = len(directions)
    refAntIdx = None
    for i in range(Na):
        if antLabels[i] == datapack.refAnt:
            refAntIdx = i
    fixtime = times[Nt>>1]
    phase = datapack.getCenterDirection()
    arrayCenter = datapack.radioArray.getCenter()
    uvw = UVW(location = arrayCenter.earth_location,obstime = fixtime,phase = phase)
    ants_uvw = antennas.transform_to(uvw)
    #make plots, M by 4
    M = (Na>>2) + 1 + 1
    fig = plt.figure(figsize=(11.,11./4.*M))
    #use direction average as phase tracking direction
    if vmax is None:  
        vmax = np.percentile(dtec.flatten(),95)
        vmax=np.max(dtec)
    if vmin is None:
        vmin = np.percentile(dtec.flatten(),5)
        vmin=np.min(dtec)
    
    i = 0 
    while i < Na:
        ax = fig.add_subplot(M,4,i+1)

        dx = np.sqrt((ants_uvw.u[i] - ants_uvw.u[refAntIdx])**2 + (ants_uvw.v[i] - ants_uvw.v[refAntIdx])**2).to(au.km).value
        ax.annotate(s="{} : {:.2g} km".format(antLabels[i],dx),xy=(.2,.8),xycoords='axes fraction')
        #ax.set_title("Ref. Proj. Dist.: {:.2g} km".format(dx))
        ax.set_xlabel("U km")
        ax.set_ylabel("V km")
        N = 25
        uvw = UVW(location=arrayCenter.earth_location,obstime=fixtime,phase=phase)
        dirs_uvw = directions.transform_to(uvw)
        factor300 = 300./dirs_uvw.w.value
            
        U,V = np.meshgrid(np.linspace(np.min(dirs_uvw.u.value*factor300),np.max(dirs_uvw.u.value*factor300),N),
                          np.linspace(np.min(dirs_uvw.v.value*factor300),np.max(dirs_uvw.v.value*factor300),N))
        D = interpNearest(dirs_uvw.u.value*factor300,dirs_uvw.v.value*factor300,dtec[i,0,:],U.flatten(),V.flatten()).reshape(U.shape)
        im = ax.imshow(D,origin='lower',extent=(np.min(U),np.max(U),np.min(V),np.max(V)),aspect='auto',
                      vmin = vmin, vmax= vmax,cmap=plt.cm.coolwarm,alpha=1.)
        sc1 = ax.scatter(dirs_uvw.u.value*factor300,dirs_uvw.v.value*factor300, c='black',
                        marker='+')
        i += 1
    ax = fig.add_subplot(M,4,Na+1)
    plt.colorbar(im,cax=ax,orientation='vertical')
    if figname is not None:
        plt.savefig("{}.png".format(figname),format='png')
    else:
        plt.show()
    
def test_plotDataPack():
    datapack = DataPack(filename="output/test/simulate/datapackSim_turbulent.hdf5")
    try:
        os.makedirs('output/test/plotDataPack')
    except:
        pass
    plotDataPack(datapack,antIdx=-1,timeIdx=[0,1,2,3], dirIdx=-1,figname='output/test/plotDataPack/fig')

def test_prepareDataPack():
    dataPack = prepareDataPack('SB120-129/dtecData.hdf5',timeStart=0,timeEnd=-1,
                           arrayFile='arrays/lofar.hba.antenna.cfg')
    dataPack.flagAntennas(['CS007HBA1','CS007HBA0','CS013HBA0','CS013HBA1'])
    dataPack.setReferenceAntenna(dataPack.antennaLabels[0])
    #'CS501HBA1'
    dataPack.save("output/test/datapackObs.hdf5")

if __name__ == '__main__':
    #transferPatchData(infoFile='SB120-129/WendysBootes.npz', 
    #                  dataFolder='SB120-129/', 
    #                  hdf5Out='SB120-129/dtecData.hdf5')
    test_plotDataPack()

