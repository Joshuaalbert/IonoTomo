
# coding: utf-8

# In[ ]:

import glob
from RadioArray import RadioArray
import numpy as np
import astropy.units as au
import astropy.time as at
import astropy.coordinates as ac
import dill
dill.settings['recurse'] = True
import pp

import os

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
                    'times':outTimes,'timeStamps':outTimeStamps,
                    'directions':outDirections,'patchNames':outPatchNames,'dtec':outDtec}
    '''
    def __init__(self,dataDict):
        '''get the astropy object defining rays and then also the dtec data'''
        self.addDataDict(**dataDict)
        self.Na = len(self.antennas)
        self.Nt = len(self.times)
        self.Nd = len(self.directions)
        print("Loaded {0} antennas, {1} times, {2} directions".format(self.Na,self.Nt,self.Nd))
    
    def addDataDict(self,**args):
        '''Set up variables here that will hold references throughout'''
        for attr in args.keys():
            try:
                setattr(self,attr,args[attr])
            except:
                print("Failed to set {0} to {1}".format(attr,args[attr]))

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
            outputLabels.append(self.timeStamps[timeIdx[j]])
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
        refAntIdx = None
        i = 0
        while i < self.Na:
            if self.antennaLabels[i] == refAnt:
                refAntIdx = i
                break
            i += 1
            
        assert refAntIdx is not None, "{} is not a valid antenna. Choose from {}".format(refAnt,self.antennaLabels)
        self.dtec -= self.dtec[refAntIdx,:,:]
        
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
        Na = self.Na - antennasFound
        dtec = np.zeros([Na,self.Nt,self.Nd],dtype=np.double)
        skip = 0
        i = 0
        while i < self.Na:
            if self.antennaLabels[i-skip] in antennaLabels:
                skip += 1
            else:
                dtec[i,:,:] = self.dtec[i+skip,:,:]
            i += 1
        self.antennaLabels = self.antennaLabels[mask]
        self.antennas = self.antennas[mask]
        self.dtec = dtec
        self.Na = len(self.antennas)
        
def PrepareData(infoFile,dataFolder,timeStart = 0, timeEnd = -1,arrayFile='arrays/lofar.hba.antenna.cfg',
                load=False,dataFilePrefix='inversion_dTEC',numThreads=6):
    '''Grab real data from soltions products. 
    Stores in a DataPack object.'''
    
    def ppFetchPatch(patchFile,timeStart,timeEnd,radioArray):
        outAntennas, outAntennaLabels, outTimes, outTimeStamps, outDtec_ = ParallelInversionProducts.fetchPatch(patchFile,timeStart,timeEnd,radioArray)
        return outAntennas, outAntennaLabels, outTimes, outTimeStamps, outDtec_
    
    assert os.path.isdir(dataFolder), "{0} is not a directory".format(dataFolder)
    dataFile = "{0}/{1}.dill".format(dataFolder,dataFilePrefix)
    print("Checking for existing preprocessed data.")
    if os.path.isfile(dataFile) and load:
        #try to load
        try:
            f = open(dataFile,'rb')
            dataDict = dill.load(f)
            print("Loaded real data from {0}".format(dataFile))
            f.close()
            generate = False
        except:
            print("Failed to load data")
            generate = True
    else:
        generate = True
    if generate:
        jobs = {}
        print("Creating jobs server")
        job_server = pp.Server(numThreads, ppservers=())
        
        print("Generating data using solutions in {0}".format(dataFolder))
        print("Using radio array file: {}".format(arrayFile))
        #get array stations (they must be in the array file to be considered for data packaging)
        radioArray = RadioArray(arrayFile,frequency=150e6)#set frequency from solutions todo
        #get patch names and directions for dataset
        info = np.load(infoFile)
        #these define the direction order
        patches = info['patches']#names
        radec = info['directions']
        outAntennas = None
        outAntennaLabels = None
        outTimes = None
        outTimeStamps = None
        outDirections = None
        outPatchNames = None
        outDirections_ = []
        outPatchNames_ = []
        outDtec_ = []#will be processed to a NaxNtxNd array
        failed = 0
        numPatches = len(patches)
        print("Loaded {0} patches".format(numPatches))
        patchIdx = 0
        while patchIdx < numPatches:
            patch = patches[patchIdx]
            rd = radec[patchIdx]
            dir = ac.SkyCoord(rd.ra,rd.dec,frame='icrs')
            #find the appropriate file (this will be standardized later)
            files = glob.glob("{0}/*_{1}_*.npz".format(dataFolder,patch))
            if len(files) == 1:
                file = files[0]
            else:
                print('Too many files found. Could not find patch: {0}'.format(patch))
                patchIdx += 1
                continue
            outPatchNames_.append(patch)
            outDirections_.append(dir)
            
            job = job_server.submit(ppFetchPatch,
                       args=(file,timeStart,timeEnd,radioArray),
                       depfuncs=(),
                       modules=('ParallelInversionProducts',))
            jobs[patchIdx] = job
            patchIdx += 1
        #Get job results
        patchIdx = 0
        while patchIdx < numPatches:
            outAntennas, outAntennaLabels, outTimes, outTimeStamps, outDtec_job = jobs[patchIdx]()
            outDtec_ += outDtec_job
            patchIdx += 1
        job_server.print_stats()
        job_server.destroy()
        dirArray = np.zeros([len(outDirections_),2])
        k = 0
        while k < len(outDirections_):
            dirArray[k,0] = outDirections_[k].ra.deg
            dirArray[k,1] = outDirections_[k].dec.deg
            k += 1
        outDirections = ac.SkyCoord(dirArray[:,0]*au.deg,dirArray[:,1]*au.deg,frame='icrs')
        outPatchNames = np.array(outPatchNames_)
        Na = len(outAntennas)
        Nt = len(outTimes)
        Nd = len(outDirections)
        outDtec = np.zeros([Na,Nt,Nd],dtype=np.double)
        count = 0
        k = 0
        while k < Nd:
            j = 0
            while j < Nt:
                i = 0
                while i < Na:
                    outDtec[i,j,k] = outDtec_[count]
                    count += 1
                    i += 1
                j += 1
            k += 1
        dataDict = {'radioArray':radioArray,'antennas':outAntennas,'antennaLabels':outAntennaLabels,
                    'times':outTimes,'timeStamps':outTimeStamps,
                    'directions':outDirections,'patchNames':outPatchNames,'dtec':outDtec}
        f = open(dataFile,'wb')
        dill.dump(dataDict,f)
        f.close()
    return DataPack(dataDict)
    
def plotDataPack(dataPack):
    import pylab as plt
    dtec = dataPack.get_dtec(antIdx = [0],dirIdx=-1,timeIdx=np.arange(6))
    directions, patchNames = dataPack.get_directions(dirIdx=-1)
    antennas, antLabels = dataPack.get_antennas(antIdx=[0])
    times,timestamps = dataPack.get_times(timeIdx=np.arange(6))
    Na = len(antennas)
    Nt = len(times)
    Nd = len(directions)
    #use direction average as phase tracking direction
    from UVWFrame import UVW
    from ENUFrame import ENU
    raMean = np.mean(directions.transform_to('icrs').ra)
    decMean = np.mean(directions.transform_to('icrs').dec)
    phase = ac.SkyCoord(raMean,decMean,frame='icrs')
    loc = dataPack.radioArray.getCenter().earth_location
    f = plt.figure()
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    j = 0
    while j < Nt:
        time = times[j]
        uvw = UVW(location=loc,obstime=time,phase = phase)
        enu = ENU(location=loc,obstime=time)
        dirs_uvw = directions.transform_to(uvw)
        dirs_enu = directions.transform_to(enu)
        factor300 = 300./dirs_uvw.w.value
        sc1 = ax1.scatter(dirs_uvw.u.value*factor300,dirs_uvw.v.value*factor300, c=dtec[0,j,:],
                        vmin=np.min(dtec),vmax=np.max(dtec),s=(100*dtec[0,j,:])**2,alpha=0.2)
        sc2 = ax2.scatter(dirs_enu.east.value*factor300,dirs_enu.north.value*factor300, c=dtec[0,j,:],
                        vmin=np.min(dtec),vmax=np.max(dtec),s=(100*dtec[0,j,:])**2,alpha=0.2)
        j += 1
    plt.colorbar(sc1)
    plt.show()

if __name__ == '__main__':
    dataPack = PrepareData(infoFile='SB120-129/WendysBootes.npz',
                           dataFolder='SB120-129/',
                           timeStart = 0, timeEnd = 10,
                           arrayFile='arrays/lofar.hba.antenna.cfg',load=False,numThreads=4)
    dataPack.flagAntennas(['CS007HBA1','CS007HBA0','CS013HBA0','CS013HBA1'])
    dataPack.setReferenceAntenna('CS501HBA1')
    plotDataPack(dataPack)


# In[ ]:



