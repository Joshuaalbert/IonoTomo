
# coding: utf-8

# In[11]:

import glob
from RadioArray import RadioArray
from ENUFrame import ENU
import numpy as np
import astropy.units as au
import astropy.time as at
import astropy.coordinates as ac
from Geometry import *
import dill
dill.settings['recurse'] = True

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
    def __init__(self,dataDict):
        '''get the astropy object defining rays and then also the dtec data'''
        self.antennas = dataDict['antennas']#a dictionary {antIdx:label,...}
        self.numAnt = len(self.antennas)
        self.times = dataDict['times']#a dictionary {timeIdx:time,...}
        self.numTimes = len(self.times)
        self.directions = dataDict['directions']#a dictionary {dirIdx:icrs}
        self.numDir = len(self.directions)
        self.dtec = dataDict['dtec'] #a dictionary {datumIdx: data}
        self.radioArray = dataDict['radioArray'] #radioArray containing antenna information

    def get_dtec_array(self,antIdx=[],timeIdx=[], dirIdx=[]):
        '''Retrieve the specified dtec solutions'''
        if antIdx is -1:
            antIdx = self.antennas.keys()
        if timeIdx is -1:
            timeIdx = self.times.keys()
        if dirIdx is -1:
            dirIdx = self.directions.keys()
        #make output
        output = np.zeros([len(antIdx),len(timeIdx),len(dirIdx)],dtype=np.double)
        #grab data
        i = 0
        while i < len(antIdx):
            j = 0
            while j < len(timeIdx):
                k = 0
                while k < len(dirIdx):
                    datumIdx = getDatumIdx(antIdx[i],timeIdx[j],dirIdx[k],self.numAnt,self.numTimes)
                    if datumIdx in self.dtec.keys():
                        output[i,j,k] = self.dtec[datumIdx]
                    else:
                        print('{0} not a valid datumidx'.format(datumIdx))
                        output[i,j,k] = np.nan
                    k += 1
                j += 1
            i += 1
        return output
    
    def get_antennas_array(self,antIdx=[]):
        '''Get the array of antenna locations'''
        if antIdx is -1:
            antIdx = self.antennas.keys()
        output = np.zeros([len(antIdx),3],dtype=np.double)
        locs = self.radioArray.locs.cartesian.xyz.to(au.km).value.transpose()
        i = 0
        while i < len(antIdx):
            idx = self.radioArray.getAntennaIdx(self.antennas[antIdx[i]])
            loc = locs[idx,:]
            i += 1
        return output
    
    def get_times_array(self,timeIdx=[]):
        '''Get the gps times'''
        if timeIdx is -1:
            timeIdx = self.times.keys()
        output = np.zeros(len(timeIdx),dtype=np.double)
        i = 0
        while i < len(timeIdx):
            if timeIdx[i] in self.times.keys():
                output[i,:] = self.times[timeIdx[i]].gps
            else:
                output[i,:] = np.nan
            i += 1
        return output
    
    def get_rays(self,antIdx=[],timeIdx=[], dirIdx=[]):
        '''Retrieve the specified rays, R^ijk'''
        if antIdx is -1:
            antIdx = self.antennas.keys()
        if timeIdx is -1:
            timeIdx = self.times.keys()
        if dirIdx is -1:
            dirIdx = self.directions.keys()
        rays = {}
        j = 0
        while j < len(timeIdx):
            enu = ENU(location=self.radioArray.getCenter().earth_location,obstime=self.times[timeIdx[j]])
            i = 0
            while i < len(antIdx):
                idx = self.radioArray.getAntennaIdx(self.antennas[antIdx[i]])
                origin = self.radioArray.locs[idx].transform_to(enu).cartesian.xyz.to(au.km).value
                k = 0
                while k < len(dirIdx):
                    dir = self.directions[dirIdx[k]].transform_to(enu).cartesian.xyz.value
                    datumIdx = getDatumIdx(antIdx[i],timeIdx[j],dirIdx[k],self.numAnt,self.numTimes)
                    if datumIdx in self.dtec.keys():
                        output[i,j,k] = self.dtec[datumIdx]
                    else:
                        print('{0} not a valid datumidx'.format(datumIdx))
                        output[i,j,k] = np.nan
                    k += 1
                i += 1
            j += 1
        return output
    
    def get_directions_array(self, timeIdx=[], dirIdx=[]):
        '''Get the array of directions in itrs'''
        if timeIdx is -1:
            timeIdx = self.times.keys()
        if dirIdx is -1:
            dirIdx = self.directions.keys()
        #output = np.zeros([len(dirIdx),3],dtype=np.double)
        output = np.zeros([len(timeIdx)*len(dirIdx),3],dtype=np.double)
        #grab data
        
        j = 0
        while j < len(timeIdx):
            enu = ENU(location = radioArray.getCenter().location,obstime=self.times[timeIdx[j]])
            k = 0
            while k < len(dirIdx):
                output[j,k,:] = self.directions[dirIdx[k]].transform_to(enu).cartesian.xyz.to(au.km).value
                k += 1
            j += 1
        return output
    
def PrepareData(infoFile,dataFolder,timeStart = 0, timeEnd = -1,arrayFile='arrays/lofar.hba.antenna.cfg',
                load=False,dataFilePrefix='inversion_dTEC'):
    '''Grab real data from soltions products. 
    Stores in a DataPack object.'''
    
    assert os.path.isdir(dataFolder), "{0} is not a directory".format(dataFolder)
    dataFile = "{0}/{1}.dill".format(dataFolder,dataFilePrefix)
    print("Checking for existing preprocessed data.")
    if os.path.isfile(dataFile):
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
        print("Generating data using solutions in {0}".format(dataFolder))
        print("Using radio array file: {}".format(arrayFile))
        radioArray = RadioArray(arrayFile,frequency=150e6)#set frequency from solutions too
        #get patch names and directions for dataset
        info = np.load(infoFile)
        #these define the direction order
        patches = info['patches']
        numPatches = len(patches)
        radec = info['directions']
        print("Loaded {0} directions".format(numPatches))
        
        #get array stations (they must be in the array file to be considered for data packaging)
        radioArray = RadioArray(arrayFile)
        stationLabels = radioArray.labels
        stationLocs = radioArray.locs.cartesian.xyz.transpose()
        numStations = radioArray.Nantennas
        print("Number of stations in array: {0}".format(numStations))

        #each time gives a different direction for each patch
        #numDirs = numTimes * numPatches #maybe a file doesn't load
        
        outAntennas = {}
        outTimes = {}
        outDirections = {}
        outDtec = {}
        
        patchIdx = 0
        failed = 0
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
            print("Loading data file: {0}".format(file))
            try:
                d = np.load(file)
            except:
                print("Failed loading data file: {0}".format(file))
                failed += 1
                patchIdx += 1
                continue
            #internal data of each patch file (directions set by infoFile)
            antennas = d['antennas']
            times = d['times'][timeStart:timeEnd]#gps tai
            numTimes = len(times)
            tecData = d['data'][timeStart:timeEnd,:]#times x antennas
            timeIdx = 0
            while timeIdx < numTimes:

                time = at.Time(times[timeIdx],format='gps',scale='tai')
                print("Processing time: {0}".format(time.isot))
                
                # get direction of patch at time wrt fixed frame
                
                antIdx = 0#index in solution table
                numAnt = len(antennas)
                while antIdx < len(antennas):
                    ant = antennas[antIdx]
                    labelIdx = radioArray.getAntennaIdx(ant)
                    if labelIdx is None:
                        print("failed to find {}".format(ant))
                    datumIdx = getDatumIdx(antIdx,timeIdx,patchIdx,numAnt,numTimes)
                    
                    #ITRS WGS84
                    stationLoc = ac.SkyCoord(*stationLocs[labelIdx,:],obstime=time,frame='itrs')
                    outAntennas[antIdx] = ant
                    outTimes[timeIdx] = time
                    outDirections[patchIdx] = dir
                    outDtec[datumIdx] = tecData[timeIdx,antIdx]
                    
                    antIdx += 1
                timeIdx += 1
            patchIdx += 1
        dataDict = {'radioArray':radioArray,'antennas':outAntennas,'times':outTimes,'directions':outDirections,'dtec':outDtec}
        f = open(dataFile,'wb')
        dill.dump(dataDict,f)
        f.close()
    return DataPack(dataDict)
    
def plotDataPack(dataPack):
    import pylab as plt
    directions = dataPack.get_directions_array(timeIdx=-1, dirIdx=-1)
    #Nt = directions.shape[0]
    #Nd = directions.shape[1]
    dtec = dataPack.dtec
    antennas = dataPack.get_antennas_array(antIdx=-1)
    Nant = antannas.shape[0]
    plotperaxis = int(np.ceil(np.sqrt(Nant)))
    dtec -= np.min(dtec)
    dtec /= np.max(dtec)
    f = plt.figure()
    print(ax)
    i = 0
    while i < Nant:
        ax = plt.subplot(plotperaxis,plotperaxis,i+1)
        for datumIdx in dtec.keys():
            antIdx,dirIdx,timeIdx = getDatum(datumIdx,numDirections,numTimes)
            if antIdx==i:
                dir = directions[timeIdx,dirIdx]
                ax.scatter(dir[0],dir[1],c=dtec[datumIdx],s=(data*50)**2,vmin=0.25,vmax=0.5)
        i += 1
    plt.show()

if __name__ == '__main__':
    dataPack = PrepareData(infoFile='SB120-129/WendysBootes.npz',
                           dataFolder='SB120-129/',
                           timeStart = 1, timeEnd = 1,
                           arrayFile='arrays/lofar.hba.antenna.cfg',load=True)
    plotDataPack(dataPack)
   
        


# In[1]:

import os


# In[3]:

help(os.path.isfile)


# In[ ]:



