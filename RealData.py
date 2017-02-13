
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

def getDatumIdx(antIdx,dirIdx,timeIdx,numDirections,numTimes):
    '''standarizes indexing'''
    idx = antIdx*numDirections*numTimes + dirIdx*numTimes + timeIdx
    return idx

def getDatum(datumIdx,numDirections,numTimes):
    timeIdx = datumIdx % numTimes
    dirIdx = (datumIdx - timeIdx)/numTimes % numDirections
    antIdx = (datumIdx - timeIdx - dirIdx*numTimes)/numDirections/numTimes
    return antIdx,dirIdx,timeIdx

class DataPack(object):
    def __init__(self,dataDict):
        '''get the astropy object defining rays and then also the dtec data'''
        self.antennas = dataDict['antennas']#a dictionary {antIdx:itrs,...}
        self.numAnt = len(self.antennas)
        self.times = dataDict['times']#a dictionary {timeIdx:time,...}
        self.numTimes = len(self.times)
        self.directions = dataDict['directions']#a dictionary {dirIdx:icrs}
        self.numDir = len(self.directions)
        self.dtec = dataDict['dtec'] #a dictionary {datumIdx: data}
        self.radioArray = dataDict['radioArray']
        
    def getDatumIdx(self,antIdx,dirIdx,timeIdx):
        '''Map antIdx, dirIdx, and timeIdx to a invertable index'''
        idx = antIdx*self.numDir*self.numTimes + dirIdx*self.numTimes + timeIdx
        return idx

    def getInvertDatumIdx(self,datumIdx):
        '''Map the datumIdx to specific antidx, dirIdx, and timeIdx'''
        timeIdx = datumIdx % self.numTimes
        dirIdx = (datumIdx - timeIdx)/self.numTimes % self.numDirections
        antIdx = (datumIdx - timeIdx - dirIdx*self.numTimes)/self.numDirections/self.numTimes
        return antIdx,dirIdx,timeIdx  
        
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
                    datumIdx = self.getDatumIdx(antIdx[i],timeIdx[j],dirIdx[k])
                    if datumIdx in self.dtec.keys():
                        output[i,j,k] = self.dtec[dataIdx]
                    else:
                        output[i,j,k] = np.nan
                    k += 1
                j += 1
            i += 1
        return output
    
    def get_antennas_array(self,antIdx=[]):
        if antIdx is -1:
            antIdx = self.antennas.keys()
        output = np.zeros([len(antIdx),3],dtype=np.double)
        i = 0
        while i < len(antIdx):
            if antIdx[i] in self.antennas.keys():
                output[i,:] = self.antennas[antIdx[i]].transform_to(ac.ITRS).cartesian.xyz.to(au.km).value
            else:
                output[i,:] = np.nan
            i += 1
        return output
    
    def get_times_array(self,timeIdx=[]):
        '''Get the ISO time'''
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
    
    def get_directions_array(self, timeIdx=[], dirIdx=[]):
        '''Get the array of directions in itrs'''
        if timeIdx is -1:
            timeIdx = self.times.keys()
        if dirIdx is -1:
            dirIdx = self.directions.keys()
        #output = np.zeros([len(dirIdx),3],dtype=np.double)
        output = np.zeros([len(timeIdx),len(dirIdx),3],dtype=np.double)
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
                    
        
    
def PrepareData(infoFile,dataFolder,timeStart = 0, timeEnd = 0,arrayFile='arrays/lofar.hba.antenna.cfg',load=False):
    '''Prepare data for continuous inversion. RadioArray, dobs, Cd, and rays.
    Tec is always relative to first antenna[0]
    Output coords are ENU frame'''
    
    print("creating radio array")
    radioArray = RadioArray(arrayFile)
    dataFile = "TecInversionData.npz"
    generate = True
    if load:
        print("Loading:",dataFile)
        try:
            #TecData = np.load(dataFile)
            #dataDict = TecData['dataDict']
            f = open(dataFile,'wb')
            dataDict = dill.load(f)
            f.close()
            generate = False
        except:
            #print(TecData.keys())
            pass
    if generate:

        #get patch names and directions for dataset
        info = np.load(infoFile)
        patches = info['patches']
        numPatches = len(patches)
        radec = info['directions']
        print("Loaded {0} patches".format(numPatches))
        #get array stations (shoud fold this into radioArray. todo)
        stationLabels = np.genfromtxt(arrayFile, comments='#',usecols = (4),dtype=type(""))
        stationLocs = np.genfromtxt(arrayFile, comments='#',usecols = (0,1,2))
        numStations = len(stationLabels)
        print("Number of stations in array: {0}".format(numStations))

        numTimes =  (timeEnd - timeStart + 1)
        print("Number of time stamps: {0}".format(numTimes))
        #each time gives a different direction for each patch
        numDirs = numTimes * numPatches #maybe a file doesn't load
        print("Number of possible directions: {0}".format(numDirs))
        
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
            
            #find the appropriate file
            files = glob.glob("{0}/*_{1}_*.npz".format(dataFolder,patch))
            if len(files) == 1:
                file = files[0]
            else:
                print('Could not find patch: {0}'.format(patch))
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
            #internal data of each patch file
            antennas = d['antennas']
            times = d['times'][timeStart:timeEnd+1]#gps tai
            tecData = d['data'][timeStart:timeEnd+1,:]#times x antennas
            timeIdx = 0
            while timeIdx < numTimes:

                time = at.Time(times[timeIdx],format='gps',scale='tai')
                print("Processing time: {0}".format(time.isot))
                
                # get direction of patch at time wrt fixed frame
                
                
                antIdx = 0#index in solution table
                while antIdx < len(antennas):
                    ant = antennas[antIdx]
                    #find index in stationLabels
                    labelIdx = 0
                    while labelIdx < numStations:
                        if stationLabels[labelIdx] == ant:
                            break
                        labelIdx += 1
                    if labelIdx >= numStations:
                        print("Could not find {0} in available stations: {1}".format(ant,stationLabels))
                        continue
                    datumIdx = getDatumIdx(antIdx,patchIdx,timeIdx,numPatches,numTimes)
                    
                    #ITRS WGS84
                    stationLoc = ac.SkyCoord(*stationLocs[labelIdx]*au.m,obstime=time,frame='itrs')
                    outAntennas[antIdx] = stationLoc
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
        #np.savez(dataFile, dataDict = dataDict)
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
   
        

