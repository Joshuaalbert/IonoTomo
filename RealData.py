
# coding: utf-8

# In[2]:

import glob
from RadioArray import RadioArray
from ENUFrame import ENU
import numpy as np
import astropy.units as au
import astropy.time as at
import astropy.coordinates as ac
from Geometry import *

def getDatumIdx(antIdx,dirIdx,timeIdx,numDirections,numTimes):
    '''standarizes indexing'''
    idx = antIdx*numDirections*numTimes + dirIdx*numTimes + timeIdx
    return idx
    
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
            TecData = np.load(dataFile)
            data = TecData['data']
            rays = TecData['rays']
            stationIndices = TecData['stationIndices']
            timeIndices = TecData['timeIndices']
            directionIndices = TecData['directionIndices']
            obstimes = TecData['obstimes']
            numAntennas=TecData['numAntennas']
            numDirections=TecData['numDirections']
            numTimes=TecData['numTimes']
            generate = False
        except:
            print(TecData.keys())
    if generate:
        #things to grab
        outAntennas = {}
        outDirections = {}
        outTimes = {}

        enu = ENU(location=radioArray.getCenter().earth_location)
        print("ENU system set: {0}".format(enu))
        numRays = 0
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
        #assume all times and antennas are same in each datafile
        recievers = []
        numTimes =  (timeEnd - timeStart + 1)
        print("Number of time stamps: {0}".format(numTimes))
        #each time gives a different direction for each patch
        numDirs = numTimes * numPatches #maybe a file doesn't load
        print("Number of possible directions: {0}".format(numDirs))
        data = []
        rays = []
        obstimes = []
        stationIndices = []
        timeIndices = []
        directionIndices = []
        patchIdx = 0
        failed = 0
        while patchIdx < numPatches:
            patch = patches[patchIdx]
            rd = radec[patchIdx]
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
                dir = ac.SkyCoord(rd.ra,rd.dec,obstime=time,frame='icrs').transform_to(enu)
                
                print("Patch elevation: {0}".format(dir.elevation))
                numRays += 1
                print("Patch east: {0} north: {1} up: {2}".format(dir.east,dir.north,dir.up))
                dir = dir.transform_to('itrs')
                
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
                    stationLoc = ac.SkyCoord(*stationLocs[labelIdx]*au.m,frame='itrs')
                    origin = stationLoc.cartesian.xyz.to(au.km).value#/wavelength enu system
                    rays.append(Ray(origin,dir.cartesian.xyz.value,id = datumIdx,time = time.gps - times[0]))
                    data.append(tecData[timeIdx,antIdx])#-tecData[timeIdx,0])#relative to first antenna?
                    obstimes.append(time.gps - times[0])
                    stationIndices.append(labelIdx)
                    timeIndices.append(timeIdx)
                    directionIndices.append(patchIdx)
                    antIdx += 1
                timeIdx += 1
            patchIdx += 1
        numAntennas=len(antennas)
        numDirections=numPatches
        numTimes=numTimes
        np.savez(dataFile, numAntennas=numAntennas,numDirections=numDirections,numTimes=numTimes,
                 rays=rays,data=data,stationIndices=stationIndices,
                timeIndices=timeIndices,directionIndices=directionIndices,obstimes = obstimes)
    return {'numAntennas':numAntennas,'numDirections':numDirections,'numTimes':numTimes,
            'rays':rays,'dtec':data,'times':obstimes,'radioArray':radioArray}

if __name__ == '__main__':
    dataDict = PrepareData(infoFile='SB120-129/WendysBootes.npz',
                           dataFolder='SB120-129/',
                           timeStart = 0, timeEnd = 0,
                           arrayFile='arrays/lofar.hba.antenna.cfg',load=False)

