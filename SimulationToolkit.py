
# coding: utf-8

# In[5]:

'''
Simulation toolkit for Ionospheric Tomography Package (IonoTomo)
This contains the meat of math.
Created by Joshua G. Albert - albert@strw.leidenuniv.nl
'''
#reload()
import numpy as np
from sys import stdout
import json
import h5py
import os
import astropy.time as at
import astropy.coordinates as ac
import astropy.units as au

from scipy.interpolate import griddata
from scipy.signal import resample

#Things we've made
from Logger import Logger
from Atmosphere import Atmosphere
from SkyModel import SkyModel
from RadioArray import RadioArray

def fft( data ):
    return np.fft.fftshift( np.fft.fft2(  data  ) )

def ifft( data ):
    return np.fft.ifft2( np.fft.ifftshift( data ) ) 
        
def sumLayers(A,dim,lower,upper,heights):
    '''Given array, get the integrate content along a given direction between
    lower and upper points.
    Uses Simpsons composite rule'''
    #array with dim axis moved to front
    h = np.abs(np.mean(heights[1:] - heights[:-1]))
    view = np.rollaxis(A,dim,0)
    resShape = view.shape[1:]
    mask = (lower <= heights) * (heights < upper)
    #print mask
    #print lower,upper
    #print np.sum(mask)
    if np.sum(mask) == 0:
        #nothing in mask (zero sum)
        return np.zeros(resShape)
    if np.sum(mask) == 1:
        #another case that could mess up simpsons rule
        return view[mask]
    view = view[mask]
    if view.shape[0] % 2 == 0:
        i = 1
        res = view[0] + view[view.shape[0]-1]
    else:
        i = 2
        res = (view[0]+view[1])/2. + view[1] + view[view.shape[0]-1]
    j = i
    while j < view.shape[0]:
        res += 4*view[j]
        j += 2
    j = i + 1
    while j < view.shape[0]-1:
        res += 2*view[j]
        j += 2
    return h/3.*res

def regrid(A,shape,*presampledAx):
    '''Uses fft to regrid ...'''
    n = len(shape)
    if len(presampledAx) != n:
        print("wrongsize sample axis")
        return
    B = np.copy(A)
    resampledAx = []
    i = 0
    while i < n:
        B,t = resample(B,shape[i],t=presampledAx[i],axis=i)
        resampledAx.append(t)
        i += 1
    return B,resampledAx

def taiTimeFromMs(julianSeconds):
    '''Mj seconds'''
    return at.Time(julianSeconds - 3822681618.9974928,format='gps',scale='tai')

def julianSeconds2Days(julianSeconds):
    return julianSeconds/86400. - 2./24.

def julianDays2Seconds(julianDays):
    86400.*(julianDays + 2./24.)
    


class Simulation(object):
    def __init__(self,simConfigJson=None,logFile=None,help=False,**args):
        #super(Simulation,self).__init__(logFile=logFile)
        #logger.__init__(self,logFile=logFile)
        if help:
            self.getAttributes(help)
            exit(0)
        logger = Logger(logFile)
        self.log = logger.log
        self.speedoflight = 299792458.
        self.simConfigJson = simConfigJson
        self.loadSimConfigJson(simConfigJson)
        if self.simConfigJson is None:
            self.simConfigJson = "SimulationConfig.json"
        #overwrite possible json params with non-None args now
        self.initializeSimulation(**args)
        #create working directory
        try:
            os.makedirs(self.workingDir)
            self.workingDir = os.path.abspath(self.workingDir)
        except:
            self.workingDir = os.path.abspath(self.workingDir)
            self.log("Working directory already exists (beware of overwrite!): {0}".format(self.workingDir)) 
        #start simulation
        self.startSimulation()

    def getAttributes(self,help = False):
        '''Store the definition of attributes "name":[type,default]'''
        self.attributes = {#'maxLayer':[int,0],
                  #'minLayer':[int,0],
                  'skyModelFile':[str,""],
                  'obsStart':[lambda s:at.Time(s,format='isot'),"2000-01-01T00:00:00.000"],
                  'sampleTime':[float,1.],
                  'obsLength':[float,600.],
                  'pointing':[np.array,np.array([0.,0.])],
                  'layerHeights':[np.array,np.array([])],
                  #'maxTime':[int,0],
                  #'minTime':[int,0],
                  'frequency':[float,150e6],
                  'wavelength':[float,0.214],
                  'arrayFile':[str,""],
                  'workingDir':[str,"./output"],
                  #'dataFolders':[list,[]],
                  'dataFolder':[str,""],
                  'precomputed':[bool,False],
                  'loadAtmosphere':[bool,False],
                  'atmosphereData':[str,""],
                  'msFile':[str,""],
                  'restrictTimestamps':[int,0],
                  'fieldId':[int,0]}
        if help:
            self.log("Attribute:[type, default]")
            self.log(self.attributes)
        return self.attributes
    def loadFromMs(self,msFile):
        '''Take observation parameters from a measurement file'''
        try:
            import pyrap.tables as pt
        except:
            self.log("Unable to import pyrap. Not loading ms file")
            return 1
        try:
            tab = pt.table(msFile,readonly=True)
        except:
            self.log("MsFile: {0} does not exist".format(msFile))
            return 2
        #time slices
        self.log("Setting time slices")
        self.sampleTime = tab.getcol('EXPOSURE')[0]
        #self.timeInit = at.Time(julianSeconds2Days(np.min(tab.getcol('TIME'))),format='jd',scale='tai')
        self.timeInit = taiTimeFromMs(np.min(tab.getcol('TIME')))
        if self.restrictTimestamps > 0:
            #at.Time(np.min(tab.getcol('TIME')),format='gps',scale='tai')
            self.timeSlices = at.Time(np.linspace(self.timeInit.gps,self.timeInit.gps+self.restrictTimestamps*self.sampleTime,self.restrictTimestamps+1),format='gps',scale='tai')
        else:
            #get unique times. for now just use inti and default
            nsample = np.ceil(self.obsLength/self.sampleTime)
            self.timeSlices = at.Time(np.linspace(self.timeInit.gps,self.timeInit.gps+nsample*self.sampleTime,nsample+1),format='gps',scale='tai')
        self.log(self.timeInit.isot)
        tab.close()
        #take fieldId as the main field
        try:
            tabField = pt.table("{0}/FIELD".format(msFile),readonly=True)
        except:
            self.log("MsFile: {0}/FIELD does not exist".format(msFile))
            return 3
        
        self.pointing = tabField.getcol('PHASE_DIR')[self.fieldId,0,:]*180./np.pi
        self.fieldName = tabField.getcol('NAME')[self.fieldId]
        tabField.close()
        self.log("Got pointing for field {0}: {1}".format(self.fieldName, self.pointing))
        #radio array
        try:
            tabAntenna= pt.table("{0}/ANTENNA".format(msFile),readonly=True)
        except:
            self.log("MsFile: {0}/ANTENNA does not exist".format(msFile))
            return 4
        antennaPos = tabAntenna.getcol('POSITION')
        self.log("Setting radio array")
        self.radioArray = RadioArray(antennaPos=antennaPos,log = self.log)
        self.radioArray.calcBaselines(self.timeSlices,self.pointing)
        self.frames = self.radioArray.frames
        tabAntenna.close()
        self.log("Finished loading parameters from msFile {0}".format(msFile))
        return 0
        
        
    def initializeSimulation(self,**args):
        '''Set up variables here that will hold references throughout'''
        attributes = self.getAttributes()
        for attr in attributes.keys():
            #if in args and non-None then overwrite what is there
            if attr in args.keys():
                if args[attr] is not None:#see if attr already inside, or else put default
                    #setattr(self,attr,getattr(self,attr,attributes[attr][1]))
                    try:
                        setattr(self,attr,attributes[attr][0](args[attr]))
                        #self.log("Set: {0} -> {1}".format(attr,attributes[attr][0](args[attr])))
                    except:
                        self.log("Could not convert {0} into {1}".format(args[attr],attributes[attr][0]))
            else:
                #already set of setting to default
                setattr(self,attr,getattr(self,attr,attributes[attr][0](attributes[attr][1])))
                #self.log("Set: {0} -> {1}".format(attr,getattr(self,attr)))
                
    def startSimulation(self):
        '''Set things to get simulation on track'''

        #set current sim directory dataFolder, and save the settings
        if self.dataFolder == "":
            i = 0
            while self.dataFolder == "":
                dataFolder = "{0}/{1}".format(self.workingDir,i)
                try:
                    os.makedirs(dataFolder)
                    self.dataFolder = os.path.abspath(dataFolder)
                except:
                    self.log("data folder already exists (avoiding overwrite!): {0}".format(dataFolder))
                if i > 1000:
                    self.log("Many data folders. May be a lock causing failed mkdir!")
                    exit(1)
                i += 1
        else:
            self.dataFolder = "{0}/{1}".format(self.workingDir,self.dataFolder.split('/')[-1])
        if not os.path.isdir(self.dataFolder):
            try:
                os.makedirs(self.dataFolder)
                self.log("Making data directory {0}".format(self.dataFolder))
            except:
                self.log("Failed to create {0}: ".format(self.dataFolder))
                exit(1)
            self.dataFolder = os.path.abspath(self.dataFolder)
        self.log("Using data folder: {0}".format(self.dataFolder))
        simConfigJson = "{0}/{1}".format(self.dataFolder,self.simConfigJson.split('/')[-1])
        if os.path.isfile(simConfigJson):
            self.log("Found config file in data folder!")
            self.loadSimConfigJson(simConfigJson)
        
        #try to load msFile
        loadSucceed = False
        if self.msFile is not "":
            if not self.loadFromMs(self.msFile):
                loadSucceed = True

        
        if not loadSucceed:
            #time slices
            self.log("Setting time slices")
            nsample = np.ceil(self.obsLength/self.sampleTime)
            self.timeInit = at.Time(self.obsStart,format='isot',scale='tai')
            self.timeSlices = at.Time(np.linspace(self.timeInit.gps,self.timeInit.gps+nsample*self.sampleTime,nsample+1),format='gps',scale='tai')
            #set radio array
            self.log("Setting radio array")
            if self.arrayFile is not "":
                self.radioArray = RadioArray(arrayFile = self.arrayFile,
                                             log =self.log)
                self.radioArray.calcBaselines(self.timeSlices,self.pointing)
                self.frames = self.radioArray.frames
            else:
                self.radioArray = RadioArray(log = self.log)
        #set frequency
        self.log("Setting frequency")
        try:
            self.setFrequency(self.frequency)
        except:
            self.setWavelength(self.wavelength)

        #set skymodel
        self.log("Setting sky model")
        self.skyModel = SkyModel(self.skyModelFile,
                                 log=self.log)
        
            
        
        #layers
        self.log("Setting layers")
        layerHeights = []
        for height in self.layerHeights:
            if height not in layerHeights:
                layerHeights.append(height)
        self.layerHeights = np.sort(np.array(layerHeights))
        #atmosphere is a box centered over the radio array, it makes data at timeslices and layers
        self.log("Setting atmosphere")
        fovDim = self.radioArray.getFov(self.getWavelength())*self.layerHeights[-1]*1000.#m
        self.atmosphere = Atmosphere(radioArray=self.radioArray,
                                     boxSize=(fovDim,fovDim,self.layerHeights[-1]*1000.),
                                     times=self.timeSlices,
                                     log=self.log,
                                     wavelength=self.getWavelength(),
                                     atmosphereData="{0}/{1}".format(self.dataFolder,self.atmosphereData),
                                     loadAtmosphere = self.loadAtmosphere)
        #saving and starting
        self.saveSimConfigJson(simConfigJson)
        self.log("Starting computations... please wait.")
        self.compute()
        self.log("Finished computations... enjoy.")
        
    def restart(self):
        self.log("Resetting simulation...")
        self.startSimulation()
        #change things 
                
    #def log(self,message):
    #    stdout.write("{0}\n".format(message))
    #    stdout.flush()
    def loadSimConfigJson(self,simConfigJson):
        '''extract sim config from json, and then call initializeSimulation'''
        if simConfigJson is None:
            return
        try:
            f = open(simConfigJson,'r')
        except:
            self.log("No file: {0}".format(simConfigJson))
            exit(1)
        try:
            jobject = json.load(f)
            self.log("Loaded from {0}:\n{1}".format(simConfigJson,json.dumps(jobject,sort_keys=True, indent=4, separators=(',', ': '))))
            self.initializeSimulation(**jobject)
        except:
            self.log("File corrupt: {0}".format(simConfigJson))
        f.close()
        
        
    def saveSimConfigJson(self,simConfigJson):
        '''Save config in a json to load later.'''
        #self.log("Saving configuration in {0}".format(simConfigJson))
        try:
            jobject = {}
            attributes = self.getAttributes()
            for attr in attributes.keys():
                jobject[attr] = getattr(self,attr,attributes[attr][0](attributes[attr][1]))
                if type(jobject[attr]) == np.ndarray:#can't store np.array
                    jobject[attr] = list(jobject[attr])
                if type(jobject[attr]) == at.core.Time:#can't store np.array
                    jobject[attr] = jobject[attr].isot
            try:
                f = open(simConfigJson,'w')
                json.dump(jobject,f,sort_keys=True, indent=4, separators=(',', ': '))
                self.log("Saved configuration in: {0}".format(simConfigJson))
                self.log("Stored:\n{0}".format(json.dumps(jobject,sort_keys=True, indent=4, separators=(',', ': '))))
            except:
                self.log("Can't open file: {0}".format(simConfigJson))
        except:
            self.log("Could not get configuration properly.")
    def setWavelength(self,wavelength):
        '''set lambda in m'''
        self.wavelength = wavelength
        self.frequency = self.speedoflight/self.wavelength
    def setFrequency(self,frequency):
        '''set nu in Hz'''
        self.frequency = frequency
        self.wavelength = self.speedoflight/self.frequency
    def getWavelength(self):
        '''return lambda in m'''
        return self.wavelength
    def getFrequency(self):
        '''return nu in Hz'''
        return self.frequency
    def getMinLayer(self):
        '''return the minimum layer, usually zero'''
        return 0
    def getMaxLayer(self):
        return len(self.layerHeights)-1
    def getMinTime(self):
        return 0
    def getMaxTime(self):
        return len(self.timeSlices)-1
    def getVisibilitiesPhase(self,timeIdx,layerIdx):
        '''Returns visibilities <E*E^*->, (x1,x2,'units'), (y1,y2,'units')'''
        vmin = np.inf
        vmax = -np.inf
        i = 0
        while i < len(self.layerHeights):
            vmin = min(vmin,np.min(self.visibility[timeIdx][i]))
            vmax = max(vmax,np.max(self.visibility[timeIdx][i]))
            i += 1
        xrad = (self.layerHeights[layerIdx]*1000. + self.atmosphere.arrayHeight)*self.atmosphere.xangle/self.getWavelength()/2.
        yrad = (self.layerHeights[layerIdx]*1000. + self.atmosphere.arrayHeight)*self.atmosphere.xangle/self.getWavelength()/2.
        return (np.angle(self.visibility[timeIdx][layerIdx]),(-xrad,xrad,'lambda'),(-yrad,yrad,'lambda'),(vmin,vmax))
    def getIntensity(self,timeIdx,layerIdx):
        '''Returns intensity <E.E>, (x1,x2,'units'), (y1,y2,'units')'''
        vmin = np.inf
        vmax = -np.inf
        i = 0
        while i < len(self.layerHeights):
            vmin = min(vmin,np.min(self.intensity[timeIdx][i]))
            vmax = max(vmax,np.max(self.intensity[timeIdx][i]))
            i += 1
        xrad = (self.layerHeights[layerIdx]*1000. + self.atmosphere.arrayHeight)*self.atmosphere.xangle/2.
        yrad = (self.layerHeights[layerIdx]*1000. + self.atmosphere.arrayHeight)*self.atmosphere.yangle/2.
        return (self.intensity[timeIdx][layerIdx],(-xrad,xrad,'m'),(-yrad,yrad,'m'),(vmin,vmax))
    def getTau(self,timeIdx,layerIdx):
        '''Returns optical thickness or transfer function, tau, (x1,x2,'units'), (y1,y2,'units')'''
        vmin = np.inf
        vmax = -np.inf
        i = 0
        while i < len(self.layerHeights):
            vmin = min(vmin,np.min(self.tau[timeIdx][i]))
            vmax = max(vmax,np.max(self.tau[timeIdx][i]))
            i += 1
        xrad = (self.layerHeights[layerIdx]*1000. + self.atmosphere.arrayHeight)*self.atmosphere.xangle/2.
        yrad = (self.layerHeights[layerIdx]*1000. + self.atmosphere.arrayHeight)*self.atmosphere.yangle/2.
        return (self.tau[timeIdx][layerIdx],(-xrad,xrad,'m'),(-yrad,yrad,'m'),(vmin,vmax))
    def getElectronDensity(self,timeIdx,layerIdx):
        '''Returns electron density, (x1,x2,'units'), (y1,y2,'units')'''
        vmin = np.inf
        vmax = -np.inf
        i = 0
        while i < len(self.layerHeights):
            vmin = min(vmin,np.min(self.electronDensity[timeIdx][i]))
            vmax = max(vmax,np.max(self.electronDensity[timeIdx][i]))
            i += 1
        xrad = (self.layerHeights[layerIdx]*1000. + self.atmosphere.arrayHeight)*self.atmosphere.xangle/2.
        yrad = (self.layerHeights[layerIdx]*1000. + self.atmosphere.arrayHeight)*self.atmosphere.yangle/2.
        return (self.electronDensity[timeIdx][layerIdx],(-xrad,xrad,'m'),(-yrad,yrad,'m'),(vmin,vmax))
    def getRefractiveIndex(self,timeIdx,layerIdx):
        '''Returns refractive index at frequency, (x1,x2,'units'), (y1,y2,'units')'''
        vmin = np.inf
        vmax = -np.inf
        i = 0
        while i < len(self.layerHeights):
            vmin = min(vmin,np.min(self.refractiveIndex[timeIdx][i]))
            vmax = max(vmax,np.max(self.refractiveIndex[timeIdx][i]))
            i += 1
        xrad = (self.layerHeights[layerIdx]*1000. + self.atmosphere.arrayHeight)*self.atmosphere.xangle/2.
        yrad = (self.layerHeights[layerIdx]*1000. + self.atmosphere.arrayHeight)*self.atmosphere.yangle/2.
        return (self.refractiveIndex[timeIdx][layerIdx],(-xrad,xrad,'m'),(-yrad,yrad,'m'),(vmin,vmax))
    def getLayerHeight(self,layerIdx):
        '''Get layer height in km'''
        return self.layerHeights[layerIdx]
    def getTimeSlice(self,timeIdx):
        '''return time in seconds since start of simulation'''
        return self.timeSlices[timeIdx].gps - self.timeInit.gps
    def compute(self):
        '''Given the parameters simulate the ionotomo, or load if precomputed'''
        if self.precomputed:
            try:
                self.loadResults()
                return
            except:
                self.log("Failed to load precomputed results.")
                exit(1)
        #save before running
        #simConfigJson = "{0}/{1}".format(self.dataFolder,self.simConfigJson.split('/')[-1])
        #self.saveSimConfigJson(simConfigJson)
        self.electronDensity = {}
        self.visibility = {}
        self.intensity = {}
        self.tau = {}
        self.refractiveIndex = {}
        #computing atmosphere or loading or whatever
        self.log("Calculating electron density/refractive index/tau per layer.")
        self.atmosphere.run()
        heights = self.atmosphere.heights - self.atmosphere.arrayHeight#heights are from earth center, so takea way array center
        #s = ac.SkyCoord(ra = pointing[0]*au.deg,dec=pointing[1]*au.deg,frame='icrs')
        #interfaces,layers = atmosphere2Layers(self.atmosphere,self.heights,self.frames,s)
        #refractive scale constant m^3Hz^2
        return
        refractiveScale = (self.atmosphere.eCharge/(2*np.pi))**2/self.atmosphere.epsilonPerm/self.atmosphere.eMass
        
        timeIdx = 0
        while timeIdx < len(self.timeSlices):
            electronDensity = []
            refractiveIndex = []
            tau = []
            layerIdx = 0
            prevLayerHeight = 0.
            while layerIdx < len(self.layerHeights):
                lowerLayerHeight = prevLayerHeight
                upperLayerHeight = self.layerHeights[layerIdx]*1000.#km to m
                shape = self.atmosphere.cells[timeIdx].shape
                B,t=regrid(self.atmosphere.cells[timeIdx],
                                              [shape[0],shape[1],len(self.layerHeights)],
                                              self.atmosphere.longitudes,
                                              self.atmosphere.latitudes,
                                              self.atmosphere.heights)
                electronDensity.append(B)
                refractiveIndex.append(np.sqrt(1. - refractiveScale*electronDensity[-1]/self.getFrequency()**2))
                fineRefractiveIndex = np.sqrt(1. - refractiveScale*self.atmosphere.cells[timeIdx]/self.getFrequency()**2)
                B,t=regrid(fineRefractiveIndex,
                                              [shape[0],shape[1],len(self.layerHeights)],
                                              self.atmosphere.longitudes,
                                              self.atmosphere.latitudes,
                                              self.atmosphere.heights)
                tau.append(B)
                prevLayerHeight = upperLayerHeight
                layerIdx += 1
            self.electronDensity[timeIdx] = np.array(electronDensity)
            self.refractiveIndex[timeIdx] = np.array(refractiveIndex)
            self.tau[timeIdx] = np.array(tau)                
            timeIdx += 1
            
        #propagate distortions
        self.log("Propagating the phase distortions/sky model.")
        #determine image size
        #Umax = self.radioArray.baselines....
        #propagate 1,0
        A = self.skyModel
        Umax = 25000/self.getWavelength()#gmrt make auto 
        Vmax = 25000/self.getWavelength()
        lres = 1./Umax#rad
        mres = 1./Vmax#rad
        cellsize = 2#shannon sampling
        Nl = int(np.ceil(self.radioArray.getFov(self.getWavelength())/lres))*cellsize
        Nm = int(np.ceil(self.radioArray.getFov(self.getWavelength())/mres))*cellsize
        l = np.linspace(-self.radioArray.getFov(self.getWavelength())/2.,
                        self.radioArray.getFov(self.getWavelength())/2.,Nl)
        m = np.linspace(-self.radioArray.getFov(self.getWavelength())/2.,
                        self.radioArray.getFov(self.getWavelength())/2.,Nm)
        L,M = np.meshgrid(l,m)
        u = np.fft.fftshift(np.fft.fftfreq(Nl,d=np.abs(l[1]-l[0])))
        v = np.fft.fftshift(np.fft.fftfreq(Nm,d=np.abs(m[1]-m[0])))
        U,V = np.meshgrid(u,v)
        return
        timeIdx = 0
        while timeIdx < len(self.timeSlices):
            self.log("Computing time index: {0} of {1}".format(timeIdx, len(self.timeSlices)))
            Isky = self.skyModel.angularIntensity(L,M,self.radioArray.frames[timeIdx],self.pointing,self.getFrequency())
            Asky = np.sqrt(Isky)#arbitrary choice
            Usky = ifft(Asky)#some phase offset arbitrary
            #self.radioArray.calcBaselines(self.)
            self.intensity[timeIdx] = np.zeros([len(self.layerHeights),Nl,Nm])
            self.visibility[timeIdx] = np.zeros_like(self.intensity[timeIdx])
            layerIdx = len(self.layerHeights) - 1
            self.visibility[timeIdx][layerIdx] = ifft(ISky)
            k = 2.*np.pi/self.getWavelength()
            Aprev = fft(Usky)
            zprev = self.layerHeights[-1]
            refractiveIndexPrev = 1.
            while layerIdx >= 0:
                #propagate from above to current layer
                zdiff = zprev - self.layerHeights[layerIdx]
                zprev = self.layerHeights[layerIdx]
                propKernel = np.exp(1j*k*np.sqrt((1j-L**2 - M**2)) * zdiff)
                A = Aprev * propKernel
                transferKernel = np.sqrt(refractiveIndexPrev/self.refractiveIndex[timeIdx][layerIdx])*np.exp(-1j*self.tau[timeIdx][layerIdx])
                refractiveIndexPrev = self.refractiveIndex[timeIdx][layerIdx]
                #regrid transfer kernel
                x = np.arange(self.refractiveIndex[timeIdx][layerIdx].shape[0])
                y = np.arange(self.refractiveIndex[timeIdx][layerIdx].shape[1])
                X,Y = np.meshgrid(x,y)
                points = np.transpose(np.array([X.flatten(),Y.flatten()]))
                x = np.arange(A.shape[0])
                y = np.arange(A.shape[0])
                gridX,gridY = np.meshgrid(x,y)
                transferKernel = griddata(points,transferKernel.flatten(),(gridX,gridY))
                U = ifft(A) * transferKernel
                self.intensity[timeIdx][layerIdx,:,:] = np.abs(U)**2
                self.visibility[timeIdx][layerIdx,:,:] = ifft(np.abs(A)**2)
                Aprev = fft(U)
                layerIdx -= 1
            timeIdx += 1
        self.log("Finished computing.")

        
    def loadResults(self):
        '''Load the results from hdf5 file.'''
        assert False,"loadResults not implemented yet"


if __name__ == '__main__':
    A = np.random.uniform(size=[5,6,7])
    heights = np.arange(6)
    lower = 1
    upper = 4
    print A.shape
    print "sum along 1 dim"
    print sumLayers(A,1,lower,upper,heights).shape

