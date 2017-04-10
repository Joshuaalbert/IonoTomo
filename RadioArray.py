
# coding: utf-8

# In[ ]:

import astropy.coordinates as ac
import astropy.units as au
import astropy.time as at
import numpy as np
import sys

class RadioArray(object):
    '''Handles the radio array object.'''
    def __init__(self,arrayFile = None,antennaPos=None,name = None,msFile=None,numAntennas=0,earthLocs=None,frequency=120e6):
        self.frequency = frequency#can be widebandwidth later
        self.Nantenna = 0
        if arrayFile is not None:
            self.arrayFile = arrayFile
            self.loadArrayFile(arrayFile)
        if antennaPos is not None:
            self.loadPosArray(antennaPos)
            
    def loadArrayFile(self,arrayFile):
        '''Loads a csv where each row is x,y,z in geocentric ITRS coords of the antennas'''
        
        try:
            types = np.dtype({'names':['X','Y','Z','diameter','station_label'],
                             'formats':[np.double,np.double,np.double,np.double,'S16']})
            d = np.genfromtxt(arrayFile,comments = '#',dtype=types)
            self.diameters = d['diameter']
            self.labels = d['station_label'].astype(str)
            self.locs = ac.SkyCoord(x=d['X']*au.m,y=d['Y']*au.m,z=d['Z']*au.m,frame='itrs')
            self.Nantenna = int(np.size(d['X']))
        except:
            d = np.genfromtxt(arrayFile,comments = '#',usecols=(0,1,2))
            self.locs = ac.SkyCoord(x=d[:,0]*au.m,y=d[:,1]*au.m,z=d[:,2]*au.m,frame='itrs')
            self.Nantenna = d.shape[0]
            self.labels = np.arange(self.Nantenna)
            self.diameters = None
        self.calcCenter()
        
    def getFov(self):
        '''get the field of view in radians'''
        return 4.*np.pi/180.
    
    def saveArrayFile(self,arrayFile):
        import time
        locs = self.locs.cartesian.xyz.to(au.m).value.transpose()
        f = open(arrayFile,'w')
        f.write('# Created on {0} by Joshua G. Albert\n'.format(time.strftime("%a %c",time.localtime())))
        f.write('# ITRS(m)\n')
        f.write('# X\tY\tZ\tdiameter\tlabels\n')
        i = 0
        while i < self.Nantenna:
            if self.diameters is not None:
                f.write('{0:1.9e}\t{1:1.9e}\t{2:1.9e}\t{3:1.4e}\t{4}'.format(locs[i,0],locs[i,1],locs[i,2],self.diameters[i],self.labels[i]))
            else:
                f.write('{0:1.9e}\t{1:1.9e}\t{2:1.9e}\t{3:d}\t{4}'.format(locs[i,0],locs[i,1],locs[i,2],-1,self.labels[i]))
            if i < self.Nantenna-1:
                f.write('\n')
            i += 1
        f.close()
        
    def loadPosArray(self,antennaPos):
        '''Load pos is shape (N,3), typically grabbed from a ms/ANTENNA table.
        Assumes it is in ITRS(m)'''
        self.locs = ac.SkyCoord(x=antennaPos[:,0]*au.m,y=antennaPos[:,1]*au.m,z=antennaPos[:,2]*au.m,frame='itrs')
        self.Nantenna = antennaPos.shape[0]
        self.calcCenter()

    def calcCenter(self):
        '''calculates the centroid of the array based on self.locs returns the ITRS of center'''
        center = np.mean(self.locs.cartesian.xyz,axis=1)
        self.center = ac.SkyCoord(x=center[0],y=center[1],z=center[2],frame='itrs')
        
        #n = self.center.itrs.earth_location.geocentric.to(au.m).value
        #self.n = n/np.sqrt(n[0]**2 + n[1]**2 + n[2]**2)
        return self.center
    
    def getCenter(self):
        '''Return the ITRS center of the array'''
        try:
            return self.center
        except:
            self.calcCenter()
            self.log("Center of array: {0}".format(self.center))
            return self.center
    
    def getAntennaIdx(self,name):
        '''Retrieve the index of the given name from labels.
        Returns None if no match.'''
        i = 0
        while i < self.Nantenna:
            if self.labels[i] == name:
                return i
            i += 1
        return None
    
    def getSunZenithAngle(self,time):
        '''Return the solar zenith angle in degrees at the given time.'''
        frame = ac.AltAz(location=self.getCenter().earth_location,obstime=time)
        sun = ac.get_sun(time).transform_to(frame)
        return 90. - sun.alt.deg
    
    def __repr__(self):
        return "Radio Array: {0:1.5e} MHz, Longitude {1:.2f} Latitude {2:.2f} Height {3:.2f}".format(self.frequency,*self.getCenter().earth_location.to_geodetic('WGS84'))

if __name__=='__main__':
    #from Logger import Logger
    #logger = Logger()
    radioArray = RadioArray(arrayFile='arrays/lofar.hba.antenna.cfg')
    print(radioArray.labels)

