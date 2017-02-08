
# coding: utf-8

# In[58]:

import astropy.coordinates as ac
import astropy.units as au
import astropy.time as at
import numpy as np

class RadioArray(object):
    '''Handles the radio array object'''
    def __init__(self,arrayFile = None,antennaPos=None,logger = None,name = None,msFile=None,numAntennas=0,earthLocs=None,frequency=120e6):
        self.frequency = frequency
        if logger is not None:
            try:
                self.log = logger.log
            except:
                print("Creating logger")
                from Logger import Logger
                logger = Logger()
                self.log = logger.log
        else:
            self.log = None
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
            self.labels = d['station_label']
            self.locs = ac.SkyCoord(x=d['X']*au.m,y=d['Y']*au.m,z=d['Z']*au.m,frame='itrs')
            self.Nantenna = np.size(d['X'])
        except:
            types = np.dtype({'names':['X','Y','Z','diameter','station_label'],
                             'formats':[np.double,np.double,np.double,np.double,'S16']})
            d = np.genfromtxt(arrayFile,comments = '#',usecols=(0,1,2))
            self.locs = ac.SkyCoord(x=d[:,0]*au.m,y=d[:,1]*au.m,z=d[:,2]*au.m,frame='itrs')
            self.Nantenna = d.shape[0]
            self.labels = np.arange(self.Nantenna)
            self.diameters = None
        self.calcCenter()
        
    def getFov(self,frequency=120e6):
        '''get the field of view in radians'''
        return 4.*np.pi/180.
    
    def saveArrayFile(self,arrayFile):
        locs = self.locs.cartesian.xyz.to(au.m).value.transpose()
        array = np.hstack([locs[0,:],locs[1,:],locs[2,:],self.diameters,self.labels])
        np.savetxt(arrayFile, array, fmt=['%.18e','%.18e','%.18e','%.6e','%s'], delimiter=',', newline='\n', header='ITRS (m)\nX,Y,Z', footer='', comments='# ')
        
    def loadPosArray(self,antennaPos):
        '''Load pos is shape (N,3), typically grabbed from a ms/ANTENNA table'''
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
        try:
            return self.center
        except:
            self.calcCenter()
            self.log("Center of array: {0}".format(self.center))
            return self.center

if __name__=='__main__':
    #from Logger import Logger
    #logger = Logger()
    radioArray = RadioArray(arrayFile='arrays/gmrtPos.csv')
    print(radioArray.getCenter().earth_location.geodetic)
    print(radioArray.getCenter())
    radioArray = RadioArray(arrayFile='arrays/lofar.hba.antenna.cfg')
    print(radioArray.getCenter().earth_location.geodetic)
    print('WGS84',radioArray.center.earth_location.to_geodetic('WGS84'))
    print('WGS84',radioArray.center.earth_location.geodetic[1].deg)
    from ENUFrame import ENU
    enu = ENU(obstime=at.Time(0,format='gps'),location=radioArray.getCenter().earth_location)
    aa = ac.AltAz(obstime=at.Time(0,format='gps'),location=radioArray.getCenter().earth_location)
    print(ac.SkyCoord(alt=90*au.deg,az=0*au.deg,frame=aa).transform_to('icrs'))
    print(ac.SkyCoord(alt=90*au.deg,az=0*au.deg,frame=aa).transform_to(enu).transform_to('icrs'))
    print(ac.SkyCoord(east=0,north=0,up=1,frame=enu).transform_to('icrs').dec)
    #print radioArray.center.earth_location.height
    #times = at.Time([0,2,4]*au.s,format='gps',scale='utc')
    #radioArray.calcBaselines(times,np.array([12,62]))
    #v = radioArray.baselines[0,:,:,1]
    #testBaselines()

