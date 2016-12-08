
# coding: utf-8

# In[17]:

import astropy.coordinates as ac
import astropy.units as au
import astropy.time as at
import numpy as np

from Geometry import itrs2uvw

def altz2hourangledec(alt,az,lat):
    H = np.arctan2(-np.sin(az)*np.cos(alt), 
                   -np.cos(az)*np.sin(lat)*np.cos(alt)+np.sin(alt)*np.cos(lat))
    if H<0:
        if H != -1:
            H += 2*np.pi
    #H[H[H<0] != -1] += np.pi*2
    H = np.mod(H,np.pi*2)
    dec = np.arcsin(np.sin(lat)*np.sin(alt) + np.cos(lat)*np.cos(alt)*np.cos(az))
    return H,dec

def calcUVWPositions(pointing,obsLoc,obsTime,antennaLocs):
    '''Pointing is [ra, dec] of phase_tracking center.
    obsLoc is ITRS of the location of the origin of the tangent plane,defined at obsTime
    obsTime is the time object
    antennaLocs are the ITRS locations of antennas'''
    
    lon = obsLoc.earth_location.geodetic[0].rad
    lat = obsLoc.earth_location.geodetic[1].rad
    
    frame = ac.AltAz(location = obsLoc, obstime = obsTime, pressure=None)
    s = ac.SkyCoord(ra = pointing[0]*au.deg,dec=pointing[1]*au.deg,frame='icrs').transform_to(frame)
    #sProj = np.eye(3) - np.outer(s.itrs.cartesian.xyz.value,s.itrs.cartesian.xyz.value)
    
    obsLoc_ = ac.SkyCoord(*obsLoc.earth_location.geocentric,obstime=obsTime,frame='itrs')
    antennaLocs_ = ac.SkyCoord(*antennaLocs.earth_location.geocentric,obstime=obsTime,frame='itrs')
    
    a = s.alt.rad
    A = s.az.rad
    lmst = obsTime.sidereal_time('mean',lon)
    hourangle = lmst.to(au.rad).value - pointing[0]*np.pi/180.
    declination = pointing[1]*np.pi/180.
    print "Hourangle, dec:",hourangle*180/np.pi,declination*180./np.pi
    
    refLoc = obsLoc_.itrs.cartesian.xyz.value
    
    Ruvw = itrs2uvw(hourangle,declination,lon,lat)
    print"u,v,w:",Ruvw
    uvw = np.zeros([len(antennaLocs),3])
    i = 0
    while i< len(antennaLocs):
        loc = antennaLocs_[i]
        #if loc.obstime.gps != obsTime.gps:
            #redefine to obsTime
        #    loc = ac.SkyCoord(*loc.earth_location.geocentric,obstime=obsTime,frame='itrs')
        relLoc = loc.itrs.cartesian.xyz.value - refLoc
        uvw[i,:] = Ruvw.dot(relLoc)
        #relLocENU = itrs2enu.dot(relLoc)
        #relLocEqatorial = transform.dot(relLoc)
        
        #h = sProj.dot(relLocENU)
        #uvw[i,0] = udir.dot(relLocENU)#relLoc.dot(eMod)#east part
        #uvw[i,1] = vdir.dot(relLocENU)
        #uvw[i,2] = wdir.dot(relLocENU)
        #uvw[i,1] = relLoc.dot(nMod)#north part
        #uvw[i,2] = relLoc.dot(s.itrs.cartesian.xyz.value)#w
        #uvw[i,:] = P.transpose().dot(relLoc)
        i += 1
    return uvw
    
def testBaselines():
    time = at.Time("2015-03-09T23:38:07.55",format="isot",scale='utc')
    obsLoc = ac.SkyCoord(1.657e+06*au.m,5.79789e+06*au.m,2.0733e+06*au.m,frame='itrs')
    antenna = ac.SkyCoord([1657011.549,1657017.879]*au.m,[5798582.2601,5798220.8201]*au.m,[2073283.1305,2073262.8205]*au.m,frame='itrs')
    #antenna2 = ac.SkyCoord(1657017.879*au.m,5798220.8201*au.m,2073262.8205*au.m,frame='itrs')
    pointing = (202.78453379, 30.50915556)
    uvw = calcUVWPositions(pointing,obsLoc,time,antenna)
    print -uvw[0,:]+uvw[1,:]
    time = at.Time("1999-12-31T17:30:00.00",format="isot",scale='tai')
    obsLoc = ac.SkyCoord(1*au.m,0*au.m,0*au.m,frame='itrs')
    antenna = ac.SkyCoord([0,0]*au.m,[0,0]*au.m,[0,1]*au.m,frame='itrs')
    #antenna2 = ac.SkyCoord(1657017.879*au.m,5798220.8201*au.m,2073262.8205*au.m,frame='itrs')
    pointing = (90, 0.)
    uvw = calcUVWPositions(pointing,obsLoc,time,antenna)
    print -uvw[0,:]+uvw[1,:]


class RadioArray(object):
    def __init__(self,arrayFile = None,antennaPos=None,log = None,name = None,msFile=None,numAntennas=0,earthLocs=None):
        
        if log is None:
            from Logger import Logger
            logger = Logger()
            self.log = logger.log
        else:
            self.log = log
        self.locs = []
        self.Nantenna = 0
        if arrayFile is not None:
            self.arrayFile = arrayFile
            self.loadArrayFile(arrayFile)
        if antennaPos is not None:
            self.loadPosArray(antennaPos)
            
    def loadArrayFile(self,arrayFile):
        '''Loads a csv where each row is x,y,z in geocentric coords of the antennas'''
        d = np.genfromtxt(arrayFile,usecols=(0,1,2))
        self.locs = ac.SkyCoord(x=d[:,0]*au.m,y=d[:,1]*au.m,z=d[:,2]*au.m,frame='itrs')
        self.Nantenna = d.shape[0]
        self.calcCenter()
    def getFov(self,wavelength):
        '''get the field of view in radians'''
        return 0.5*np.pi/180.
    def saveArrayFile(self,arrayFile):
        pass
    def loadPosArray(self,antennaPos):
        '''Load pos is shape (N,3), typically grabbed from a ms/ANTENNA table'''
        self.locs = ac.SkyCoord(x=antennaPos[:,0]*au.m,y=antennaPos[:,1]*au.m,z=antennaPos[:,2]*au.m,frame='itrs')
        self.Nantenna = antennaPos.shape[0]
        self.calcCenter()

    def calcBaselines(self,times,pointing):
        '''Ordering of baselines is colmajor stacking. 
        At each time, u,v,w point east,north, and out.
        ant(i) < ant(j) b_ij = x_j - x_i as per MS 2.0 definition
        (https://casa.nrao.edu/Memos/CoordConvention.pdf)
        '''
        self.baselines = np.zeros([len(times),self.Nantenna,self.Nantenna,3])
        self.antennaLocs = np.zeros([len(times),self.Nantenna,3])
        self.pointing = ac.SkyCoord(ra = pointing[0]*au.deg,dec=pointing[1]*au.deg,frame='icrs')
        s = self.pointing
        
        self.frames = []
        
        count = 0
        while count < len(times):
            #print('Calculating baselines for: {0}'.format(times[count]))
            time = times[count]
            self.antennaLocs[count,:,:] = calcUVWPositions(pointing,self.center,time,self.locs)                
            i = 0
            while i < self.Nantenna:
                j = 0
                while j < self.Nantenna:
                    self.baselines[count,i,j,:] = self.antennaLocs[count,j,:] - self.antennaLocs[count,i,:]
                    j += 1
                i += 1
            count += 1

    def calcCenter(self):
        '''calculates the centroid of the array based on self.locs returns the ITRS of center'''
        center = np.mean(self.locs.cartesian.xyz,axis=1)
        self.center = ac.ITRS(x=center[0],y=center[1],z=center[2])
        self.log("Center of array: {0}".format(self.center))
        #n = self.center.itrs.earth_location.geocentric.to(au.m).value
        #self.n = n/np.sqrt(n[0]**2 + n[1]**2 + n[2]**2)
        return self.center
    
    def getCenter(self):
        try:
            return self.center
        except:
            self.calcCenter()
            return self.center

if __name__=='__main__':
    from Logger import Logger
    logger = Logger()
    radioArray = RadioArray(arrayFile='arrays/gmrtPos.csv',log=logger.log)
    radioArray = RadioArray(arrayFile='arrays/lofar.hba.antenna.cfg',log=logger.log)
    print(radioArray.getCenter().earth_location.geodetic[2].value)
    #print radioArray.center.earth_location.height
    #times = at.Time([0,2,4]*au.s,format='gps',scale='utc')
    #radioArray.calcBaselines(times,np.array([12,62]))
    #v = radioArray.baselines[0,:,:,1]
    #testBaselines()


# In[ ]:



