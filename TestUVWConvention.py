
# coding: utf-8

# In[ ]:

from Geometry import itrs2uvw
import numpy as np

def altaz2hourangledec(alt,az,lat):
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

