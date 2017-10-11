from ionotomo.astro.frames.uvw_frame import UVW
from ionotomo.astro.frames.pointing_frame import Pointing
from ionotomo.astro.frames.enu_frame import ENU
import astropy.coordinates as ac
import astropy.time as at
import astropy.units as u
import numpy as np

def test_enu():
    time = at.Time("2017-01-26T17:07:00.000",format='isot',scale='utc')
    loc = ac.EarthLocation(lon=10*u.deg,lat=0*u.deg,height=0*u.km)
    enu = ENU(location=loc,obstime=time)
#    print("With dim test:")
    enucoords = ac.SkyCoord(east = np.array([0,1])*u.m,
                            north=np.array([0,1])*u.m,
                            up=np.array([0,1])*u.m,frame=enu)
#    print (enucoords)
    #enucoords.transform_to('itrs')
    c2 = enucoords.transform_to('itrs').transform_to(enu)
    assert np.all(np.isclose(enucoords.cartesian.xyz.value, c2.cartesian.xyz.value))

#    print("Without dim test:")
    enucoords = ac.SkyCoord(east = np.array([0,1]),
                            north=np.array([0,1]),
                            up=np.array([0,1]),frame=enu)
    #print(enucoords.transform_to('itrs'))
    c2 = enucoords.transform_to('itrs').transform_to(enu)
    assert np.all(np.isclose(enucoords.cartesian.xyz.value, c2.cartesian.xyz.value))



def test_uvw():
    def compVectors(a,b):
        a = a.cartesian.xyz.value
        a /= np.linalg.norm(a)
        b = b.cartesian.xyz.value
        b /= np.linalg.norm(b)
        h = np.linalg.norm(a-b)
        return h < 1e-8
    # with X - East, Z - NCP and Y - Down
    time = at.Time("2017-01-26T17:07:00.000",format='isot',scale='utc')
    loc = ac.EarthLocation(lon=10*u.deg,lat=10*u.deg,height=0*u.km)
    enu = ENU(location=loc,obstime=time)
    x = ac.SkyCoord(1,0,0,frame=enu)
    z = ac.SkyCoord(0,np.cos(loc.geodetic[1].rad),np.sin(loc.geodetic[1].rad),frame=enu)
    #ncp = ac.SkyCoord(0*u.one,0*u.one,1*u.one,frame='itrs').transform_to(enu)
    y = ac.SkyCoord(0,np.sin(loc.geodetic[1].rad),-np.cos(loc.geodetic[1].rad),frame=enu)
    lst = ac.AltAz(alt=90*u.deg,az=0*u.deg,location=loc,obstime=time).transform_to(ac.ICRS).ra
    #ha = lst - ra
    print("a) when ha=0,dec=90  uvw aligns with xyz")
    ha = 0*u.deg
    ra = lst - ha
    dec = 90*u.deg
    phaseTrack = ac.SkyCoord(ra,dec,frame=ac.ICRS)
    uvw = UVW(obstime=time,location=loc,phase=phaseTrack)
    U = ac.SkyCoord(1,0,0,frame=uvw).transform_to(enu)
    V = ac.SkyCoord(0,1,0,frame=uvw).transform_to(enu)
    W = ac.SkyCoord(0,0,1,frame=uvw).transform_to(enu)
    assert compVectors(U,x),"fail test a, u != x"
    assert compVectors(V,y),"fail test a, v != y"
    assert compVectors(W,z),"fail test a, w != z"
    #print("passed a")
    #print("b) v, w, z are always on great circle")
    assert np.cross(V.cartesian.xyz.value,W.cartesian.xyz.value).dot(z.cartesian.xyz.value) < 1e-10, "Not on the great circle"
    #print("passed b")
    #print("c) when ha = 0 U points east")
    ha = 0*u.deg
    ra = lst - ha
    dec = 35*u.deg
    phaseTrack = ac.SkyCoord(ra,dec,frame=ac.ICRS)
    uvw = UVW(obstime=time,location=loc,phase=phaseTrack)
    U = ac.SkyCoord(1*u.m,0*u.m,0*u.m,frame=uvw).transform_to(enu)
    V = ac.SkyCoord(0*u.m,1*u.m,0*u.m,frame=uvw).transform_to(enu)
    W = ac.SkyCoord(0*u.m,0*u.m,1*u.m,frame=uvw).transform_to(enu)
    assert np.cross(V.cartesian.xyz.value,W.cartesian.xyz.value).dot(z.cartesian.xyz.value) < 1e-10, "Not on the great circle"
    east = ac.SkyCoord(1,0,0,frame=enu)
    assert compVectors(U,east),"fail test c, u != east"
    #print("passed c")
    #print("d) when dec=0 and ha = -6 w points east")
    ha = -6*u.hourangle
    ra = lst - ha
    dec = 0*u.deg
    phaseTrack = ac.SkyCoord(ra,dec,frame=ac.ICRS)
    uvw = UVW(obstime=time,location=loc,phase=phaseTrack)
    U = ac.SkyCoord(1*u.m,0*u.m,0*u.m,frame=uvw).transform_to(enu)
    V = ac.SkyCoord(0*u.m,1*u.m,0*u.m,frame=uvw).transform_to(enu)
    W = ac.SkyCoord(0*u.m,0*u.m,1*u.m,frame=uvw).transform_to(enu)
    assert np.cross(V.cartesian.xyz.value,W.cartesian.xyz.value).dot(z.cartesian.xyz.value) < 1e-10, "Not on the great circle"
    assert compVectors(W,east),"fail test d, w != east"
    #print("passed d")

def test_pointing():
    def compVectors(a,b):
        a = a.cartesian.xyz.value
        a /= np.linalg.norm(a)
        b = b.cartesian.xyz.value
        b /= np.linalg.norm(b)
        h = np.linalg.norm(a-b)
        return h < 1e-8
    #print("Test uv conventions when fix and obs time are equal")
    # with X - East, Z - NCP and Y - Down
    time = at.Time("2017-01-26T17:07:00.000",format='isot',scale='utc')
    loc = ac.EarthLocation(lon=10*u.deg,lat=10*u.deg,height=0*u.km)
    enu = ENU(location=loc,obstime=time)
    x = ac.SkyCoord(1,0,0,frame=enu)
    z = ac.SkyCoord(0,np.cos(loc.geodetic[1].rad),np.sin(loc.geodetic[1].rad),frame=enu)
    #ncp = ac.SkyCoord(0*u.one,0*u.one,1*u.one,frame='itrs').transform_to(enu)
    y = ac.SkyCoord(0,np.sin(loc.geodetic[1].rad),-np.cos(loc.geodetic[1].rad),frame=enu)
    lst = ac.AltAz(alt=90*u.deg,az=0*u.deg,location=loc,obstime=time).transform_to(ac.ICRS).ra
    #ha = lst - ra
    #print("a) when ha=0,dec=90  pointing aligns with xyz")
    ha = 0*u.deg
    ra = lst - ha
    dec = 90*u.deg
    phaseTrack = ac.SkyCoord(ra,dec,frame=ac.ICRS)
    pointing = Pointing(obstime=time,location=loc,phase=phaseTrack,fixtime=time)
    U = ac.SkyCoord(1,0,0,frame=pointing).transform_to(enu)
    V = ac.SkyCoord(0,1,0,frame=pointing).transform_to(enu)
    W = ac.SkyCoord(0,0,1,frame=pointing).transform_to(enu)
    assert compVectors(U,x),"fail test a, u != x"
    assert compVectors(V,y),"fail test a, v != y"
    assert compVectors(W,z),"fail test a, w != z"
    #print("passed a")
    #print("b) v, w, z are always on great circle")
    assert np.cross(V.cartesian.xyz.value,W.cartesian.xyz.value).dot(z.cartesian.xyz.value) < 1e-10, "Not on the great circle"
    #print("passed b")
    #print("c) when ha = 0 U points east")
    ha = 0*u.deg
    ra = lst - ha
    dec = 35*u.deg
    phaseTrack = ac.SkyCoord(ra,dec,frame=ac.ICRS)
    pointing = Pointing(obstime=time,location=loc,phase=phaseTrack,fixtime=time)
    U = ac.SkyCoord(1*u.m,0*u.m,0*u.m,frame=pointing).transform_to(enu)
    V = ac.SkyCoord(0*u.m,1*u.m,0*u.m,frame=pointing).transform_to(enu)
    W = ac.SkyCoord(0*u.m,0*u.m,1*u.m,frame=pointing).transform_to(enu)
    assert np.cross(V.cartesian.xyz.value,W.cartesian.xyz.value).dot(z.cartesian.xyz.value) < 1e-10, "Not on the great circle"
    east = ac.SkyCoord(1,0,0,frame=enu)
    assert compVectors(U,east),"fail test c, u != east"
    #print("passed c")
    #print("d) when dec=0 and ha = -6 w points east")
    ha = -6*u.hourangle
    ra = lst - ha
    dec = 0*u.deg
    phaseTrack = ac.SkyCoord(ra,dec,frame=ac.ICRS)
    pointing = Pointing(obstime=time,location=loc,phase=phaseTrack,fixtime=time)
    U = ac.SkyCoord(1*u.m,0*u.m,0*u.m,frame=pointing).transform_to(enu)
    V = ac.SkyCoord(0*u.m,1*u.m,0*u.m,frame=pointing).transform_to(enu)
    W = ac.SkyCoord(0*u.m,0*u.m,1*u.m,frame=pointing).transform_to(enu)
    assert np.cross(V.cartesian.xyz.value,W.cartesian.xyz.value).dot(z.cartesian.xyz.value) < 1e-10, "Not on the great circle"
    assert compVectors(W,east),"fail test d, w != east"
    #print("passed d")
    #print("More tests")
    fixtime = at.Time("2017-01-26T07:00:00.000",format='isot',scale='tai')
    time = at.Time("2017-01-26T13:00:00.000",format='isot',scale='tai')
    loc = ac.EarthLocation(lon=0*u.deg,lat=0*u.deg,height=0*u.km)
    lst = ac.AltAz(alt=90*u.deg,az=0*u.deg,location=loc,obstime=fixtime).transform_to(ac.ICRS).ra
    #ha = lst - ra
    ha = 0*u.deg
    ra = lst - ha
    dec = 0*u.deg
    phaseTrack = ac.SkyCoord(ra,dec,frame=ac.ICRS)
    #print(phaseTrack)
    pointing = Pointing(obstime=fixtime,location=loc,phase=phaseTrack,fixtime=fixtime)
    #print(phaseTrack.transform_to(pointing))
    #from ENUFrame import ENU
    enu = ENU(location=loc,obstime=time)
    east = ac.SkyCoord(1,0,0,frame=enu)
    print(east.cartesian.xyz)
    print(east.transform_to(pointing).cartesian.xyz)
    print(phaseTrack.transform_to(pointing).cartesian.xyz)
