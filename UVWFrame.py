
# coding: utf-8

# In[ ]:

from __future__ import (absolute_import, unicode_literals, division,
                        print_function)
import numpy as np

import astropy.units as u

from astropy.coordinates.baseframe import (BaseCoordinateFrame, FrameAttribute,
                         TimeFrameAttribute, QuantityFrameAttribute,
                         RepresentationMapping, EarthLocationAttribute,
                              frame_transform_graph)

from astropy.coordinates.transformations import FunctionTransform
from astropy.coordinates.representation import (SphericalRepresentation,
                              UnitSphericalRepresentation,CartesianRepresentation)
from astropy.coordinates import ITRS,ICRS,AltAz

class CoordinateAttribute(FrameAttribute):
    """
    A frame attribute which is a coordinate object. It can be given as a
    low-level frame class *or* a `~astropy.coordinates.SkyCoord`, but will
    always be converted to the low-level frame class when accessed.
    Parameters
    ----------
    frame : a coordinate frame class
        The type of frame this attribute can be
    default : object
        Default value for the attribute if not provided
    secondary_attribute : str
        Name of a secondary instance attribute which supplies the value if
        ``default is None`` and no value was supplied during initialization.
    """
    def __init__(self, frame, default=None, secondary_attribute=''):
        self._frame = frame
        super(CoordinateAttribute, self).__init__(default, secondary_attribute)

    def convert_input(self, value):
        """
        Checks that the input is a SkyCoord with the necessary units (or the
        special value ``None``).
        Parameters
        ----------
        value : object
            Input value to be converted.
        Returns
        -------
        out, converted : correctly-typed object, boolean
            Tuple consisting of the correctly-typed object and a boolean which
            indicates if conversion was actually performed.
        Raises
        ------
        ValueError
            If the input is not valid for this attribute.
        """
        if value is None:
            return None, False
        elif isinstance(value, self._frame):
            return value, False
        else:
            if not hasattr(value, 'transform_to'):
                raise ValueError('"{0}" was passed into a '
                                 'CoordinateAttribute, but it does not have '
                                 '"transform_to" method'.format(value))
            transformedobj = value.transform_to(self._frame)
            if hasattr(transformedobj, 'frame'):
                transformedobj = transformedobj.frame
            return transformedobj, True

class UVW(BaseCoordinateFrame):
    """
    Written by Joshua G. Albert - albert@strw.leidenuniv.nl
    A coordinate or frame in the UVW system.  

    This frame has the following frame attributes, which are necessary for
    transforming from UVW to some other system:

    * ``obstime``
        The time at which the observation is taken.  Used for determining the
        position and orientation of the Earth.
    * ``location``
        The location on the Earth.  This can be specified either as an
        `~astropy.coordinates.EarthLocation` object or as anything that can be
        transformed to an `~astropy.coordinates.ITRS` frame.
    * ``phaseDir``
        The phase tracking center of the frame.  This can be specified either as an
        (ra,dec) `~astropy.units.Qunatity` or as anything that can be
        transformed to an `~astropy.coordinates.ICRS` frame.

    Parameters
    ----------
    representation : `BaseRepresentation` or None
        A representation object or None to have no data (or use the other keywords)
    u : :class:`~astropy.units.Quantity`, optional, must be keyword
        The u coordinate for this object (``v`` and ``w`` must also be given and
        ``representation`` must be None).
    v : :class:`~astropy.units.Quantity`, optional, must be keyword
        The v coordinate for this object (``u`` and ``w`` must also be given and
        ``representation`` must be None).
    w : :class:`~astropy.units.Quantity`, optional, must be keyword
        The w coordinate for this object (``u`` and ``v`` must also be given and
        ``representation`` must be None).

    Notes
    -----
    This is useful for radio astronomy.

    """

    frame_specific_representation_info = {
        'cartesian': [RepresentationMapping('x', 'u'),
                      RepresentationMapping('y', 'v'),
                     RepresentationMapping('z','w')],
    }
    
    default_representation = CartesianRepresentation

    obstime = TimeFrameAttribute(default=None)
    location = EarthLocationAttribute(default=None)
    phase = CoordinateAttribute(ICRS,default=None)

    def __init__(self, *args, **kwargs):
        super(UVW, self).__init__(*args, **kwargs)


@frame_transform_graph.transform(FunctionTransform, ITRS, UVW)
def itrs_to_uvw(itrs_coo, uvw_frame):
    '''Defines the transformation between ITRS and the UVW frame.'''
    
    #if np.any(itrs_coo.obstime != uvw_frame.obstime):
    #    itrs_coo = itrs_coo.transform_to(ITRS(obstime=uvw_frame.obstime))
        
    # if the data are UnitSphericalRepresentation, we can skip the distance calculations
    is_unitspherical = (isinstance(itrs_coo.data, UnitSphericalRepresentation) or
                        itrs_coo.cartesian.x.unit == u.one)
    
    lon, lat, height = uvw_frame.location.to_geodetic('WGS84')
    lst = AltAz(alt=90*u.deg,az=0*u.deg,location=uvw_frame.location,obstime=uvw_frame.obstime).transform_to(ICRS).ra
    ha = (lst - uvw_frame.phase.ra).to(u.radian).value
    dec = uvw_frame.phase.dec.to(u.radian).value
    lonrad = lon.to(u.radian).value - ha
    latrad = dec #lat.to(u.radian).value + 
    sinlat = np.sin(latrad)
    coslat = np.cos(latrad)
    sinlon = np.sin(lonrad)
    coslon = np.cos(lonrad)
    north = [-sinlat*coslon,
                      -sinlat*sinlon,
                      coslat]
    east = [-sinlon,coslon,0]
    up = [coslat*coslon,coslat*sinlon,sinlat]
    R = np.array([east,north,up])
    
    if is_unitspherical:
        #don't need to do distance calculation
        p = itrs_coo.cartesian.xyz.value
        diff = p
        penu = R.dot(diff)
    
        rep = CartesianRepresentation(x = u.Quantity(penu[0],u.one,copy=False),
                                     y = u.Quantity(penu[1],u.one,copy=False),
                                     z = u.Quantity(penu[2],u.one,copy=False),
                                     copy=False)
    else:
        p = itrs_coo.cartesian.xyz
        p0 = ITRS(*uvw_frame.location.geocentric,obstime=uvw_frame.obstime).cartesian.xyz
        diff = (p.T-p0).T
        penu = R.dot(diff)
      
        rep = CartesianRepresentation(x = penu[0],#u.Quantity(penu[0],u.m,copy=False),
                                     y = penu[1],#u.Quantity(penu[1],u.m,copy=False),
                                     z = penu[2],#u.Quantity(penu[2],u.m,copy=False),
                                     copy=False)

    return uvw_frame.realize_frame(rep)


@frame_transform_graph.transform(FunctionTransform, UVW, ITRS)
def uvw_to_itrs(uvw_coo, itrs_frame):
    #p = itrs_frame.cartesian.xyz.to(u.m).value
    #p0 = np.array(enu_coo.location.to(u.m).value)
    #p = np.array(itrs_frame.location.to(u.m).value)
    
    
    lon, lat, height = uvw_coo.location.to_geodetic('WGS84')
    lst = AltAz(alt=90*u.deg,az=0*u.deg,location=uvw_coo.location,obstime=uvw_coo.obstime).transform_to(ICRS).ra
    ha = (lst - uvw_coo.phase.ra).to(u.radian).value
    dec = uvw_coo.phase.dec.to(u.radian).value
    lonrad = lon.to(u.radian).value - ha
    latrad = dec #lat.to(u.radian).value + 
    sinlat = np.sin(latrad)
    coslat = np.cos(latrad)
    sinlon = np.sin(lonrad)
    coslon = np.cos(lonrad)
    
    north = [-sinlat*coslon,
                      -sinlat*sinlon,
                      coslat]
    east = [-sinlon,coslon,0]
    up = [coslat*coslon,coslat*sinlon,sinlat]
    R = np.array([east,north,up])
    
    if isinstance(uvw_coo.data, UnitSphericalRepresentation) or uvw_coo.cartesian.x.unit == u.one:
        diff = R.T.dot(uvw_coo.cartesian.xyz)
        p = diff
        rep = CartesianRepresentation(x = u.Quantity(p[0],u.one,copy=False),
                                     y = u.Quantity(p[1],u.one,copy=False),
                                     z = u.Quantity(p[2],u.one,copy=False),
                                     copy=False)
    else:
        diff = R.T.dot(uvw_coo.cartesian.xyz)
        p0 = ITRS(*uvw_coo.location.geocentric,obstime=uvw_coo.obstime).cartesian.xyz
        #print (R,diff)
        p = (diff.T + p0).T
        #print (p)
        rep = CartesianRepresentation(x = p[0],#u.Quantity(p[0],u.m,copy=False),
                                     y = p[1],#u.Quantity(p[1],u.m,copy=False),
                                     z = p[2],#u.Quantity(p[2],u.m,copy=False),
                                     copy=False)

    return itrs_frame.realize_frame(rep)
    
    #return ITRS(*p*u.m,obstime=enu_coo.obstime).transform_to(itrs_frame)
    
@frame_transform_graph.transform(FunctionTransform, UVW, UVW)
def uvw_to_uvw(from_coo, to_frame):
    # for now we just implement this through ITRS to make sure we get everything
    # covered
    return from_coo.transform_to(ITRS(obstime=from_coo.obstime,location=from_coo.location)).transform_to(to_frame)

if __name__ == '__main__':
    import astropy.coordinates as ac
    import astropy.time as at
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
    from ENUFrame import ENU
    enu = ENU(location=loc,obstime=time)
    x = ac.SkyCoord(1,0,0,frame=enu)
    z = ac.SkyCoord(0,np.cos(loc.geodetic[1].rad),np.sin(loc.geodetic[1].rad),frame=enu)
    #ncp = ac.SkyCoord(0*u.one,0*u.one,1*u.one,frame='itrs').transform_to(enu)
    y = ac.SkyCoord(0,np.sin(loc.geodetic[1].rad),-np.cos(loc.geodetic[1].rad),frame=enu)
    lst = ac.AltAz(alt=90*u.deg,az=0*u.deg,location=loc,obstime=time).transform_to(ICRS).ra
    #ha = lst - ra
    print("a) when ha=0,dec=90  uvw aligns with xyz")
    ha = 0*u.deg
    ra = lst - ha
    dec = 90*u.deg
    phaseTrack = ac.SkyCoord(ra,dec,frame='icrs')
    uvw = UVW(obstime=time,location=loc,phase=phaseTrack)
    U = ac.SkyCoord(1,0,0,frame=uvw).transform_to(enu)
    V = ac.SkyCoord(0,1,0,frame=uvw).transform_to(enu)
    W = ac.SkyCoord(0,0,1,frame=uvw).transform_to(enu)
    assert compVectors(U,x),"fail test a, u != x"
    assert compVectors(V,y),"fail test a, v != y"
    assert compVectors(W,z),"fail test a, w != z"
    print("passed a")
    print("b) v, w, z are always on great circle")
    assert np.cross(V.cartesian.xyz.value,W.cartesian.xyz.value).dot(z.cartesian.xyz.value) < 1e-10, "Not on the great circle"
    print("passed b")
    print("c) when ha = 0 U points east")
    ha = 0*u.deg
    ra = lst - ha
    dec = 35*u.deg
    phaseTrack = ac.SkyCoord(ra,dec,frame='icrs')
    uvw = UVW(obstime=time,location=loc,phase=phaseTrack)
    U = ac.SkyCoord(1*u.m,0*u.m,0*u.m,frame=uvw).transform_to(enu)
    V = ac.SkyCoord(0*u.m,1*u.m,0*u.m,frame=uvw).transform_to(enu)
    W = ac.SkyCoord(0*u.m,0*u.m,1*u.m,frame=uvw).transform_to(enu)
    assert np.cross(V.cartesian.xyz.value,W.cartesian.xyz.value).dot(z.cartesian.xyz.value) < 1e-10, "Not on the great circle"
    east = ac.SkyCoord(1,0,0,frame=enu)
    assert compVectors(U,east),"fail test c, u != east"
    print("passed c")
    print("d) when dec=0 and ha = -6 w points east")
    ha = -6*u.hourangle
    ra = lst - ha
    dec = 0*u.deg
    phaseTrack = ac.SkyCoord(ra,dec,frame='icrs')
    uvw = UVW(obstime=time,location=loc,phase=phaseTrack)
    U = ac.SkyCoord(1*u.m,0*u.m,0*u.m,frame=uvw).transform_to(enu)
    V = ac.SkyCoord(0*u.m,1*u.m,0*u.m,frame=uvw).transform_to(enu)
    W = ac.SkyCoord(0*u.m,0*u.m,1*u.m,frame=uvw).transform_to(enu)
    assert np.cross(V.cartesian.xyz.value,W.cartesian.xyz.value).dot(z.cartesian.xyz.value) < 1e-10, "Not on the great circle"
    assert compVectors(W,east),"fail test d, w != east"
    print("passed d")

