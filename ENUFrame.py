
# coding: utf-8

# In[14]:

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
from astropy.coordinates import ITRS



class ENU(BaseCoordinateFrame):
    """
    Written by Joshua G. Albert - albert@strw.leidenuniv.nl
    A coordinate or frame in the East-North-Up (ENU) system.  

    This frame has the following frame attributes, which are necessary for
    transforming from ENU to some other system:

    * ``obstime``
        The time at which the observation is taken.  Used for determining the
        position and orientation of the Earth.
    * ``location``
        The location on the Earth.  This can be specified either as an
        `~astropy.coordinates.EarthLocation` object or as anything that can be
        transformed to an `~astropy.coordinates.ITRS` frame.

    Parameters
    ----------
    representation : `BaseRepresentation` or None
        A representation object or None to have no data (or use the other keywords)
    east : :class:`~astropy.units.Quantity`, optional, must be keyword
        The east coordinate for this object (``north`` and ``up`` must also be given and
        ``representation`` must be None).
    north : :class:`~astropy.units.Quantity`, optional, must be keyword
        The east coordinate for this object (``east`` and ``up`` must also be given and
        ``representation`` must be None).
    up : :class:`~astropy.units.Quantity`, optional, must be keyword
        The east coordinate for this object (``north`` and ``east`` must also be given and
        ``representation`` must be None).

    Notes
    -----
    This is useful as an intermediate frame between ITRS and UVW for radio astronomy

    """

    frame_specific_representation_info = {
        'cartesian': [RepresentationMapping('x', 'east'),
                      RepresentationMapping('y', 'north'),
                     RepresentationMapping('z','up')],
    }
    
    default_representation = CartesianRepresentation

    obstime = TimeFrameAttribute(default=None)
    location = EarthLocationAttribute(default=None)
    #pressure = QuantityFrameAttribute(default=0, unit=u.hPa)
    #temperature = QuantityFrameAttribute(default=0, unit=u.deg_C)
    #relative_humidity = FrameAttribute(default=0)
    #obswl = QuantityFrameAttribute(default=1*u.micron, unit=u.micron)

    def __init__(self, *args, **kwargs):
        super(ENU, self).__init__(*args, **kwargs)

    @property
    def elevation(self):
        """
        Elevation above the horizon of the direction, in radians
        """
        return np.arctan2(self.up,np.sqrt(self.north**2 + self.east**2))




@frame_transform_graph.transform(FunctionTransform, ITRS, ENU)
def itrs_to_enu(itrs_coo, enu_frame):
    
    
    
    lon, lat, height = enu_frame.location.to_geodetic('WGS84')
    sinlat = np.sin(lat.to(u.radian).value)
    coslat = np.cos(lat.to(u.radian).value)
    sinlon = np.sin(lon.to(u.radian).value)
    coslon = np.cos(lon.to(u.radian).value)
    north = [-sinlat*coslon,
                      -sinlat*sinlon,
                      coslat]
    east = [-sinlon,coslon,0]
    up = [coslat*coslon,coslat*sinlon,sinlat]
    R = np.array([east,north,up])
    try:
        p = itrs_coo.cartesian.xyz.to(u.m).value
        p0 = np.array(enu_frame.location.to(u.m).value)
        diff = p-p0
        penu = R.dot(diff)
      
        rep = CartesianRepresentation(x = u.Quantity(penu[0],u.m,copy=False),
                                     y = u.Quantity(penu[1],u.m,copy=False),
                                     z = u.Quantity(penu[2],u.m,copy=False),
                                     copy=False)
    except:
        p = itrs_coo.cartesian.xyz.value
        diff = p
        penu = R.dot(diff)
      
        rep = CartesianRepresentation(x = u.Quantity(penu[0],None,copy=False),
                                     y = u.Quantity(penu[1],None,copy=False),
                                     z = u.Quantity(penu[2],None,copy=False),
                                     copy=False)
        

    return enu_frame.realize_frame(rep)


@frame_transform_graph.transform(FunctionTransform, ENU, ITRS)
def enu_to_itrs(enu_coo, itrs_frame):
    #p = itrs_frame.cartesian.xyz.to(u.m).value
    #p0 = np.array(enu_coo.location.to(u.m).value)
    #p = np.array(itrs_frame.location.to(u.m).value)
    
    
    lon, lat, height = enu_coo.location.to_geodetic('WGS84')
    sinlat = np.sin(lat.to(u.radian).value)
    coslat = np.cos(lat.to(u.radian).value)
    sinlon = np.sin(lon.to(u.radian).value)
    coslon = np.cos(lon.to(u.radian).value)
    north = [-sinlat*coslon,
                      -sinlat*sinlon,
                      coslat]
    east = [-sinlon,coslon,0]
    up = [coslat*coslon,coslat*sinlon,sinlat]
    R = np.array([east,north,up])
    
    try:
        diff = R.T.dot(enu_coo.cartesian.xyz.to(u.m).value)
        p0 = np.array(enu_coo.location.to(u.m).value)
        p = diff + p0
        rep = CartesianRepresentation(x = u.Quantity(p[0],u.m,copy=False),
                                     y = u.Quantity(p[1],u.m,copy=False),
                                     z = u.Quantity(p[2],u.m,copy=False),
                                     copy=False)
    except:
        diff = R.T.dot(enu_coo.cartesian.xyz.value)
        p = diff
        rep = CartesianRepresentation(x = u.Quantity(p[0],None,copy=False),
                                     y = u.Quantity(p[1],None,copy=False),
                                     z = u.Quantity(p[2],None,copy=False),
                                     copy=False)

    return itrs_frame.realize_frame(rep)
    
    #return ITRS(*p*u.m,obstime=enu_coo.obstime).transform_to(itrs_frame)
    
@frame_transform_graph.transform(FunctionTransform, ENU, ENU)
def enu_to_enu(from_coo, to_frame):
    # for now we just implement this through ITRS to make sure we get everything
    # covered
    return from_coo.transform_to(ITRS(obstime=from_coo.obstime,location=from_coo.location)).transform_to(to_frame)

if __name__ == '__main__':
    import astropy.coordinates as ac
    import astropy.time as at
    loc1 = ac.SkyCoord(x=np.array([1.1,1])*u.m,y=[2,1]*u.m,z=[1,1]*u.m,obstime=at.Time(0,format='gps'),frame='itrs')
    loc = ac.EarthLocation(x=1*u.m,y=0*u.m,z=0*u.m)
    time = at.Time(1,format='gps')
    h = loc1.transform_to('itrs')
    enu = ENU(obstime=at.Time(0,format='gps'),location=loc)
    print(enu.location.geocentric)
    loc3 = ac.SkyCoord(np.array([1.1,1])*u.m,[2,1]*u.m,[1,1]*u.m,frame=enu)
    print(loc3.transform_to('itrs'))
    locenu = loc1.transform_to(enu)
    print("locenu:",locenu)
    print(locenu.elevation)
    print(loc1.transform_to(enu).transform_to('itrs').spherical)
    print(loc1.transform_to(enu).transform_to('itrs').transform_to(enu))
    aa = ac.AltAz(obstime=time,location=loc)
    s = ac.SkyCoord(ra=45*u.deg,dec=45*u.deg)
    print("unit:",s.transform_to('itrs').spherical.distance)
    print(s.transform_to(aa))
    print(s.transform_to(aa).transform_to(enu))
    print(s.transform_to(aa).transform_to(enu).transform_to(aa))


# In[ ]:



