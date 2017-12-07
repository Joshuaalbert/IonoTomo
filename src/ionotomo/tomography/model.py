import tensorflow as tf
import numpy as np
from ionotomo.settings import TFSettings
from ionotomo import *
from ionotomo.ionosphere.iri import a_priori_model

import astropy.coordinates as ac
import astropy.units as au
import astropy.time as at

def calc_rays_and_initial_model(antennas,directions,times,zmax=1000.,res_n=201,spacing=10.):
    """Create straight line rays from given antennas, directions and times.
    antennas : astropy.coordinates.ITRS convertible
        The antenna locations
    """
    res_n = (res_n >> 1)*2 + 1
    fixtime = times[0]
    phase_center = ac.SkyCoord(np.mean(directions.ra.deg)*au.deg,
            np.mean(directions.dec.deg)*au.deg,frame='icrs')
    array_center = ac.SkyCoord(np.mean(antennas.x.to(au.m).value)*au.m,
                               np.mean(antennas.y.to(au.m).value)*au.m,
                               np.mean(antennas.z.to(au.m).value)*au.m,frame='itrs')
    rays = np.zeros((len(antennas),len(times),len(directions),3,res_n),dtype=float)
    factor = np.linspace(0,zmax,res_n)
    for j in range(len(times)):
        uvw = Pointing(location = array_center.earth_location,obstime = times[j],
                fixtime=fixtime, phase = phase_center)
        ants_uvw = antennas.transform_to(uvw).cartesian.xyz.to(au.km).value.T
        dirs_uvw = directions.transform_to(uvw).cartesian.xyz.value.T
        rays[:,j,:,0,:] = ants_uvw[:,0][:,None,None] + dirs_uvw[:,0][None,:,None]*factor[None,None,:]
        rays[:,j,:,1,:] = ants_uvw[:,1][:,None,None] + dirs_uvw[:,1][None,:,None]*factor[None,None,:]
        rays[:,j,:,2,:] = ants_uvw[:,2][:,None,None] + dirs_uvw[:,2][None,:,None]*factor[None,None,:]
    xmax = np.max(rays[:,:,:,0,:])
    ymax = np.max(rays[:,:,:,1,:])
    zmax = np.max(rays[:,:,:,2,:])
    xmin = np.min(rays[:,:,:,0,:])
    ymin = np.min(rays[:,:,:,1,:])
    zmin = np.min(rays[:,:,:,2,:])
    xvec = np.arange(xmin-spacing*3,xmax+spacing*3,spacing)
    yvec = np.arange(ymin-spacing*3,ymax+spacing*3,spacing)
    zvec = np.arange(zmin-spacing*3,zmax+spacing*3,spacing)
    
    uvw = Pointing(location = array_center.earth_location,obstime = times[0],
                fixtime=fixtime, phase = phase_center)
    X,Y,Z = np.meshgrid(xvec,yvec,zvec,indexing='ij')
    heights =  ac.SkyCoord(X.flatten()*au.km,Y.flatten()*au.km,Z.flatten()*au.km,
            frame=uvw).transform_to('itrs').earth_location.to_geodetic('WGS84')[2].to(au.km).value
    lat = array_center.earth_location.to_geodetic('WGS84').lat.to(au.deg).value
    lon = array_center.earth_location.to_geodetic('WGS84').lon.to(au.deg).value
    ne_model = a_priori_model(heights,zmax,lat,lon,fixtime).reshape(X.shape)
    return rays, (xvec,yvec,zvec), ne_model


