import astropy.units as au
import astropy.coordinates as ac
import numpy as np
from ionotomo.astro.frames.uvw_frame import UVW
from ionotomo.ionosphere.iri import a_priori_model
from ionotomo.geometry.tri_cubic import TriCubic
from ionotomo.ionosphere.covariance import Covariance

import logging as log

def determine_inversion_domain(spacing,antennas, directions, pointing, zmax, padding = 20):
    '''Determine the domain of the inversion'''
    ants = antennas.transform_to(pointing).cartesian.xyz.to(au.km).value.transpose()
    dirs = directions.transform_to(pointing).cartesian.xyz.value.transpose()
    #old
    uend = np.add.outer(ants[:,0],dirs[:,0]*zmax/dirs[:,2])
    vend = np.add.outer(ants[:,1],dirs[:,1]*zmax/dirs[:,2])
    wend = np.add.outer(ants[:,2],dirs[:,2]*zmax/dirs[:,2])
    
    
    umin = min(np.min(ants[:,0]),np.min(uend.flatten()))-spacing*padding
    umax = max(np.max(ants[:,0]),np.max(uend.flatten()))+spacing*padding
    vmin = min(np.min(ants[:,1]),np.min(vend.flatten()))-spacing*padding
    vmax = max(np.max(ants[:,1]),np.max(vend.flatten()))+spacing*padding
    wmin = min(np.min(ants[:,2]),np.min(wend.flatten()))-spacing*padding
    wmax = max(np.max(ants[:,2]),np.max(wend.flatten()))+spacing*padding
    Nu = np.ceil((umax-umin)/spacing)
    Nv = np.ceil((vmax-vmin)/spacing)
    Nw = np.ceil((wmax-wmin)/spacing)
    uvec = np.linspace(umin,umax,int(Nu))
    vvec = np.linspace(vmin,vmax,int(Nv))
    wvec = np.linspace(wmin,wmax,int(Nw))
    log.info("Found domain u in {}..{}, v in {}..{}, w in {}..{}".format(umin,umax,vmin,vmax,wmin,wmax))
    return uvec,vvec,wvec

def turbulent_perturbation(tci,sigma = 3.,corr = 20., nu = 5./2.):    
    cov_obj = Covariance(tci,sigma,corr,nu)
    B = cov_obj.realization()
    return B
    

def create_initial_model(datapack,ant_idx = -1, time_idx = -1, dir_idx = -1, zmax = 1000.,spacing=5.,padding=20,thin_f = False):
    antennas,antenna_labels = datapack.get_antennas(ant_idx = ant_idx)
    patches, patch_names = datapack.get_directions(dir_idx=dir_idx)
    times,timestamps = datapack.get_times(time_idx=time_idx)
    Na = len(antennas)
    Nt = len(times)
    Nd = len(patches)  
    #Setting up ionosphere to use
    log.info("Using radio array {}".format(datapack.radio_array))
    phase = datapack.get_center_direction()
    log.info("Using phase center {} {}".format(phase.ra,phase.dec))
    fixtime = times[Nt>>1]
    log.info("Fixing frame at {}".format(fixtime.isot))
    uvw = UVW(location = datapack.radio_array.get_center().earth_location,obstime = fixtime,phase = phase)
    log.info("Elevation is {}".format(uvw.elevation))
    zenith = datapack.radio_array.get_sun_zenith_angle(fixtime)
    log.info("Sun at zenith angle {}".format(zenith))
    log.info("Creating ionosphere model...")
    xvec,yvec,zvec = determine_inversion_domain(spacing,antennas, patches,uvw, zmax, padding = padding)
    X,Y,Z = np.meshgrid(xvec,yvec,zvec,indexing='ij')
    log.info("Nx={} Ny={} Nz={} number of cells: {}".format(len(xvec),len(yvec),len(zvec),np.size(X)))
    coords = ac.SkyCoord(X.flatten()*au.km,Y.flatten()*au.km,Z.flatten()*au.km,frame=uvw).transform_to('itrs').earth_location.to_geodetic('WGS84')
    heights = coords[2].to(au.km).value#height in geodetic
    ne_model = a_priori_model(heights,zenith,thin_f=thin_f).reshape(X.shape)
    ne_model[ne_model<4e7] = 4e7
    return TriCubic(xvec,yvec,zvec,ne_model)

def create_turbulent_model(datapack,corr=20.,seed=None, **initial_model_kwargs):
    if seed is not None:
        np.random.seed(seed)
    #ne_tci = create_initial_model(datapack,ant_idx = ant_idx, time_idx = time_idx, dir_idx = dir_idx, zmax = zmax, spacing= spacing,padding=padding)
    ne_tci = create_initial_model(datapack,**initial_model_kwargs)
    dn_max = np.sqrt(1 - 8.98**2 * 1e10/datapack.radio_array.frequency**2) - np.sqrt(1 - 8.98**2 * 5e10/datapack.radio_array.frequency**2)
    log.info("Max dn {}".format(dn_max))
    n_max = np.sqrt(1 - 8.98**2 * 4e7/datapack.radio_array.frequency**2)
    dn = turbulent_perturbation(ne_tci,sigma=dn_max/2.,corr=corr,nu=2./3.)
    dn += dn_max/2.
    #dn *= ne_tci.M/np.max(ne_tci.M)
    #ne = ne_tci.M*np.exp(dm)
    n = np.sqrt(1 - 8.98**2 * ne_tci.M/datapack.radio_array.frequency**2)   
    n -= dn
    n[n>n_max] = n_max
    
#    n /= np.max(n)
#    n *= 0.99999
    #n -= np.min(n)
    #n /= np.max(n)
    #n *= 0.007
    #n += (1-0.007)
    ne = (n**2 - 1.)/-8.98**2 * datapack.radio_array.frequency**2
    pert_tci = TriCubic(ne_tci.xvec,ne_tci.yvec,ne_tci.zvec,ne)
    return pert_tci


