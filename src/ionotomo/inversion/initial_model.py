
# coding: utf-8

# In[1]:

import astropy.units as au
import astropy.time as at
import astropy.coordinates as ac
import numpy as np
from scipy.special import gamma
from uvw_frame import UVW
from IRI import a_priori_model
from tri_cubic import TriCubic
from Covariance import Covariance

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
    print("Found domain u in {}..{}, v in {}..{}, w in {}..{}".format(umin,umax,vmin,vmax,wmin,wmax))
    return uvec,vvec,wvec

def turbulent_perturbation(TCI,sigma = 3.,corr = 20., nu = 5./2.):    
    cov_obj = Covariance(TCI,sigma,corr,nu)
    B = cov_obj.realization()
    return B
    

def create_initial_model(datapack,ant_idx = -1, time_idx = -1, dir_idx = -1, zmax = 1000.,spacing=5.,padding=20):
    antennas,antenna_labels = datapack.get_antennas(ant_idx = ant_idx)
    patches, patch_names = datapack.get_directions(dir_idx=dir_idx)
    times,timestamps = datapack.get_times(time_idx=time_idx)
    Na = len(antennas)
    Nt = len(times)
    Nd = len(patches)  
    #Setting up ionosphere to use
    print("Using radio array {}".format(datapack.radio_array))
    phase = datapack.get_center_direction()
    print("Using phase center {} {}".format(phase.ra,phase.dec))
    fixtime = times[Nt>>1]
    print("Fixing frame at {}".format(fixtime.isot))
    uvw = UVW(location = datapack.radio_array.get_center().earth_location,obstime = fixtime,phase = phase)
    print("Elevation is {}".format(uvw.elevation))
    zenith = datapack.radio_array.get_sun_zenith_angle(fixtime)
    print("Sun at zenith angle {}".format(zenith))
    print("Creating ionosphere model...")
    xvec,yvec,zvec = determine_inversion_domain(spacing,antennas, patches,uvw, zmax, padding = padding)
    X,Y,Z = np.meshgrid(xvec,yvec,zvec,indexing='ij')
    print("Nx={} Ny={} Nz={} number of cells: {}".format(len(xvec),len(yvec),len(zvec),np.size(X)))
    coords = ac.SkyCoord(X.flatten()*au.km,Y.flatten()*au.km,Z.flatten()*au.km,frame=uvw).transform_to('itrs').earth_location.to_geodetic('WGS84')
    heights = coords[2].to(au.km).value#height in geodetic
    neModel = a_priori_model(heights,zenith).reshape(X.shape)
    neModel[neModel<4e7] = 4e7
    return TriCubic(xvec,yvec,zvec,neModel)

def createTurbulentlModel(datapack,ant_idx = -1, time_idx = -1, dir_idx = -1, zmax = 1000., spacing=5.):
    ne_tci = create_initial_model(datapack,ant_idx = ant_idx, time_idx = time_idx, dir_idx = dir_idx, zmax = zmax, spacing= spacing)
    dM = turbulent_perturbation(ne_tci,sigma=np.log(5.),corr=25.,nu=7./2.)
    pertTCI = TriCubic(ne_tci.xvec,ne_tci.yvec,ne_tci.zvec,ne_tci.get_shaped_array()*np.exp(dM))
    return pertTCI
    
def test_create_initial_model():
    from real_data import DataPack
    datapack = DataPack(filename="output/test/datapack_obs.hdf5")
    ne_tci = create_initial_model(datapack,ant_idx = -1, time_idx = -1, dir_idx = -1, zmax = 1000.)
    ne_tci.save("output/test/neModel.hdf5")
    
def test_create_turbulent_model():
    from real_data import DataPack
    from PlotTools import animate_tci_slices
    import os
    datapack = DataPack(filename="output/test/datapack_obs.hdf5")
    for i in range(1):
        ne_tci = createTurbulentlModel(datapack,ant_idx = -1, time_idx = [0], dir_idx = -1, zmax = 1000.)
        try:
            os.makedirs("output/test/InitialModel/turbulent-{}/fig".format(i))
        except:
            pass
        #ne_tci.save("output/test/InitialModel/turbulent-{}/neModelTurbulent.hdf5".format(i))
        #animate_tci_slices(ne_tci,"output/test/InitialModel/turbulent-{}/fig".format(i))
    
if __name__ == '__main__':
    #test_create_initial_model()
    test_create_turbulent_model()


# In[ ]:



