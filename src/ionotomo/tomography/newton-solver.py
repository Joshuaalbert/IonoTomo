import tensorflow as tf
import numpy as np
from ionotomo.settings import TFSettings
from ionotomo import *
from ionotomo.ionosphere.iri import a_priori_model
from ionotomo.tomography.interpolation import RegularGridInterpolator
ufrom ionotomo.tomography.linear_operators import TECForwardEquation
import astropy.coordinates as ac
import astropy.units as au
import astropy.time as at

TECU = 1e13 #TEC unit / km

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




def newton_algorithm(d_obs, CdCt, args):
    factr = 1e7
    pgtol = 1e-2
    eps = np.finfo(float).eps
    max_iter = 20
    
    antennas,directions,times,spacing = args

    rays, grid, model_0 = calc_rays_and_initial_model(antennas,directions,times,zmax=1000.,res_n=201,spacing=10.)
    grid = np.array(grid).T
    Na,Nt,Nd,_,Ns = rays.shape
    nx,ny,nz = model_0.shape
    graph = tf.Graph()
    
    with graph.as_default():
        with tf.name_scope("newton_algorithm"):
            _model_0 = tf.placeholder(shape=model_0.shape,dtype=TFSettings.tf_float,name='model_0')
            _grid = [tf.placeholder(shape=n,dtype=TFSettings.tf_float,name='grid{}'.format(i)) for i,n in enumerate([nx,ny,nz])]
            _rays = tf.placeholder(shape=rays.shape,dtype=TFSettings.tf_float,name='rays')
            _d_obs = tf.placeholder(shape=d_obs.shape,dtype=TFSettings.tf_float,name='d_obs')
            _CdCt = tf.placeholder(shape=CdCt.shape,dtype=TFSettings.tf_float,name='CdCt')
            _model_prior = tf.placeholder(shape=model_0.shape,dtype=TFSettings.tf_float,name='model_prior')         
            model_prior = _model_prior

            model = tf.get_variable("model",dtype=TFSettings.tf_float,initializer=_model_0)
            g_op = TECForwardEquation(0,_grid,1e11*tf.exp(model),_rays)
            g = g_op.matmul(tf.ones_like(model))
            dd = tf.square(_d_obs - g)
            r = g_op.matmul(model_prior - model)
            s = dd - r
            S = CdCt
            snr = s/S
            dm = snr

            wdd = dd / CdCt
            chi2 = tf.reduce_sum(wdd)/2.
            optimizer = tf.train.AdamOptimizer(0.01)
            minimize_op = optimizer.apply_gradients([(dm,model)])
    sess = tf.InteractiveSession(graph=graph)
    sess.run(tf.global_variables_initializer(), feed_dict = {_model_0: np.log(model_0/1e11),
                                            _model_prior: np.log(model_0/1e11)})
    S_np1 = 1e50
    S_n = S_np1
    model_np1 = model_0
    iter = 0
    while ((S_n - S_np1)/max(np.abs(S_n),np.abs(S_np1),1.) > factr*eps and np.max(np.abs(dm)) > pgtol and iter < max_iter) or iter < 5:
        S_n = S_np1
        model_n = model_np1
        S_np1,model_np1 = sess.run([chi2,model],
                feed_dict = {
                    _grid[0]:grid[0], _grid[1]:grid[1],_grid[2]:grid[2],_rays: rays, _d_obs:d_obs,_CdCt:CdCt})
        dm = model_np1 - model_n
        iter += 1
        print("Iter {} : chi2 {}".format(iter,S_np1))
    sess.close()

if __name__ == '__main__':
    datapack = generate_example_datapack()
    antennas, antenna_labels = datapack.get_antennas(-1)
    directions,patch_names = datapack.get_directions(-1)
    times,timestamps = datapack.get_times(-1)
    args = (antennas,directions,times,10.)
    d_obs = np.random.normal(0,0.01,size=(len(antennas),len(times),len(directions)))
    CdCt = np.ones_like(d_obs)*0.001**2
    newton_algorithm(d_obs, CdCt, args)
