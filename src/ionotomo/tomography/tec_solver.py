import tensorflow as tf
import numpy as np
import pylab as plt
from ionotomo import *
import time as tm
import h5py
from ionotomo.settings import TFSettings

import gpflow as gp

from gpflow.transforms import positive
from gpflow import priors
from gpflow.decors import params_as_tensors
from gpflow.params import Parameter
from gpflow.likelihoods import Likelihood

from ionotomo.tomography.simulate import simulate_tec
import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au

class VisibilityLikelihood(Likelihood):
    def __init__(self, antenna1, antenna2,Na,Nd,Nt,Nf, freqs, Vmod, var=1.,dropout=0.2,name=None):
        super(VisibilityLikelihood, self).__init__(name)
        assert len(antenna1) == len(antenna2)
        self.antenna1 = antenna1
        self.antenna2 = antenna2
        self.freqs = freqs
        self.Vmod = Vmod
        self.Na = Na
        self.Nt = Nt
        self.Nd = Nd
        self.Nf = Nf
        self.c = Parameter(np.zeros([Na,Nt,1,1]),prior=priors.Gaussian(0.,1.))
        self.variance_V = var
        self.var_scale = Parameter(1.,transform=positive) 
        self.dropout = dropout

    @params_as_tensors
    def logp(self, F, Y):
        """The Loss || V^_ijk - gik gjk* V^mod_ijk ||^2 / sigma_V^2 / 2
        F is"""

        #F is tec shape (Na,Nt,Nd)
        F = tf.reshape(F,(self.Na,self.Nt,self.Nd,1))

        #Na,Nt,Nd,Nf
        phi = self.c - 8.4480e9/self.freqs[None,None,None,:] * F

        #Nb,Nt,Nd,Nf
        phi_ij = tf.gather(phi,self.antenna2,axis=0) - tf.gather(phi,self.antenna1,axis=0)
        G = tf.exp(1j*tf.cast(phi_ij,tf.complex128))


        dY = tf.reduce_sum(self.Vmod * G, axis=2) - tf.reshape(Y,(self.Na,self.Nt,self.Nf))
        dY = tf.nn.dropout(dY,self.dropout)

        residuals_real = tf.sqrt(tf.square(tf.real(dY))/(self.var_scale * self.variance_V) + 1.) - 1.
        residuals_imag = tf.sqrt(tf.square(tf.imag(dY))/(self.var_scale * self.variance_V) + 1.) - 1.
        return tf.reduce_mean(residuals_real + residuals_imag)

    def _check_targets(self, Y_np):  # pylint: disable=R0201
        if np.array(list(Y_np)).dtype != settings.np_float:
            raise ValueError('use {}, even for discrete variables'.format(settings.np_float))
        
    

def test_solve_gp():
    """Solve for tec and constant assuming.
    P(c,tec|V) \propto L(V|c,tec) P(c) GP(tec)
    """
    Na = 4
    Nt = 1
    Nd = 5
    Nf = 2

    antennas = np.arange(Na)
    directions = np.random.normal(size=[Nd,2])
    freqs = np.arange(Nf)+1
    antenna1 = []
    antenna2 = []
    X = []
    V = []
    V_mod = []
    s = np.random.uniform(size=Nd)

    g = np.exp(1j*np.random.normal(size=[Na,Nd]))

    
    for i1 in range(Na):
        for i2 in range(i1,Na):
            vmod = np.zeros(Nd,dtype=type(1j))
            for k in range(Nd):
                vmod[k] = s[k]*np.exp(-1j*(i2-i1))
            antenna1.append(i1)
            antenna2.append(i2)
            V_mod.append(vmod)
            V.append(np.sum(vmod * g[i1,:] * g[i2,:].conjugate()))

    V = np.tile(np.array(V)[:,None,None],(1,Nt,Nf)).flatten()
    V_mod = np.tile(np.array(V_mod)[:,None,:,None],(1,Nt,1,Nf))

    for k in range(Nd):
        for i in range(Na):
            X.append(np.concatenate([directions[k,:],[i]]))
    X = np.array(X)


    coreg = gp.kernels.Coregion(1, output_dim=2, rank=1, active_dims=[2])
    coreg.W.set_trainable=False
    k1 = gp.kernels.RBF(2)
    K = k1*coreg
    likelihood = VisibilityLikelihood(antenna1, antenna2,Na,Nd,Nt,Nf, freqs, V_mod, var=1.,dropout=0.2)
    m =  gp.models.VGP(X, V, kern=K, likelihood=likelihood, num_latent=1)

def mask_timestamp(time,time_idx):
    times = np.unique(time)
    mask = np.isin(time,times[time_idx])
    return mask

def interp_nearest(x,y,z,x_,y_):
    """given input arrays (x,y,z) return nearest neighbour to (x_, y_)
    Return same shape as x_"""
    dx = np.subtract.outer(x_,x)
    dy = np.subtract.outer(y_,y)
    r = dx**2
    dy *= dy
    r += dy
    np.sqrt(r,out=r)
    arg = np.argmin(r,axis=1)
    z_ = z[arg]
    return z_

def make_xstar(X,N=50):
    """Make a screen NxN based on obs coords X.
    Returns shape (N*N, 2) array"""
    xmin = np.min(X[:,0])
    ymin = np.min(X[:,1])
    xmax = np.max(X[:,0])
    ymax = np.max(X[:,1])
    xvec = np.linspace(xmin,xmax,N)
    yvec = np.linspace(ymin,ymax,N)
    x,y = np.meshgrid(xvec,yvec,indexing='ij')
    Xstar = np.array([x.flatten(),y.flatten()]).T
    return Xstar

def angular_space(dirs_uvw):
    """Go to angular space, small angles mean not much difference"""
    X = np.array([np.arctan2(dirs_uvw.u.value,dirs_uvw.w.value),
                  np.arctan2(dirs_uvw.v.value,dirs_uvw.w.value)]).T
    return X



def tec_solve(V,V_mod,freqs,ant1, ant2, time, antenna_labels, ra, dec, weight_spectrum, flags,timestep=[0], freq_step = 10, tec_prior=None, c_prior=None):
    if not isinstance(timestep,(tuple,list)):
        timestep = [timestep]
    time_mask = mask_timestamp(time[...],timestep)
    freq_mask = range(0,len(freqs),freq_step)
    V = V[time_mask,...][:,freq_mask,:]
    V_mod = V_mod[time_mask,...][:,freq_mask,...]
    ant1 = ant1[time_mask,...]
    ant2 = ant2[time_mask,...]
    time = time[time_mask,...]
    weight_spectrum = weight_spectrum[time_mask,...][:,freq_mask,:]
    flags = flags[time_mask,...][:,freq_mask,:]
    freqs = freqs[freq_mask]
    antenna_labels = antenna_labels[...].astype('U')

    Na = len(antenna_labels)
    Nd = len(ra)
    times = np.unique(time)
    Nt = len(times)
    time_idx = np.zeros(len(time),dtype=int)
    for j in range(Nt):
        time_idx[time == times[j]] = j
    Nf = len(freqs)

    radio_array = RadioArray(array_file=RadioArray.lofar_array)
    antennas = radio_array.get_antenna_locs()
    lofar_labels = radio_array.get_antenna_labels()
    lab_map = []
    for lab in antenna_labels:
        for i in range(len(lofar_labels)):
            if lab == lofar_labels[i]:
                lab_map.append(i)
                break
    #in correct order
    antennas = antennas[lab_map]
    directions = ac.SkyCoord(ra*au.deg, dec*au.deg,frame='icrs')
    times = at.Time(times[timestep[0]:timestep[-1]+1]/86400.,format='mjd',scale='tai')
    #print(antennas.shape,directions.shape,times.shape)
    if tec_prior is None:
        tec_prior = simulate_tec(antennas,directions,times)
    if c_prior is None:
        c_prior = np.zeros([Na,Nt],dtype=float)
    #print(tec_prior)
    #print(tec_prior.shape)


    graph = tf.Graph()
    
    with graph.as_default():
        with tf.name_scope("tec_solver"):
            # num_baselines*Nt, Nf, 4
            _V_init = tf.placeholder(shape=V.shape,dtype=TFSettings.tf_complex,name='V')
            _V = tf.get_variable("V",initializer=_V_init,trainable=False)
            # num_baselines*Nt, Nf, 4
            _weight_spectrum_init = tf.placeholder(shape=weight_spectrum.shape,dtype=TFSettings.tf_float,name='weight_spectrum')
            _weight_spectrum= tf.get_variable("weight_spectrum",initializer=_weight_spectrum_init,trainable=False)
            # num_baselines*Nt, Nf, 4
            _flags_init = tf.placeholder(shape=flags.shape,dtype=TFSettings.tf_float,name='flags')
            _flags = tf.get_variable("flags",initializer=_flags_init,trainable=False)
            # num_baselines*Nt, Nf, 4, Nd
            _V_mod_init = tf.placeholder(shape=V_mod.shape,dtype=TFSettings.tf_complex,name='V_mod')
            _V_mod = tf.get_variable("V_mod",initializer=_V_mod_init,trainable=False)
            _tec_init = tf.placeholder(shape = [Na,Nt,Nd], dtype=TFSettings.tf_float,name='tec_init')
            _tec = tf.get_variable("tec",initializer=_tec_init)
            _c_init = tf.placeholder(shape = [Na, Nt], dtype=TFSettings.tf_float,name='c_init')
            _c = tf.get_variable("c",initializer=_c_init)
            _freqs_init = tf.placeholder(shape = freqs.shape, dtype=TFSettings.tf_float,name='freqs')
            _freqs = tf.get_variable("freqs",initializer=_freqs_init,trainable=False)
            _ant1_init = tf.placeholder(shape = ant2.shape, dtype=TFSettings.tf_int,name='ant1')
            _ant1 = tf.get_variable("ant1",initializer=_ant1_init,trainable=False)
            _ant2_init = tf.placeholder(shape = ant1.shape, dtype=TFSettings.tf_int,name='ant2')
            _ant2 = tf.get_variable("ant2",initializer=_ant2_init,trainable=False)
            _time_init = tf.placeholder(shape = time_idx.shape, dtype=TFSettings.tf_int,name='time')
            _time = tf.get_variable("time",initializer=_time_init,trainable=False)

            _dropout = tf.placeholder(shape = (), dtype=TFSettings.tf_float,name='dropout')
            _learning_rate = tf.placeholder(shape = (), dtype=TFSettings.tf_float,name='learning_rate')
            
            def _remove_nans_complex(a):
                real = tf.real(a)
                imag = tf.imag(a)
                real = tf.where(tf.is_nan(real), tf.zeros_like(real), real)
                imag = tf.where(tf.is_nan(imag), tf.zeros_like(imag), imag)
                return tf.cast(real,a.dtype) + 1j*tf.cast(imag,a.dtype)
            _V = _remove_nans_complex(_V)
            _V_mod = _remove_nans_complex(_V_mod)
            
            #[Na, Nt, Nd, Nf]
            phi_ik = _c[:,:,tf.newaxis, tf.newaxis] - 8.4480e9/_freqs[tf.newaxis,tf.newaxis,tf.newaxis,:] * _tec[:,:,:,tf.newaxis]
            #[Na, Nt, Nd, Nf]
            g_i = tf.exp(1j*tf.cast(phi_ik,TFSettings.tf_complex))
            g_j_conj = tf.conj(g_i)
            
            indices = tf.stack([_ant1, _time],axis=1)
            #[num_baselines * Nt, Nf, 1, Nd]
            g_i = tf.transpose(tf.gather_nd(g_i,indices)[:,:,tf.newaxis,:],perm=[0,3,2,1])
            #
            indices = tf.stack([_ant2, _time],axis=1)
            #[num_baselines * Nt, Nf, 1, Nd]
            g_j_conj = tf.transpose(tf.gather_nd(g_j_conj,indices)[:,:,tf.newaxis,:],perm=[0,3,2,1])
            g_ij = g_i * g_j_conj
            d = tf.reduce_sum(_V_mod * g_ij,axis=-1)
            weights = tf.where(tf.equal(_flags,1),tf.zeros_like(_flags),_weight_spectrum)
            dropout_mask = tf.nn.dropout(tf.ones_like(weights),_dropout)
            wdd_real = tf.real(_V - d)*dropout_mask
            wdd_imag = tf.imag(_V - d)*dropout_mask

            chi2_real = tf.square(wdd_real)*weights
            #chi2_real = tf.where(tf.is_nan(chi2_real),tf.zeros_like(chi2_real), chi2_real)
            chi2_imag = tf.square(wdd_imag)*weights
            #chi2_imag = tf.where(tf.is_nan(chi2_imag),tf.zeros_like(chi2_imag), chi2_imag)
            chi2_real = tf.reduce_mean(chi2_real)
            chi2_imag = tf.reduce_mean(chi2_imag)
            _chi2 = (chi2_real + chi2_imag)/2.
            _chi2_phase = tf.square(tf.angle(_V) - tf.angle(d))*dropout_mask
            _chi2_phase = tf.reduce_mean(tf.where(tf.is_nan(_chi2_phase), tf.zeros_like(_chi2_phase),_chi2_phase))

            _c_prior_loss = tf.reduce_mean(tf.square(_c)/2.)
            _time_prior_loss = tf.reduce_mean(tf.square(_tec[:,1:,:] - _tec[:,:-1,:])/0.001**2/2.)

            optimizer = tf.train.AdamOptimizer(_learning_rate)
            minimize_op = optimizer.minimize(_chi2 + _c_prior_loss+_time_prior_loss + _chi2_phase)

    sess = tf.InteractiveSession(graph=graph)
    sess.run(tf.global_variables_initializer(), 
            feed_dict = {_tec_init : tec_prior,
                _c_init : c_prior,
                _V_init : V,
                _weight_spectrum_init : weight_spectrum,
                _flags_init : flags,
                _V_mod_init : V_mod,
                _freqs_init : freqs,
                _ant1_init : ant1,
                _ant2_init : ant2,
                _time_init : time_idx})
    epoch = 0
    max_iter = 100
    chi2_array = []
    train_time = []
    while epoch < max_iter:
        lr = 0.01 / ( (epoch / 15) * 2. + 1.)
        dp = min(0.2 + 0.1 * (epoch / 20),0.5)
        
        _,chi2,c_prior_loss,time_prior_loss,chi2_phase = sess.run([minimize_op, _chi2, _c_prior_loss,_time_prior_loss,_chi2_phase], feed_dict = {_dropout : dp,
            _learning_rate : lr})
        chi2_array.append(chi2)
        train_time.append(tm.mktime(tm.gmtime()))
        epoch += 1
        print("Epoch {} | chi2 {} | c_prior_loss {} | time_prior_loss {} | phase lose {}".format(epoch, chi2, c_prior_loss,time_prior_loss,chi2_phase))
        if chi2 < 0.1:
            break
    c_res,tec_res = sess.run([_c, _tec])
    plt.plot(train_time,chi2_array)
    plt.ylabel("loss")
    plt.xlabel("time (s)")
    plt.yscale("log")
    plt.savefig("training_curve_{}.png".format(timestep))
    plt.show()
    return c_res, tec_res, antennas, directions, times[timestep]
    


    
def plot_res(tec,antennas, directions, times):
    phase_center = ac.SkyCoord(np.mean(directions.ra.deg)*au.deg,
            np.mean(directions.dec.deg)*au.deg,frame='icrs')
    array_center = ac.SkyCoord(np.mean(antennas.x.to(au.m).value)*au.m,
                               np.mean(antennas.y.to(au.m).value)*au.m,
                               np.mean(antennas.z.to(au.m).value)*au.m,frame='itrs')
    fixtime = times[0]
    uvw = Pointing(location = array_center.earth_location,obstime = times[0],fixtime=fixtime, phase = phase_center)
    ants_uvw = antennas.transform_to(uvw)
    dirs_uvw = directions.transform_to(uvw)
    X_angular = angular_space(dirs_uvw)
    X = angular_space(dirs_uvw)
    Xstar = make_xstar(X,25)
    tec_nearest = interp_nearest(X_angular[:,0],X_angular[:,1],tec[51,0,:],Xstar[:,0],Xstar[:,1]).reshape([25,25])
    tec_cm = plt.cm.coolwarm

    fig = plt.figure(figsize=(6,6))
    
    extent = (np.min(X[:,0]), np.max(X[:,0]),np.min(X[:,1]), np.max(X[:,1]))

    ax = fig.add_subplot(1,1,1)
    #plot first time slot for example
    #print(X.shape,y.shape,sigma_y.shape)
    sc = ax.imshow(tec_nearest.T,origin='lower',
                    extent=extent,
                    cmap=tec_cm)#X[0][:,0],X[0][:,1],c=y[0],marker='+')
    plt.colorbar(sc)
    ax.set_title("Measured tec")
    plt.show()

    




def main(vis_data):
    with h5py.File(vis_data,'r') as f:
        c,tec, antennas, directions, time = tec_solve(f['V'],
                                    f['V_mod'],
                                    f['freqs'],
                                    f['a1'], 
                                    f['a2'], 
                                    f['time'], 
                                    f['antenna_labels'], 
                                    f['ra'], 
                                    f['dec'],
                                    f['weight_spectrum'],
                                    f['flags'],timestep=[0,1], freq_step = 10)
        tec -= tec[0,...]
        plot_res(tec,antennas, directions, time)
        c,tec, antennas, directions, time = tec_solve(f['V'],
                                    f['V_mod'],
                                    f['freqs'],
                                    f['a1'], 
                                    f['a2'], 
                                    f['time'], 
                                    f['antenna_labels'], 
                                    f['ra'], 
                                    f['dec'],
                                    f['weight_spectrum'],
                                    f['flags'],timestep=[0,1], freq_step = 5,c_prior=c,tec_prior=tec)
        tec -= tec[0,...]
        plot_res(tec,antennas, directions, time)

        


        f = h5py.File("res0.hdf5","w")
        f['tec'] = tec
        f['c'] = c
        f['ra'] = directions.ra.deg
        f['dec'] = directions.dec.deg
        f.close()

            
if __name__ == '__main__':
    main("vis_data.hdf5")
