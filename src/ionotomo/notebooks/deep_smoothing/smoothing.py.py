
# coding: utf-8

# In[1]:

import numpy as np
import tensorflow as tf
import logging
logging.basicConfig(format='%(asctime)s %(message)s')
import pylab as plt
import cmocean
from scipy.spatial import cKDTree
from ionotomo.tomography.pipeline import Pipeline
from ionotomo.settings import TFSettings
from timeit import default_timer
from ionotomo import *
import astropy.coordinates as ac
import astropy.units as au
import gpflow as gp
import sys
import h5py
import threading
from timeit import default_timer
#%matplotlib notebook
from concurrent import futures
from functools import partial
from threading import Lock
import astropy.units as au
import astropy.time as at
from collections import deque
from doubly_stochastic_dgp.dgp import DGP

from ionotomo.bayes.gpflow_contrib import GPR_v2,Gaussian_v2
from scipy.cluster.vq import kmeans2


def _synced_minibatch(*X,minibatch_size=100,seed=0, sess=None, shuffle = True):
    init_placeholders = tuple([tf.placeholder(gp.settings.tf_float,shape=x.shape) for x in X])
    data = tf.data.Dataset.from_tensor_slices(init_placeholders)
    data = data.repeat()
    if shuffle:
        data = data.shuffle(buffer_size=X[0].shape[0], seed=seed)
    data = data.batch(batch_size=tf.constant(minibatch_size,dtype=tf.int64))
    iterator_tensor = data.make_initializable_iterator()
    if sess is not None:
        sess.run(iterator_tensor.initializer, feed_dict={p:x for p,x in zip(init_placeholders,X)})
    return init_placeholders, iterator_tensor.initializer, iterator_tensor.get_next()

class WeightedSVGP(gp.models.svgp.SVGP):
    def __init__(self, obs_weight, X, Y, kern, likelihood, feat=None,
                 mean_function=None,
                 num_latent=None,
                 q_diag=False,
                 whiten=True,
                 minibatch_size=None,
                 Z=None,
                 num_data=None,
                 **kwargs):
        super(WeightedSVGP,self).__init__(X, Y, kern, likelihood, feat=feat,
                 mean_function=mean_function,
                 num_latent=num_latent,
                 q_diag=q_diag,
                 whiten=whiten,
                 minibatch_size=None,
                 Z=Z,
                 num_data=num_data,
                 **kwargs)
        self.obs_weight = gp.DataHolder(obs_weight)             if minibatch_size is None else gp.Minibatch(obs_weight,batch_size=minibatch_size, seed=0)

        
    @gp.params_as_tensors
    def _build_likelihood(self):
        """
        This gives a variational bound on the model likelihood.
        """

        # Get prior KL.
        KL = self.build_prior_KL()

        # Get conditionals
        fmean, fvar = self._build_predict(self.X, full_cov=False)

        # Get variational expectations.
        var_exp = self.likelihood.variational_expectations(fmean, fvar, self.Y) * self.obs_weight

        # re-scale for minibatch size
        scale = tf.cast(self.num_data, gp.settings.float_type) / tf.cast(tf.shape(self.X)[0], gp.settings.float_type)

        return tf.reduce_sum(var_exp) * scale - KL

class WeightedDGP(DGP):
    def __init__(self,obs_weight, X, Y, Z, kernels, likelihood, 
                 num_outputs=None,num_data=None,
                 mean_function=gp.mean_functions.Zero(),  # the final layer mean function
                 **kwargs):
        pass

class Smoothing(object):
    """
    Class for all types of GP smoothing/conditioned prediction
    """

    def __init__(self,datapack):
        if isinstance(datapack, str):
            datapack = DataPack(filename=datapack)
        self.datapack = datapack

    def _make_coord_array(t,d):
        """Static method to pack coordinates
        """
        Nt,Nd = t.shape[0],d.shape[0]
        X = np.zeros([Nt,Nd,3],dtype=np.float64)
        for j in range(Nt):
            for k in range(Nd):
                X[j,k,0] = t[j]
                X[j,k,1:3] = d[k,:]
        X = np.reshape(X,(Nt*Nd,3))
        return X
    
    def _build_sgp_model(self, sess, weights, X, Y, ls_scale, y_scale, 
                         minibatch_size=500, M=1000,Z=None, feature_trainable=False,ls_init=(200,1.),
                         ls_trainable=(True,True), likelihood_var_trainable=True, verbose=False):
        """
        Build svgp model
        """
        N, num_latent = Y.shape
        Z = kmeans2(X, M, minit='points')[0] if Z is None else Z
        num_data = X.shape[0]
        _,_, data = _synced_minibatch(weights, X, Y,minibatch_size=minibatch_size, sess = sess,shuffle=True)
        weights,X,Y = data
        
        with gp.defer_build():
            k_time = gp.kernels.RBF(1,active_dims = [0],
                                    lengthscales=[0.5 if ls_init[0] is None else ls_init[0]/ls_scale[0]])
            k_space = gp.kernels.RBF(2,active_dims = [1,2],
                                    lengthscales=[1.0 if ls_init[1] is None else ls_init[1]/ls_scale[1]])
            for k,f in zip([k_time,k_space],ls_trainable):
                k.lengthscales.set_trainable(f)
                if not f:
                    logging.warning("Setting {} non-trainable".format(k))
            k_time.lengthscales.prior = gp.priors.Gaussian(0.,200./ls_scale[0])
            k_space.lengthscales.prior = gp.priors.Gaussian(1./ls_scale[1],1./ls_scale[1])
            kern = k_time*k_space
            mean = gp.mean_functions.Zero()
            m = WeightedSVGP(weights, X, Y, kern, mean_function = mean, 
                    likelihood=gp.likelihoods.Gaussian(), 
                    Z=Z, num_latent=num_latent,num_data=num_data,
                             minibatch_size=None, whiten=True)
            m.likelihood.variance.set_trainable(likelihood_var_trainable)
            m.q_sqrt = m.q_sqrt.value * 1e-5
            m.feature.set_trainable(feature_trainable)
            m.compile()
        if verbose:
            logging.warning(m)
        return m
    
    def _build_dgp_model(self, depth, sess, weight, X, Y, ls_scale, y_scale,  
                         minibatch_size=500, Z=None,M=100,feature_trainable=False,ls_init=(None,None,None),
                         ls_trainable=(True,True,True),likelihood_var_trainable=True, verbose=False):
        """
        Build svgp model
        """
        N, num_latent = Y.shape
        Z = kmeans2(X, M, minit='points')[0]
        with gp.defer_build():
            k_time = gp.kernels.RBF(1,active_dims = [0],
                                    lengthscales=[0.3 if ls_init[0] is None else ls_init[0]/ls_scale[0]])
            k_space = gp.kernels.RBF(2,active_dims = [1,2],
                                    lengthscales=[0.3 if ls_init[1] is None else ls_init[1]/ls_scale[1]])
            k_freq = gp.kernels.RBF(1,active_dims = [3],
                                    lengthscales=[10. if ls_init[2] is None else ls_init[2]/ls_scale[2]])
            for k,f in zip([k_time,k_space,k_freq],ls_trainable):
                k.lengthscales.set_trainable(f)
                if not f:
                    logging.warning("Setting {} non-trainable".format(k))
            k_time.lengthscales.prior = gp.priors.Gaussian(0,1./3.)
            k_space.lengthscales.prior = gp.priors.Gaussian(1./ls_scale[1],0.5/ls_scale[1])
            kern = k_time*k_space*k_freq
            
            mean = gp.mean_functions.Zero()
            kernels = [kern]
            for l in range(1,depth):
                kernels.append(RBF(4-l, lengthscales=2., variance=2.,ARD=True))
                #kernels[-1].lengthscales.prior = gp.priors.Gaussian(0,1./3.)
            m = DGP(X, Y, Z, kernels, gp.likelihoods.Gaussian(), 
                        minibatch_size=minibatch_size,
                        num_outputs=num_latent,num_samples=1)

            # start things deterministic 
            for layer in m.layers[:-1]:
                layer.q_sqrt = layer.q_sqrt.value * 1e-5 
            for layer in m.layers:
                layer.feature.Z.set_trainable(feature_trainable)
            m.compile()
        if verbose:
            logging.warning(m)
        return m
        
    def _build_model(self,m_type, sess, weight, X, Y, ls_scale, y_scale, **kwargs):
        """
        Build a GP model depending on m_type
        m_type: str, one of 'sgp', 'dgp2', 'dgp3'
        **kwargs are passes to the constructor of the model type.
        """
        if m_type == 'sgp':
            return self._build_sgp_model(sess, weight, X, Y, ls_scale, y_scale,**kwargs)
        elif m_type == 'dgp2':
            return self._build_dgp_model(2,sess,weight, X, Y, ls_scale, y_scale,**kwargs)
        elif m_type == 'dgp3':
            return self._build_dgp_model(3,sess, weight, X, Y, ls_scale, y_scale,**kwargs)
        raise ValueError("{} is invalid model type".format(m_type))

    def _solve_interval(self,phase, error, coords, lock, error_sigma_clip=None, m_type='sgp', 
                        iterations=1000, pargs=None,verbose=False,model_kwargs={}):
        """
        Solve the block of data independently over antennas assuming homogeneity.
        phase: array of shape (Na, Nt, Nd, Nf)
        errors: array of shape (Na, Nt, Nd, Nf), -1 in an element means to mask
        coords: tuple of arrays of shape (Nt,) (Nd,2) (Nf,)
        lock: a mutable lock or None
        m_type: str the model type to use, see build_model
        pargs: str or None, thing to print on start of block
        """
        try:
            if pargs is not None:
                logging.warning("{}".format(pargs))
            
            Na,Nt,Nd,Nf = phase.shape
            
            y = phase.transpose((1,2,0,3)).reshape((Nt*Nd,Na*Nf))#Nt*Nd,Na*Nf
            sigma_y = error.transpose((1,2,0,3)).reshape((Nt*Nd,Na*Nf))#Nt*Nd,Na*Nf
            mask = sigma_y < 0.#Nt*Nd,Na*Nf
            
            y_mean = (y*np.bitwise_not(mask)).sum(axis=0) / (np.bitwise_not(mask).sum(axis=0))#Na*Nf
            y_scale = np.sqrt((y**2*np.bitwise_not(mask)).sum(axis=0)                               / (np.bitwise_not(mask).sum(axis=0)) - y_mean**2) + 1e-6#Na*Nf
            y = (y - y_mean)/y_scale#Nt*Nd,Na*Nf
            var_y = (sigma_y/y_scale)**2#Nt*Nd,Na*Nf
            
            if error_sigma_clip is not None:
                log_var_y = np.log(var_y)#Nt*Nd,Na*Nf
                log_var_y[mask] = np.nan
                E_log_var_y = np.nanmean(log_var_y,axis=0)#Na*Nf
                std_log_var_y = np.nanstd(log_var_y,axis=0)#Na*Nf
                clip_mask = (log_var_y - E_log_var_y) > error_sigma_clip*std_log_var_y#Nt*Nd,Na*Nf
                ignore_mask = np.bitwise_or(mask,clip_mask)#Nt*Nd,Na*Nf
            else:
                ignore_mask = mask
            keep_mask = np.bitwise_not(ignore_mask)#Nt*Nd,Na*Nf
            weight = 1./(var_y+1e-6)#Nt*Nd,Na*Nf
            weight_norm = np.stack([np.percentile(weight[keep_mask[:,i],i],50) for i in range(Na*Nf)],axis=-1)
            weight /= weight_norm + 1e-6
            plt.hist(weight.flatten(),bins=20)
            plt.show()
            weight = np.ones(y.shape)
            weight[ignore_mask] = 0.
             
            t,d = coords
            t_scale = t.std() + 1e-6
            d_scale = np.sqrt((d.std(axis=0)**2).mean()) + 1e-6
            ls_scale = (t_scale,d_scale)
            t = (t - t.mean()) / t_scale
            d = (d - d.mean(axis=0)) / d_scale
            X = Smoothing._make_coord_array(t,d)#Nt*Nd,3
            model_kwargs['Z'] = Smoothing._make_coord_array(t[::3],d)
            
            with tf.Session(graph=tf.Graph()) as sess:
                lock.acquire() if lock is not None else None
                try:
                    model = self._build_model(m_type, sess, weight, X, y, ls_scale, y_scale, **model_kwargs)
                finally:
                    lock.release() if lock is not None else None
                logging.warning("log-likelihood {}".format(model.compute_log_likelihood()))
                opt = gp.train.AdamOptimizer(1e-2)
                opt.minimize(model, maxiter=iterations)
                logging.warning("log-likelihood {}".format(model.compute_log_likelihood()))
                # smooth
                kern_lengthscales = (
                        model.kern.rbf_1.lengthscales.value[0]*ls_scale[0],
                        model.kern.rbf_2.lengthscales.value[0]*ls_scale[1]
                        )
                kern_variance = model.kern.rbf_1.variance.value*model.kern.rbf_2.variance.value*y_scale**2
                if verbose:
                    logging.warning(kern_lengthscales)
                    logging.warning(kern_variance)
                predict_minibatch = 1000
                for start in range(0,X.shape[0],predict_minibatch):
                    stop = min(start+predict_minibatch,X.shape[0])
                    Xs = X[start:stop,:]
                    ystar,varstar = model.predict_y(Xs)#batch,Na
                    ystar = ystar * y_scale + y_mean
                    varstar = varstar * y_scale**2
                    
                    y[start:stop,:] = ystar
                    var_y[start:stop,:] = varstar
            phase = y.reshape([Nt,Nd,Na,Nf]).transpose((2,0,1,3))
            variance = var_y.reshape([Nt,Nd,Na,Nf]).transpose((2,0,1,3))
            return phase,variance,kern_lengthscales, kern_variance
        except Exception as e:
            print(e)
            
    def solve_and_apply_ensemble(self, save_datapack, ant_idx, time_idx, dir_idx, freq_idx, iterations,
                                 interval, shift, init_solutions = None, num_threads=1,verbose=False,model_kwargs = {}):
        if init_solutions is not None:
            data = np.load(init_solutions)

            kern_ls = data['kern_ls']
            kern_var = data['kern_var']
            kern_times = data['time']
            kern_antenna_labels = data['antenna']

        
        datapack = self.datapack
        directions, patch_names = datapack.get_directions(dir_idx)
        times,timestamps = datapack.get_times(time_idx)
        antennas,antenna_labels = datapack.get_antennas(ant_idx)
        freqs = datapack.get_freqs(freq_idx)

        if ant_idx is -1:
            ant_idx = range(len(antennas))
        if time_idx is -1:
            time_idx = range(len(times))
        if freq_idx is -1:
            freq_idx = range(len(freqs))
        if dir_idx is -1:
            dir_idx = range(len(directions))

        phase = datapack.get_phase(ant_idx,time_idx,dir_idx,freq_idx)
        Na,Nt,Nd,Nf = phase.shape
        logging.warning("Working on shapes {}".format(phase.shape))

        assert interval <= Nt

        variance = datapack.get_variance(ant_idx,time_idx,dir_idx,freq_idx)
        error = np.sqrt(variance)
        data_mask = variance < 0
        error[data_mask] = -1
        logging.warning("Total masked phases: {}".format(np.sum(data_mask)))

        t = times.mjd*86400.#mjs
        d = np.array([directions.ra.deg, directions.dec.deg]).T
        
        lock = Lock()
        with futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            jobs = []

            for j,aj in enumerate(time_idx[::shift]):
                start = j*shift
                stop = min(start+interval,Nt)
                time_slice = slice(start,stop,1)
                
                if init_solutions is not None:
                    ###
                    # interpolate kern_params with this interval/shift
                    mean_time = np.mean(times.gps[time_slice])

                    model_kwargs['ls_init'] = (
                            np.interp(mean_time, kern_times, kern_ls[ant_idx,:,1].mean(0)),
                            np.interp(mean_time, kern_times, kern_ls[ant_idx,:,0].mean(0))
                            )
                    #logging.warning(model_kwargs['ls_init'])
                    
                # initial ls_scale (if they exist)

# (phase, error, coords, lock, error_sigma_clip=4., m_type='sgp', 
#                         iterations=1000, pargs=None,verbose=False,model_kwargs={}):
                jobs.append(executor.submit(
                    self._solve_interval,
                    phase[:,time_slice,:,:],
                    error[:,time_slice,:,:],
                    (t[time_slice],d),
                    lock,
                    error_sigma_clip = None,
                    m_type='sgp',
                    iterations=iterations,
                    pargs="Working on time chunk ({}) {} to ({}) {}".format(
                        start,timestamps[start],stop-1,timestamps[stop-1]),
                    verbose=verbose,
                    model_kwargs = model_kwargs
                    )
                    )
            results = futures.wait(jobs)
            if verbose:
                logging.warning(results)
            results = [j.result() for j in jobs]

            phase_mean = np.zeros(phase.shape)
            phase_weights = np.zeros(phase.shape)
            variance_mean = np.zeros(variance.shape)
            res_idx = 0
            for j,aj in enumerate(time_idx[::shift]):
                start = j*interval
                stop = min((j+1)*interval,Nt)
                time_slice = slice(start,stop,1)
                res = results[res_idx]
                p,v,kern_ls,kern_var = res
                phase_mean[:,time_slice,:,:] += p/(v+1e-3)
                phase_weights[:,time_slice,:,:] += 1./(v+1e-3)
                res_idx += 1
            variance_mean = 1/(phase_weights+1e-3)
            phase_mean /= (phase_weights+1e-3)
            datapack.set_phase(phase_mean, ant_idx=ant_idx,time_idx=time_idx,dir_idx=dir_idx,freq_idx=freq_idx)
            datapack.set_variance(variance_mean, ant_idx=ant_idx,time_idx=time_idx,dir_idx=dir_idx,freq_idx=freq_idx)
            datapack.save(save_datapack)
            
    
    def solve_and_apply_ensemble_blocked(self, save_datapack, ant_idx, time_idx, dir_idx, freq_idx, 
                                 interval, shift, num_blocks = 4, num_threads=1,verbose=False):
        
        datapack = self.datapack
        directions, patch_names = datapack.get_directions(dir_idx)
        times,timestamps = datapack.get_times(time_idx)
        antennas,antenna_labels = datapack.get_antennas(ant_idx)
        freqs = datapack.get_freqs(freq_idx)

        if ant_idx is -1:
            ant_idx = range(len(antennas))
        if time_idx is -1:
            time_idx = range(len(times))
        if freq_idx is -1:
            freq_idx = range(len(freqs))
        if dir_idx is -1:
            dir_idx = range(len(directions))

        phase = datapack.get_phase(ant_idx,time_idx,dir_idx,freq_idx)
        Na,Nt,Nd,Nf = phase.shape
        logging.warning("Working on shapes {}".format(phase.shape))

        assert interval <= Nt

        variance = datapack.get_variance(ant_idx,time_idx,dir_idx,freq_idx)
        error = np.sqrt(variance)
        data_mask = variance < 0
        error[data_mask] = -1
        logging.warning("Total masked phases: {}".format(np.sum(data_mask)))

        t = times.mjd*86400.#mjs
        d = np.array([directions.ra.deg, directions.dec.deg]).T
        f = freqs
        coords = (t,d,f)
        
        lock = Lock()
        with futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            jobs = []
            mean_count = np.zeros(phase.shape)

            for i,ai in enumerate(ant_idx):
                for j,aj in enumerate(time_idx[::shift]):
                    start = j*shift
                    stop = min(start+interval,Nt)
                    time_slice = slice(start,stop,time_sampling)

                    ###
                    # interpolate kern_params with this interval/shift
                    mean_time = np.mean(times.gps[time_slice])
                    
                    # d, t, f, v
                    kern_params = [
                            np.interp(mean_time, kern_times, kern_ls[i,:,0]),
                            np.interp(mean_time, kern_times, kern_ls[i,:,1]),
                            np.interp(mean_time, kern_times, kern_ls[i,:,2]),
                            np.interp(mean_time, kern_times, kern_var[i,:,0])
                            ]

                    mean_count[i,time_slice,directional_slice,freq_slice] += 1
                    
                    
#                    for l,al in enumerate(freq_idx):
#                        freq_slice = slice(l,l+1)
                    jobs.append(executor.submit(
                        Smoothing._apply_block_svgp,
                        phase[i,time_slice,directional_slice,freq_slice].copy(),
                        error[i,time_slice,directional_slice,freq_slice].copy(),
                        (t[time_slice],d[directional_slice],f[freq_slice]),
                        lock,
                        kern_params=kern_params,
                        pargs="Working on {} time chunk ({}) {} to ({}) {} at {} to {} MHz".format(antenna_labels[i],
                            start,timestamps[start],stop-1,timestamps[stop-1], freqs[0]/1e6, freqs[-1]/1e6),
                        verbose=verbose
                        )
                        )
            results = futures.wait(jobs)
            if verbose:
                logging.warning(results)
            results = [j.result() for j in jobs]

            phase_mean = np.zeros(phase.shape)
            variance_mean = np.zeros(variance.shape)
            res_idx = 0
            for i,ai in enumerate(ant_idx):
                for j,aj in enumerate(time_idx[::shift]):
                    start = j*interval
                    stop = min((j+1)*interval,Nt)
                    time_slice = slice(start,stop,time_sampling)
                    res = results[res_idx]
                    phase_mean[i,time_slice,directional_slice,freq_slice] += res[0]
                    variance_mean[i,time_slice,directional_slice,freq_slice] += res[1]
                    res_idx += 1
            phase_mean /= mean_count
            variance_mean /= mean_count
            datapack.set_phase(phase_mean, ant_idx=ant_idx,time_idx=time_idx,dir_idx=dir_idx,freq_idx=freq_idx)
            datapack.set_variance(variance_mean, ant_idx=ant_idx,time_idx=time_idx,dir_idx=dir_idx,freq_idx=freq_idx)

            datapack.save(save_datapack)
    
if __name__=='__main__':
    import os

    if len(sys.argv) == 2:
        starting_datapack = sys.argv[1]
    else:
        starting_datapack = "../../data/rvw_datapack_full_phase_dec27.hdf5"
    smoothing = Smoothing(starting_datapack)
    model_kwargs = {'minibatch_size':500, 'M':40*15,'feature_trainable':False,'ls_init':(300,1.5),
                         'ls_trainable':(True,True,True), 'verbose':True,'likelihood_var_trainable':True}
#     smoothing.solve_and_apply_ensemble(starting_datapack.replace('.hdf5','_smoothed_ensemble.hdf5'),
#                                        range(5,7), -1, -1, range(2), iterations=1000,
#                                  interval = 30, shift = 6, init_solutions='../../bayes/gp_params_fixed_scales.npz',
#                                        num_threads=1,verbose=True,model_kwargs = model_kwargs)

#     smoothing.solve_time_intervals("gp_params.npz",range(1,62),-1,-1,range(0,20),32,32,num_threads=16,verbose=True)
#    refined_params = smoothing.refine_statistics_timeonly('gp_params.npz')
#    print(refined_params.shape)
#    smoothing.solve_time_intervals("gp_params_fixed_scales.npz",range(1,62),-1,-1,range(0,20),32,32,num_threads=16,verbose=True,refined_params=refined_params)
#    plt.ion()
#    smoothing.refine_statistics_timeonly('gp_params.npz')
#    smoothing.refine_statistics('gp_params.npz')
#    smoothing.refine_statistics_timeonly('gp_params_fixed_scales.npz')
#    smoothing.refine_statistics('gp_params_fixed_scales.npz')
#    plt.ioff()
#     smoothing.apply_solutions(starting_datapack.replace('.hdf5','_refined_smoothed.hdf5'), 
#             "gp_params_fixed_scales.npz",range(1,62), -1, -1, range(0,20), 32, 32, num_threads=1,verbose=True)


# In[9]:

phase = smoothing.datapack.get_phase(-1,-1,-1,-1)
freqs = smoothing.datapack.get_freqs(-1)


# In[11]:

tec = phase*freqs/-8.4480e9


# In[30]:

tec.me(-1)[51,0,:]


# In[34]:

df = phase[...,1:]-phase[...,:-1]


# In[38]:

plt.hist(df.flatten(),bins=100)
plt.show()


# In[39]:

q = np.linspace(0,100,100)
p = np.percentile(df.flatten(),q)


# In[42]:

plt.plot(q,p)
plt.show()


# In[44]:

np.where(np.abs(df)>0.5)


# In[73]:

wp = np.angle(np.exp(1j*phase))


# In[57]:

wp[:-1]-wp[1:]


# In[58]:

wp


# In[74]:

uwp = np.unwrap(wp)


# In[75]:

tec = uwp*freqs/-8.4480e9


# In[77]:

tec.mean(-1)


# In[70]:

#sum f_i * nu/C - sum (f_i+2piK)*nu/C = sum 2piK nu_i/C = 2piK mean(nu)/C


# In[72]:

2*np.pi*np.mean(freqs)/-8.4480e9


# In[78]:

tec[51,2347,28,:]


# In[88]:

y=((uwp[...,None] + np.arange(-3,4)*np.pi*2)*freqs[:,None]/-8.4480e9).std(3).argmin(-1)


# In[89]:

y.max()


# In[90]:

y.min()


# In[3]:


f = np.angle(np.exp(1j*smoothing.datapack.get_phase([51],[0],-1,[0]).flatten()))

directions, _ = smoothing.datapack.get_directions(-1)
freqs = smoothing.datapack.get_freqs(-1)
x,y = directions.ra.deg,directions.dec.deg
x -= x.mean()
y -= y.mean()
X = np.array([x,y]).T



import pymc3 as pm




with pm.Model() as model:
    df = pm.Uniform('df',lower=0,upper=1,shape=[f.size])
    f = (f + df*(2*np.pi))*(freqs[0]/-8.4480e9)
    f -= f.mean()
    f /= f.std()
    cov_func = pm.gp.cov.ExpQuad(2, ls=[0.5,0.5])
    gp = pm.gp.Marginal(cov_func=cov_func)
    y0_ = gp.marginal_likelihood('y0',X,f,0.01)
    trace = pm.sample(10000,chains=4)
pm.traceplot(trace,combined=True)
plt.show()


# In[4]:

pm.summary(trace)


# In[10]:

np.where((smoothing.datapack.get_phase([51],[0],-1,[0]).flatten() -            np.angle(np.exp(1j*smoothing.datapack.get_phase([51],[0],-1,[0]).flatten())))/2/np.pi < 0.5)


# In[ ]:



