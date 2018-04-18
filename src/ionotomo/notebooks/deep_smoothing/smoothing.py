
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

from scipy.spatial.distance import pdist,squareform
import os


# In[72]:

class NNComposedKernel(gp.kernels.Kernel):
    """
    This kernel class allows for easily adding a NN (or other function) to a GP model.
    The kernel does not actually do anything with the NN.
    """
    
    def __init__(self, kern, f, f_scope):
        """
        kern.input_dim needs to be consistent with the output dimension of f
        """
        super().__init__(kern.input_dim,active_dims=kern.active_dims)
        self.kern = kern
        self._f = lambda x: tf.cast(f(x), gp.settings.float_type) #function to call on input
        self._f_scope = f_scope #learnable variables that f depends on
        
    def f(self, X):
        if X is not None:
            return self._f(X)
    
    def _get_f_vars(self):        
        return tf.trainable_variables(scope=self._f_scope)

    @gp.autoflow([gp.settings.float_type, [None,None]])
    def compute_f(self, X):
        return self.f(X)
    
    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)        
        return self.kern.K(self.f(X), self.f(X2), presliced=True)
    
    def Kdiag(self, X, presliced=False):
        if not presliced:
            X,_ = self._slice(X, None)
        return self.kern.Kdiag(self.f(X))

class KernelSpaceInducingPoints(gp.features.InducingPoints):
    def Kuf(self, kern, Xnew):
        assert isinstance(kern, KernelWithNN)
        return kern.K(self.Z, kern.f(Xnew))

class NNComposedKernel_(KernelWithNN):
    """
    This kernel class applies f() to X before calculating K
    """
    
    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        return super().K(self.f(X), self.f(X2),presliced=True)
    
    def Kdiag(self, X, presliced=False):
        if not presliced:
            X = self._slice(X, None)
        return super().Kdiag(self.f(X),presliced=True)
    
# we need to add these extra functions to the model so the tensorflow variables get picked up
class NN_SVGP(gp.models.svgp.SVGP):
    def get_NNKernels(self,kern=None):
        if kern is None:
            kern = self.kern
        out = []
        for c in kern.children.values():
            if isinstance(c,gp.kernels.Kernel):
                if isinstance(c,NNComposedKernel):
                    out.append(c)
                else:
                    out = out + self.get_NNKernels(c)
        return out
    @property
    def all_f_vars(self):
        NN_kerns = self.get_NNKernels()
        f_vars = []
        for k in NN_kerns:
            f_vars = f_vars + k._get_f_vars()
        return f_vars
        
            
    @property
    def trainable_tensors(self):
        f_vars = self.all_f_vars
        try:
            return super().trainable_tensors + f_vars
        except:
            return super().trainable_tensors
            
    @property
    def initializables(self):
        f_vars = self.all_f_vars
        try:
            return super().initializables + f_vars
        except:
            return super().initializables
        
def scaled_square_dist(X,X2,lengthscales):
    '''
    r_ij = sum_k (x_ik - x_jk)*(x_ik - x_jk)
        = sum_k x_ik*x_ik - x_ik*x_jk - x_jk*x_ik + x_jk*x_jk
        =  Xs - 2 X.X^t + Xs.T
        
    r_ij = sum_k (x_ik - y_jk)*(x_ik - y_jk)
        = sum_k x_ik*x_ik - x_ik*y_jk - y_jk*x_ik + y_jk*y_jk
        =  Xs - 2 X.X^t + Ys.T
    '''
    X = X / lengthscales
    Xs = tf.reduce_sum(tf.square(X), axis=1)

    if X2 is None:
        dist = -2 * tf.matmul(X, X, transpose_b=True)
        dist += tf.reshape(Xs, (-1, 1))  + tf.reshape(Xs, (1, -1))
        return dist

    X2 = X2 / lengthscales
    X2s = tf.reduce_sum(tf.square(X2), axis=1)
    dist = -2 * tf.matmul(X, X2, transpose_b=True)
    dist += tf.reshape(Xs, (-1, 1)) + tf.reshape(X2s, (1, -1))
    return dist

def anisotropic_modulation(ndim, M, scope):
    def modulation(X,ndim=ndim,M=M,scope=scope):
        with tf.variable_scope(scope,reuse=tf.AUTO_REUSE) as scope:
            factor = tf.get_variable("factor",shape=(M,1),dtype=gp.settings.float_type,
                                     initializer=\
                                     tf.zeros_initializer(dtype=gp.settings.float_type))
            factor = 1.5*tf.nn.sigmoid(factor) + 0.25 # between 0.25 and 1.75 modulations starting at 1.
            points = tf.get_variable("points",shape=(M,ndim),dtype=gp.settings.float_type,
                                     initializer=\
                                     tf.random_uniform_initializer(minval=-2,maxval=2,dtype=gp.settings.float_type))
            scale = tf.nn.softplus(tf.get_variable("scale",shape=(),dtype=gp.settings.float_type,
                                    initializer=tf.ones_initializer(dtype=gp.settings.float_type))) + 1e-6
            dist = scaled_square_dist(X,points,scale)
            weights = tf.exp(-dist/2.) #N, M
            weights /= tf.reduce_sum(weights,axis=1,keepdims=True,name='weights')# N,1
            factor = tf.matmul(weights, factor,name='factor')#N, 1
            res = X/factor      
            return res
    return modulation
        
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

class WeightedSVGP(NN_SVGP):
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
        scale = scale / tf.reduce_mean(self.obs_weight)
        return tf.reduce_sum(var_exp) * scale - KL


def get_only_vars_in_model(variables, model):
    reader = tf.train.NewCheckpointReader(model)
    var_to_shape_map = reader.get_variable_to_shape_map()
    vars_in_model = [k for k in sorted(var_to_shape_map)]
    out_vars = []
    for var in variables:
        v = var.name.split(":")[0]
        
        if v in vars_in_model:
            if tuple(var.shape.as_list()) != reader.get_tensor(v).shape:
                logging.warning("{} has shape mis-match: {} {}".format(v,
                    tuple(var.shape.as_list()), reader.get_tensor(v).shape))
                continue
            out_vars.append(var)
    return out_vars

def rename(model,prefix='WeightedSVGP',index=""):
    with tf.Session(graph=tf.Graph()) as sess:
        reader = tf.train.NewCheckpointReader(model)
        var_to_shape_map = reader.get_variable_to_shape_map()
        vars_in_model = [k for k in sorted(var_to_shape_map)]
        for v in vars_in_model:
            t = reader.get_tensor(v)
            if 'WeightedSVGP' in v:
                new_name = "/".join(['WeightedSVGP-{}'.format(index)] + v.split('/')[1:])
                var =  tf.Variable(t,name=new_name)
            else:
                var = tf.Variable(t,name=v)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.save(sess, model)


# In[73]:



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

    def __init__(self,datapack, proj_dir):
        if isinstance(datapack, str):
            datapack = DataPack(filename=datapack)
        self.datapack = datapack
        self.proj_dir = os.path.abspath(proj_dir)
        try:
            os.makedirs(self.proj_dir)
        except:
            pass

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
    
    def _make_coord_array_full(a,t,d,f):
        """Static method to pack coordinates
        """
        Na,Nt,Nd,Nf = a.shape[0],t.shape[0],d.shape[0],f.shape[0]
        X = np.zeros([Na,Nt,Nd,Nf,6],dtype=np.float64)
        for i in range(Na):
            for j in range(Nt):
                for k in range(Nd):
                    for l in range(Nf):
                        X[i,j,k,l,0:2] = a[i,:]
                        X[i,j,k,l,2] = t[j]
                        X[i,j,k,l,3:5] = d[k,:]
                        X[i,j,k,l,5] = f[l]
        X = np.reshape(X,(Na*Nt*Nd*Nf,6))
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
            m.q_sqrt = m.q_sqrt.value * 0.4
            m.feature.set_trainable(feature_trainable)
            m.compile()
        if verbose:
            logging.warning(m)
        return m
    
    def _build_sgp_model_full(self, sess, weights, X, Y, ls_scale, y_scale, 
                         minibatch_size=500, M=1000,Z=None, feature_trainable=False,ls_init=(5.,200,1.),
                         ls_trainable=(True, True,True), likelihood_var_trainable=True, verbose=False):
        """
        Build svgp model
        """
        N, num_latent = Y.shape
        Z = kmeans2(X, M, minit='points')[0] if Z is None else Z
        num_data = X.shape[0]
        _,_, data = _synced_minibatch(weights, X, Y,minibatch_size=minibatch_size, sess = sess,shuffle=True)
        weights,X,Y = data
        
        
        with gp.defer_build():
            k_space = gp.kernels.RBF(2,active_dims = [0,1],
                                    lengthscales=[1.0 if ls_init[0] is None else ls_init[0]/ls_scale[0]])
            k_time = gp.kernels.RBF(1,active_dims = [2],
                                    lengthscales=[0.5 if ls_init[1] is None else ls_init[1]/ls_scale[1]])
            k_dir = gp.kernels.RBF(2,active_dims = [3,4],
                                    lengthscales=[1.0 if ls_init[2] is None else ls_init[2]/ls_scale[2]])
            k_freq = gp.kernels.Polynomial(1,active_dims = [5], degree=1.,variance=1.)
            for k,f in zip([k_space,k_time,k_dir],ls_trainable):
                k.lengthscales.set_trainable(f)
                if not f:
                    logging.warning("Setting {} non-trainable".format(k))
            k_space.lengthscales.prior = gp.priors.Gaussian(0.,10./ls_scale[0])
            k_time.lengthscales.prior = gp.priors.Gaussian(0.,200./ls_scale[1])
            k_dir.lengthscales.prior = gp.priors.Gaussian(1./ls_scale[2],1./ls_scale[2])
            
            # allow length scale to change depending on loc |X - X'|^2/ls^2 -> |X/f(X) - X'/f(X')|^2/ls^2
            k_space = NNComposedKernel(k_space, anisotropic_modulation(ndim=2, M=7, scope='f_space_ls'), 'f_space_ls')
            k_time = NNComposedKernel(k_time, anisotropic_modulation(ndim=1, M=4, scope='f_time_ls'), 'f_time_ls')
            k_variance = NNComposedKernel(
                gp.kernels.Polynomial(2,active_dims = [0,1], degree=1.,variance=1.), 
                anisotropic_modulation(ndim=2, M=7, scope='f_space_var'), 'f_space_var')
            
            kern = k_variance*k_space*k_time*k_dir*k_freq
            mean = gp.mean_functions.Zero()
            m = WeightedSVGP(weights, X, Y, kern, mean_function = mean, 
                    likelihood=gp.likelihoods.Gaussian(), 
                    Z=Z, num_latent=num_latent,num_data=num_data,
                             minibatch_size=None, whiten=True)
            m.likelihood.variance.set_trainable(likelihood_var_trainable)
            m.q_sqrt = m.q_sqrt.value * 0.4
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
        elif m_type == 'sgp_full':
            return self._build_sgp_model_full(sess, weight, X, Y, ls_scale, y_scale,**kwargs)
#         elif m_type == 'dgp2':
#             return self._build_dgp_model(2,sess,weight, X, Y, ls_scale, y_scale,**kwargs)
#         elif m_type == 'dgp3':
#             return self._build_dgp_model(3,sess, weight, X, Y, ls_scale, y_scale,**kwargs)
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
#             plt.hist(weight.flatten(),bins=20)
#             plt.show()
            weight = np.ones(y.shape)
            weight[ignore_mask] = 0.
             
            t,d = coords
            t_scale = t.std() + 1e-6
            d_scale = np.sqrt((d.std(axis=0)**2).mean()) + 1e-6
            ls_scale = (t_scale,d_scale)
            t = (t - t.mean()) / t_scale
            d = (d - d.mean(axis=0)) / d_scale
            X = Smoothing._make_coord_array(t,d)#Nt*Nd,3
            ###
            # set Z explicitly to spacing of ::3 in time
            model_kwargs['Z'] = Smoothing._make_coord_array(t[::3],d)
            
            with tf.Session(graph=tf.Graph()) as sess:
                lock.acquire() if lock is not None else None
                try:
                    model = self._build_model(m_type, sess, weight, X, y, ls_scale, y_scale, **model_kwargs)
                finally:
                    lock.release() if lock is not None else None
                logging.warning("Initial log-likelihood {}".format(model.compute_log_likelihood()))
                opt = gp.train.AdamOptimizer(1e-2)
                opt.minimize(model, maxiter=iterations)
                logging.warning("Final log-likelihood {}".format(model.compute_log_likelihood()))
                # smooth
                kern_lengthscales = (
                        model.kern.rbf_1.lengthscales.value[0]*ls_scale[0],
                        model.kern.rbf_2.lengthscales.value[0]*ls_scale[1]
                        )
                kern_variance = model.kern.rbf_1.variance.value*model.kern.rbf_2.variance.value*y_scale**2
                if verbose:
                    logging.warning(model)
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
            logging.warning(e)
            
    def _solve_interval_full(self,model_file, phase, error, coords, lock, load_model = None, error_sigma_clip=None, m_type='sgp_full', 
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
        assert "_full" in m_type
#         try:
        if pargs is not None:
            logging.warning("{}".format(pargs))

        Na,Nt,Nd,Nf = phase.shape
        freqs = coords[3]

        y = (phase*(freqs/-8.4480e9)).reshape((Na*Nt*Nd*Nf,1)) #Na*Nt*Nd*Nf, 1
        sigma_y = (error*(freqs/8.4480e9)).reshape((Na*Nt*Nd*Nf,1)) #Na*Nt*Nd*Nf, 1
        mask = sigma_y < 0. #Na*Nt*Nd*Nf, 1

        y_mean = np.average(y,weights = np.bitwise_not(mask))# scalar
        y_scale = np.sqrt(np.average(y**2,weights = np.bitwise_not(mask))                          - y_mean**2) + 1e-6 #scalar
        y = (y - y_mean)/y_scale#Na*Nt*Nd*Nf, 1
        var_y = (sigma_y/y_scale)**2#Na*Nt*Nd*Nf, 1

        if error_sigma_clip is not None:
            log_var_y = np.log(var_y)#Na*Nt*Nd*Nf, 1
            log_var_y[mask] = np.nan
            E_log_var_y = np.nanmean(log_var_y,axis=0)#1
            std_log_var_y = np.nanstd(log_var_y,axis=0)#1
            clip_mask = (log_var_y - E_log_var_y) > error_sigma_clip*std_log_var_y##Na*Nt*Nd*Nf, 1
            ignore_mask = np.bitwise_or(mask,clip_mask)#Na*Nt*Nd*Nf, 1
        else:
            ignore_mask = mask
        keep_mask = np.bitwise_not(ignore_mask)#Na*Nt*Nd*Nf, 1
#             weight = 1./(var_y+1e-6)#Na*Nt*Nd*Nf, 1
#             weight_norm = np.stack([np.percentile(weight[keep_mask[:,i],i],50) for i in range(Na*Nf)],axis=-1)
#             weight /= weight_norm + 1e-6
# #             plt.hist(weight.flatten(),bins=20)
# #             plt.show()
        weight = np.ones(y.shape)
        weight[ignore_mask] = 0.

        x,t,d,f = coords
        x_scale = np.sqrt((x.std(axis=0)**2).mean()) + 1e-6
        t_scale = t.std() + 1e-6
        d_scale = np.sqrt((d.std(axis=0)**2).mean()) + 1e-6
        f_scale = f.std() + 1e-6
        ls_scale = (x_scale,t_scale,d_scale,f_scale)
        x = (x - x.mean(axis=0)) / x_scale
        t = (t - t.mean()) / t_scale
        d = (d - d.mean(axis=0)) / d_scale
        f = (f - f.mean()) / f_scale
        X = Smoothing._make_coord_array_full(x,t,d,f)#Na*Nt*Nd*Nf,6
        ###
        # set Z explicitly to spacing of ::3 in time
        model_kwargs['Z'] = None#Smoothing._make_coord_array(t[::3],d)


        with tf.Session(graph=tf.Graph()) as sess:
            lock.acquire() if lock is not None else None
            try:
                model = self._build_model(m_type, sess, weight, X, y, ls_scale, y_scale, **model_kwargs)
            finally:
                lock.release() if lock is not None else None
            if load_model is not None:
                try:
                    all_vars = model.trainable_tensors
                    rename(load_model,prefix='WeightedSVGP',index=model.index)
                    all_vars = get_only_vars_in_model(all_vars,load_model)
                    saver = tf.train.Saver(all_vars)
                    saver.restore(sess, load_model)
                    model.compile()
                    logging.warning("Loaded model {}".format(load_model))
                    logging.warning(model)
                except Exception as e:
                    logging.warning(e)
                    logging.warning("Unable to load {}".format(load_model))
            

            logging.warning("Initial log-likelihood {}".format(model.compute_log_likelihood()))
            opt = gp.train.AdamOptimizer(1e-2)
            opt.minimize(model, maxiter=iterations)
            logging.warning("Final log-likelihood {}".format(model.compute_log_likelihood()))
            f_vars = model.all_f_vars
            for var,val in zip(f_vars,sess.run(f_vars)):
                logging.warning("{} {}".format(var.name,val))
            all_vars = model.trainable_tensors
            saver = tf.train.Saver(all_vars)
            save_path = saver.save(sess, model_file)
            logging.warning("Saved model to {}".format(save_path))

            if verbose:
                logging.warning(model)
            predict_minibatch = 1000
            for start in range(0,X.shape[0],predict_minibatch):
                stop = min(start+predict_minibatch,X.shape[0])
                Xs = X[start:stop,:]
                ystar,varstar = model.predict_y(Xs)#minibatch,1
                ystar = ystar * y_scale + y_mean
                varstar = varstar * y_scale**2#minibaatch,1

                y[start:stop,:] = ystar
                var_y[start:stop,:] = varstar
        phase = y.reshape([Na,Nt,Nd,Nf])*(-8.4480e9/freqs)**2
        variance = var_y.reshape([Na,Nt,Nd,Nf])*(-8.4480e9/freqs)**2
        return phase,variance
#         except Exception as e:
#             logging.warning(e)
#             logging.warning("Failed interval solve {}".format(model_file))
            
    def solve_and_apply_ensemble(self, save_datapack, ant_idx, time_idx, dir_idx, freq_idx, iterations,
                                 interval, shift, init_solutions = None, num_threads=1,verbose=False,
                                 model_kwargs = {}):
        """
        Solve the problem using model_kwargs and then take an ensemble average over interval
        and shift.
        """
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

        enu = ENU(obstime=times[0],location = self.datapack.radio_array.get_center())
        
        ant_enu = antennas.transform_to(enu)
        x = np.array([ant_enu.east.to(au.km).value, ant_enu.north.to(au.km).value]).T
        t = times.mjd*86400.#mjs
        d = np.array([directions.ra.deg, directions.dec.deg]).T
        f = freqs
        
        lock = Lock()
        with futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            jobs = []

            for j,aj in enumerate(time_idx[::shift]):
                start = j*shift
                stop = min(start+interval,Nt)
                time_slice = slice(start,stop,1)
                
#                 if init_solutions is not None:
#                     ###
#                     # interpolate kern_params with this interval/shift
#                     mean_time = np.mean(times.gps[time_slice])

#                     model_kwargs['ls_init'] = (
#                             np.interp(mean_time, kern_times, kern_ls[ant_idx,:,1].mean(0)),
#                             np.interp(mean_time, kern_times, kern_ls[ant_idx,:,0].mean(0))
#                             )
                    #logging.warning(model_kwargs['ls_init'])
                    
                # initial ls_scale (if they exist)

# (phase, error, coords, lock, error_sigma_clip=4., m_type='sgp', 
#                         iterations=1000, pargs=None,verbose=False,model_kwargs={}):
                model_file = os.path.join(self.proj_dir,"model_{}_{}".format(start,stop))
                self._solve_interval_full(
                    model_file,
                    phase[:,time_slice,:,:],
                    error[:,time_slice,:,:],
                    (x, t[time_slice],d,f),
                    lock,
                    load_model = model_file,
                    error_sigma_clip = None,
                    m_type='sgp_full',
                    iterations=iterations,
                    pargs="Working on time chunk ({}) {} to ({}) {}".format(
                        start,timestamps[start],stop-1,timestamps[stop-1]),
                    verbose=verbose,
                    model_kwargs = model_kwargs
                    )
                return
                jobs.append(executor.submit(
                    self._solve_interval_full,
                    model_file,
                    phase[:,time_slice,:,:],
                    error[:,time_slice,:,:],
                    (x, t[time_slice],d,f),
                    lock,
                    load_model = model_file,
                    error_sigma_clip = None,
                    m_type='sgp_full',
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
                p,v = res
                phase_mean[:,time_slice,:,:] += p/(v+1e-6)
                variance_mean[:,time_slice,:,:] += 1.
                phase_weights[:,time_slice,:,:] += 1./(v+1e-6)
                res_idx += 1
            phase_mean /= (phase_weights+1e-6)
            variance_mean /= (phase_weights+1e-6)
            datapack.set_phase(phase_mean, ant_idx=ant_idx,time_idx=time_idx,dir_idx=dir_idx,freq_idx=freq_idx)
            datapack.set_variance(variance_mean, ant_idx=ant_idx,time_idx=time_idx,dir_idx=dir_idx,freq_idx=freq_idx)
            datapack.save(save_datapack)
    
if __name__=='__main__':
    import os

    if len(sys.argv) == 2:
        starting_datapack = sys.argv[1]
    else:
        starting_datapack = "../../data/rvw_datapack_full_phase_dec27_unwrap.hdf5"
    smoothing = Smoothing(starting_datapack,'projects')
    model_kwargs = {'minibatch_size':500, 'M':40*15,'feature_trainable':False,'ls_init':(5.,70,0.3),
                         'ls_trainable':(True,True,True), 'verbose':False,'likelihood_var_trainable':True}
    smoothing.solve_and_apply_ensemble(starting_datapack.replace('.hdf5','_smoothed_ensemble.hdf5'),
                                       range(5,7), -1, -1, range(2), iterations=10,
                                 interval = 30, shift = 6, init_solutions='../../bayes/gp_params_fixed_scales.npz',
                                       num_threads=1,verbose=True,model_kwargs = model_kwargs)

# #     smoothing.solve_time_intervals("gp_params.npz",range(1,62),-1,-1,range(0,20),32,32,num_threads=16,verbose=True)
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


# May 4: 
#     Pick-up Truck (Early as possible)
#     Drive to Antsirabe (170km 4h)
#     City/Lakes/Fill Gas/Fill travel food
# May 5:
#     Drive to Ranomafana National Park (230km 5h)
# May 6: 
#     Hike Ranomafana
# May 7: 
#     Ranomafana to Isalo National Park (350km 6h)
# May 8: 
#     Hike Isalo
# May 9: 
#     Isalo to Antsirabe (530km 10h)
# May 10: 
#     Antsirabe to Mahambo (590km 12h)
#     -- OR --
#     Antsirabe to Toamasina (500km 10h)
# May 11: 
#     Mahambo to Ile Sainte Marie (Boat)
#     -- OR --
#     Toamasina to Mahambo (90km 3h)
#     Mahambo to Ile Sainte Marie (Boat)
# May 12: 
#     Ile Sainte Marie / Ile Aux Nattes
# May 13: 
#     Ile Sainte Marie / Ile Aux Nattes
# May 14: 
#     Ile Sainte Marie / Ile Aux Nattes
# May 15: 
#     Ile Sainte Marie to Mahambo (early Boat)
#     Mahambo to Antanarivo (450km 10h)
# May 16: 
#     Antananrivo to Andasibe (150km 4-6h)
# May 17: 
#     Hike Andasibe
# May 18: 
#     Andasibe to Antanarivo (150km 4-6h)
# May 19: 
#     2 am flight
