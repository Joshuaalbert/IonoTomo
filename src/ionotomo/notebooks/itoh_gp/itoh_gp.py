
# coding: utf-8

# In[1]:


import matplotlib
matplotlib.use('Agg')

import numpy as np
from scipy.cluster.vq import kmeans2 
import pylab as plt
plt.style.use('ggplot')
import astropy.units as au
import os



import gpflow as gp
from heterogp.latent import Latent
from gpflow import settings
from gpflow.decors import params_as_tensors,autoflow
from gpflow.quadrature import hermgauss
from gpflow import settings
from gpflow import transforms
from gpflow import logdensities as densities

from gpflow.decors import params_as_tensors
from gpflow.decors import params_as_tensors_for
from gpflow.decors import autoflow
from gpflow.params import Parameter
from gpflow.params import Parameterized
from gpflow.params import ParamList
from gpflow.quadrature import hermgauss
from gpflow.likelihoods import Likelihood
import logging
logging.basicConfig(format='%(asctime)s %(message)s')
import tensorflow as tf
import h5py


# In[2]:


from gpflow.actions import Loop, Action
from gpflow.training import AdamOptimizer

class PrintAction(Action):
    def __init__(self, model, text):
        self.model = model
        self.text = text
        
    def run(self, ctx):
        likelihood = ctx.session.run(self.model.likelihood_tensor)
        logging.warning('{}: iteration {} likelihood {:.4f}'.format(self.text, ctx.iteration, likelihood))
#         logging.warning(self.model)

class SendSummary(Action):
    def __init__(self, model, writer):
        self.model = model
        self.writer = writer
        self.summary = tf.summary.merge_all()
        
    def run(self, ctx):
        summary = ctx.session.run(self.summary)
        self.writer.add_summary(summary,global_step=ctx.iteration)

        
from gpflow.training import NatGradOptimizer, AdamOptimizer, XiSqrtMeanVar

def run_with_adam_and_nat(model, lr,iterations, callback=None, gamma = 0.001):
    if gamma == 0:
        adam = AdamOptimizer(lr).make_optimize_action(model)
        actions = [adam]
        actions = actions if callback is None else actions + [callback]

        Loop(actions, stop=iterations)()
        model.anchor(model.enquire_session())
        return
    
        
    
    var_list = [(model.f_latent.q_mu, model.f_latent.q_sqrt)]

    # we don't want adam optimizing these
    model.f_latent.q_mu.set_trainable(False)
    model.f_latent.q_sqrt.set_trainable(False)

    adam = AdamOptimizer(lr).make_optimize_action(model)
    natgrad = NatGradOptimizer(gamma).make_optimize_action(model, var_list=var_list)
    
    actions = [adam, natgrad]
    actions = actions if callback is None else actions + [callback]

    Loop(actions, stop=iterations)()
    model.anchor(model.enquire_session())

    
    


# # Create input data

# In[3]:


from ionotomo import *
from ionotomo.astro.real_data import phase_screen_datapack




def make_coord_array(*X):
    """
    Return the design matrix from coordinates.
    """
    def add_dims(x,where,sizes):
        shape = []
        tiles = []
        for i in range(len(sizes)):
            if i not in where:
                shape.append(1)
                tiles.append(sizes[i])
            else:
                shape.append(-1)
                tiles.append(1)
        return np.tile(np.reshape(x,shape),tiles)
    N = [x.shape[0] for x in X]
    X_ = []

    for i,x in enumerate(X):
        for dim in range(x.shape[1]):
            X_.append(add_dims(x[:,dim],[i], N))
    X = np.stack(X_,axis=-1)
    
    return np.reshape(X,(-1,X.shape[-1]))

def make_data_vec(Y,freqs):
    """
    Takes Y of shape [..., Nf, N]
    returns [...,N+1] where last is freq of observation"""
    shape = Y.shape
    for _ in range(len(shape)-2):
        freqs = freqs[None,...]
    freqs = freqs[...,None]
    # freqs is [1,1,...,Nf,1]
    tiles = list(shape)
    tiles[-1] = 1
    tiles[-2] = 1
    freqs = np.tile(freqs,tiles)
    # ..., N+1
    return np.concatenate([Y, freqs],axis=-1)



# # Decide on some priors

# In[4]:


from scipy.optimize import fmin,minimize

def log_normal_solve(mode,uncert):
    def func(x):
        mu,sigma2 = x
        mode_ = np.exp(mu-sigma2)
        var_ = (np.exp(sigma2 ) - 1) * np.exp(2*mu + sigma2)
        return (mode_ - mode)**2 + (var_ - uncert**2)**2
    res = minimize(func,(mode,uncert**2))
#         res = fmin(func, (mode,uncert**2))
    return res.x[0],np.sqrt(res.x[1])


# In[ ]:






# # Direction

# In[11]:


try:
    @tf.RegisterGradient('WrapGrad')
    def _wrap_grad(op,grad):
        phi = op.inputs[0]
        return tf.ones_like(phi)*grad
except:
    pass#already defined

def wrap(phi):
    out = tf.atan2(tf.sin(phi),tf.cos(phi))
    with tf.get_default_graph().gradient_override_map({'Identity': 'WrapGrad'}):
        return tf.identity(out)
    
from heterogp.likelihoods import HeteroscedasticLikelihood
float_type = settings.float_type
class HeteroWrappedPhaseGaussian(HeteroscedasticLikelihood):
    def __init__(self, log_noise_latent, tec_scale=0.01, freq=140e6, name=None):
        super().__init__(log_noise_latent, name=name)
        self.variance = gp.params.Parameter(
            1.0, transform=gp.transforms.positive, dtype=gp.settings.float_type)
        self.tec_scale = tec_scale
        self.num_gauss_hermite_points = 20
        self.freq = tf.convert_to_tensor(freq,dtype=settings.float_type,name='test_freq') # frequency the phase is calculated at for the predictive distribution
        self.tec_conversion = tf.convert_to_tensor(tec_scale * -8.4480e9,dtype=settings.float_type,name='tec_conversion') # rad Hz/ tecu
        self.tec2phase = tf.convert_to_tensor(self.tec_conversion / self.freq,dtype=settings.float_type,name='tec2phase')
    
    
    @params_as_tensors
    def logp(self, F, Y, freqs=None,hetero_variance=None,**unused_kwargs):
        """The log-likelihood function."""
        tec2phase = self.tec_conversion/freqs
        phase = wrap(F*tec2phase)
        dphase = wrap(phase) - wrap(Y) # Ito theorem
        
        arg = tf.stack([-0.5*tf.square(dphase + 2*np.pi*k)/hetero_variance - 0.5 * tf.log((2*np.pi) * hetero_variance)                     for k in range(-2,3,1)],axis=-1)
        return tf.reduce_logsumexp(arg,axis=-1)
        
#         dphase = wrap(wrap(phase) - wrap(Y)) # Ito theorem
#         return densities.gaussian(dphase, tf.fill(tf.shape(F),tf.cast(0.,settings.float_type)), hetero_variance)

    @params_as_tensors
    def conditional_mean(self, F, eval_freq=None,hetero_variance=None, **unused_kwargs):  # pylint: disable=R0201
        """The mean of the likelihood conditioned on latent."""
        eval_freq = self.freq if eval_freq is None else eval_freq
        tec2phase = self.tec_conversion/eval_freq
        phase = F*tec2phase
        return phase

    @params_as_tensors
    def conditional_variance(self, F,hetero_variance=None, **unused_kwargs):
        return hetero_variance  
    
    @params_as_tensors
    def hetero_noise(self,X):
        """
        Calculates the heterscedastic variance at points X.
        X must be of shape [S, N, D]
        Returns [S,N,num_latent]
        """
        return tf.fill(tf.shape(X[:,:,:1]),tf.cast(self.variance,settings.float_type)) 
        log_noise,_,_ = self.log_noise_latent.sample_from_conditional(X,full_cov=True)
        hetero_noise = tf.exp(log_noise)
        hetero_noise = tf.where(hetero_noise < self.min_noise, 
                                tf.fill(tf.shape(hetero_noise), self.min_noise), 
                                hetero_noise)
        return hetero_noise
    
def weights_and_mean_uncert(phase,N=200):
    def w(x):
        return np.arctan(np.sin(x),np.cos(x))
    weights = []
    for k in range(phase.shape[1]):
        dphase = phase[:,k]
        dphase = w(w(dphase[:-1]) - w(dphase[1:]))
        dphase = np.pad(dphase,(0,N),mode='symmetric')
        uncert = np.sqrt(np.convolve(dphase**2, np.ones((N,))/N, mode='valid',))
        weights.append(uncert)
    weights = np.stack(weights,axis=-1)#uncert
    mean_uncert = max(1e-3,np.mean(weights))
    weights = 1./weights**2
    weights /= np.mean(weights)
    weights[np.isnan(weights)] = 1.
    return weights, mean_uncert
    
        
from heterogp.hgp import HGP

class WrappedPhaseHGP(HGP):
    def __init__(self,X, Y, Z, kern, likelihood, 
                 mean_function=gp.mean_functions.Zero, 
                 minibatch_size=None,
                 num_latent = None, 
                 num_samples=1,
                 num_data=None,
                 whiten=True):
        super(WrappedPhaseHGP,self).__init__(X, Y, Z, kern, likelihood, 
                 mean_function=mean_function, 
                 minibatch_size=minibatch_size,
                 num_latent = num_latent, 
                 num_samples=num_samples,
                 num_data=num_data,
                 whiten=whiten)

    def E_log_p_Y(self, X, Y):
        """
        Calculate the expectation of the data log likelihood under the variational distribution
         with MC samples
        """
        X = tf.tile(X[None,:,:],[self.num_samples,1,1])
        Fmean, Fvar = self._build_predict(X, full_cov=False, S=None)
#         f = self.f_latent.sample_from_conditional(X, z=None, full_cov=False)
        hetero_variance = tf.square(self.likelihood.hetero_noise(X))
        lik_freqs = Y[:,-1:]
        weights = Y[:,-2:-1]
#         ###
#         # could do the reparametrization trick f ~ f_latent.predict_sample(
#         logp = self.likelihood.logp(f,self.Y[:,:-1],lik_freqs,hetero_variance)
#         var_exp = tf.reduce_mean(logp,axis=0)
#         return var_exp
        var_exp = self.likelihood.variational_expectations(Fmean, Fvar, Y[:,:-2], freqs = lik_freqs, hetero_variance=hetero_variance)  # S, N, D
        return tf.reduce_mean(var_exp, 0)*weights  # N, D

    @params_as_tensors
    def KL_tensors(self):
        KL = [self.f_latent.KL()]
        if hasattr(self.likelihood,'log_noise_latent'):
            KL.append(self.likelihood.log_noise_latent.KL())
        if hasattr(self.f_latent.kern,'log_ls_latent'):
            KL.append(self.f_latent.kern.log_ls_latent.KL())
        if hasattr(self.f_latent.kern,'log_sigma_latent'):
            KL.append(self.f_latent.kern.log_sigma_latent.KL())
        return KL  

    @autoflow((float_type, [None, None]), (tf.int32, []))
    def predict_y(self, Xnew, num_samples):
        """
        Draws the predictive mean and variance at the points `X`
        num_samples times.
        X should be [N,D] and this returns [S,N,num_latent], [S,N,num_latent]
        """
        Xnew = tf.tile(Xnew[None,:,:],[num_samples,1,1])
        Fmean, Fvar = self._build_predict(Xnew, full_cov=False, S=None)
        hetero_variance = tf.square(self.likelihood.hetero_noise(Xnew))
        return self.likelihood.predict_mean_and_var(Fmean, Fvar, hetero_variance=hetero_variance)
    
    @autoflow((float_type, [None, None]), (tf.int32, []))
    def predict_dtec(self, Xnew, num_samples):
        """
        Draws the predictive mean and variance at the points `X`
        num_samples times.
        X should be [N,D] and this returns [S,N,num_latent], [S,N,num_latent]
        """
        Xnew = tf.tile(Xnew[None,:,:],[num_samples,1,1])
        mean, var = self._build_predict(Xnew, full_cov=False, S=None)
        return mean*self.likelihood.tec_scale, var*self.likelihood.tec_scale**2

    @autoflow((float_type, [None, None]), (float_type, [None, None]), (tf.int32, []))
    def predict_density(self, Xnew, Ynew, num_samples):
        Xnew = tf.tile(Xnew[None,:,:],[num_samples,1,1])
        Fmean, Fvar = self._build_predict(Xnew, full_cov=False, S=None)
        hetero_variance = tf.square(self.likelihood.hetero_variance(Xnew))
        lik_freqs = Ynew[:,-1:]
        l = self.likelihood.predict_density(Fmean, Fvar, Ynew[:,:-2], freqs=lik_freqs, hetero_variance=hetero_variance)
        log_num_samples = tf.log(tf.cast(num_samples, float_type))
        return tf.reduce_logsumexp(l - log_num_samples, axis=0)

def gamma_prior(mode,std):
    a = std/mode#sqrt(k)/(k-1)
    shape = (2* a**2 + np.sqrt((4 * a**2 + 1)/a**4) * a**2 + 1)/(2 *a**2)
    scale = std/np.sqrt(shape)
    return gp.priors.Gamma(shape,scale)

from heterogp.latent import Latent


def w(x):
    return np.arctan2(np.sin(x),np.cos(x))

from ionotomo.plotting.plot_datapack import DatapackPlotter,animate_datapack

def run_solve(flags):
    os.environ["KMP_BLOCKTIME"] = "1"
    os.environ["KMP_SETTINGS"] = "1"
    os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"
    if flags.intra_op_threads > 0:
        os.environ["OMP_NUM_THREADS"]= str(flags.intra_op_threads)
    
    
    dp = DataPack(filename='../../data/rvw_datapack_full_phase_dec27_wideband.hdf5')


    ant_idx = -1
    times_,_ = dp.get_times(-1)
    Nt_ = len(times_)
    end_time = min(flags.end_time, len(times_))
    time_idx = range(flags.start_time,end_time)
    dir_idx = -1
    freq_idx = -1#range(4,20)
    
    
    
    

    phase = dp.get_phase(ant_idx,time_idx,dir_idx,freq_idx)

    times,_ = dp.get_times(time_idx)
    antennas,antenna_labels = dp.get_antennas(ant_idx)
    freqs = dp.get_freqs(freq_idx) 
    directions,patch_names = dp.get_directions(dir_idx)
    Na,Nt,Nd,Nf = phase.shape


    X_d = np.array([directions.ra.deg,directions.dec.deg]).T
    X_t = times.mjd[:,None]*86400.#mjs
    enu = ENU(obstime=times[0],location=dp.radio_array.get_center())
    ant_enu = antennas.transform_to(enu)
    X_a = np.array([ant_enu.east.to(au.km).value, ant_enu.north.to(au.km).value]).T

    d_std = X_d.std(0).mean() + 1e-6
    t_std = X_t.std() + 1e-6
    a_std = X_a.std(0).mean() + 1e-6

    X_a = (X_a - X_a.mean(0)) / a_std
    X_t = (X_t - X_t.mean()) / t_std
    d_mean = X_d.mean(0)
    X_d = (X_d - X_d.mean(0)) / d_std

    #~8 arcmin resolution
    phase_screen_dp = phase_screen_datapack(30,ant_idx=ant_idx,time_idx=time_idx,freq_idx=freq_idx,datapack=dp)
    directions_,patch_names_ = phase_screen_dp.get_directions(-1)
    Nd_ = len(directions_)
    X_d_ = np.array([directions_.ra.deg,directions_.dec.deg]).T
    X_d_ = (X_d_ - d_mean) / d_std
    
    if not os.path.exists(flags.solution_file):
        with h5py.File(flags.solution_file,'a') as f:
            f['dtec'] = np.zeros([Na,Nt_,Nd_])
            f['dtec_facets'] = np.zeros([Na,Nt_,Nd])
            f['ra'] = directions_.ra.deg
            f['dec'] = directions_.dec.deg
            f['dtec_variance'] = np.zeros([Na,Nt_,Nd_])
            f['dtec_facets_variance'] = np.zeros([Na,Nt_,Nd])
            f['count_facets'] = np.zeros([Na,Nt_,Nd])
            f['count'] = np.zeros([Na,Nt_,Nd_])
            f['ra_facets'] = directions.ra.deg
            f['dec_facets'] = directions.dec.deg
            f['time'] = times_.mjd*86400.     

    
    def make_hetero_model(X,Y,freqs,M=None,minibatch_size=None,Z = None, eval_freq=140e6):
        N, num_latent = Y.shape
        _, D = X.shape
        M = M or N
        if Z is None:
            Z = kmeans2(X, M, minit='points')[0] if N < 10000 else X[np.random.choice(N,size=M,replace=False),:]
        with gp.defer_build():

            log_noise_mean_func = gp.mean_functions.Constant(log_noise_mean[0])
            log_noise_mean_func.c.set_trainable(False)
    #         log_noise_mean_func.c.prior = gp.priors.Gaussian(log_noise_mean[0],log_noise_mean[1]**2)
            log_noise_kern = gp.kernels.RBF(3,variance=0.01**2)#log_noise_kern_var[0])
            log_noise_kern.variance.set_trainable(False)# = gamma_prior(0.05,0.05)
    #         log_noise_kern.variance.prior = gp.priors.LogNormal(log_noise_kern_var[0],log_noise_kern_var[1]**2)

            log_noise_Z = Z#X[np.random.choice(N,size=42*9,replace=False),:]
            log_noise_latent = Latent(log_noise_Z, 
                                      log_noise_mean_func, log_noise_kern, num_latent=1, whiten=False, name=None)
    #         log_noise_latent.feature.set_trainable(False)

            # Define the likelihood
            likelihood = HeteroWrappedPhaseGaussian(log_noise_latent,freq=eval_freq,tec_scale = flags.tec_scale)
            likelihood.variance = np.exp(lik_var[0])
            likelihood.variance.prior = gp.priors.LogNormal(lik_var[0],lik_var[1]**2)
            likelihood.variance.set_trainable(True)

            kern_time = gp.kernels.Matern52(1,active_dims=[0])
            kern_time.lengthscales = np.exp(tec_kern_time_ls[0])
            kern_time.lengthscales.set_trainable(True)
            kern_time.lengthscales.prior = gp.priors.LogNormal(tec_kern_time_ls[0],tec_kern_time_ls[1]**2)#gamma_prior(70./t_std, 50./t_std)
            kern_time.variance = np.exp(tec_kern_var[0])
            kern_time.variance.set_trainable(True)
            kern_time.variance.prior = gp.priors.LogNormal(tec_kern_var[0],tec_kern_var[1]**2)#gamma_prior(0.001, 0.005)

            kern_space = gp.kernels.Matern52(2,active_dims=[1,2],variance=1.)
            kern_space.variance.set_trainable(False)
            kern_space.lengthscales = np.exp(tec_kern_dir_ls[0])
            kern_space.lengthscales.set_trainable(True)
            kern_space.lengthscales.prior = gp.priors.LogNormal(tec_kern_dir_ls[0],tec_kern_dir_ls[1]**2)#gamma_prior(0.3/d_std,0.2/d_std)

            white = gp.kernels.White(3)
            white.variance = 0.0005**2/flags.tec_scale**2
            white.variance.set_trainable(False)
            kern = kern_time*kern_space + white

            mean = gp.mean_functions.Constant(0.)#tec_mean_mu)
            mean.c.set_trainable(False)
            mean.c.prior = gp.priors.Gaussian(tec_mean_mu,tec_mean_var)


            model = WrappedPhaseHGP(X, Y, Z, kern, likelihood, 
                     mean_function=mean, 
                     minibatch_size=minibatch_size,
                     num_latent = num_latent-2, 
                     num_samples=1,
                     num_data=N,
                     whiten=False)
            model.f_latent.feature.set_trainable(True)
            model.compile()
            tf.summary.scalar('likelihood',-model.likelihood_tensor)
        return model



    gp.settings.numerics.jitter = flags.jitter
    iterations = flags.iterations
    learning_rate = flags.learning_rate
    minibatch_size = flags.minibatch_size
    i = flags.antenna
    freq_l = np.argmin((freqs - flags.eval_freq)**2)

    X = make_coord_array(X_t,X_d,freqs[:,None])[:,:-1]# N, 3
    M = flags.inducing
    if M is None:
        Z = make_coord_array(X_t[::flags.time_skip,:],X_d[::1,:])
    else:
        Z = None
        if M > 1.0:
            M = int(M)
        else:
            M = int(M*X_t.shape[0]*X_d.shape[0])
        assert M > 0, "Need at least one inducing point"
    
    ###
    # get stat weights of data-points
    weights, uncert_mean = weights_and_mean_uncert(phase[i,:,:,0],N=200)
    # Nt, Nd, Nf
    weights = np.tile(weights[:,:,None],(1,1,Nf))
    # Nt, Nd, Nf, 2
    data_vec = np.stack([w(phase[i,:,:,:]), weights], axis=-1)
    # Nt, Nd, Nf, 3
    
    Y = make_data_vec(data_vec,freqs)#N2
    Y = Y.reshape((-1, Y.shape[-1]))
    y_mean = Y[:,:-2].mean()*0.
    Y[:,:-2] -= y_mean
    
    ###
    # Using half-normal priors for positive params these should represent limits of distriubtion
    log_noise_mean = log_normal_solve(0.35, 0.65)
    log_noise_kern_var = log_normal_solve(log_normal_solve(0.25, 0.25)[1]**2, np.abs(log_normal_solve(0.25, 0.1)[1]**2 - log_normal_solve(0.25, 0.25)[1]**2))

    lik_var = log_normal_solve(uncert_mean, uncert_mean*0.25)
    
    tec_mean_mu, tec_mean_var = 0./flags.tec_scale, (0.005)**2/flags.tec_scale**2

    tec_kern_time_ls = log_normal_solve(50./t_std, 20./t_std)

    tec_kern_dir_ls = log_normal_solve(0.5/d_std, 0.3/d_std)

    tec_kern_sigma = 0.005/flags.tec_scale
    tec_kern_var = log_normal_solve(tec_kern_sigma**2,0.1*tec_kern_sigma**2)

#     print("Log_noise mean Gaussian",log_noise_mean,'median (rad)',np.exp(log_noise_mean[0]))
#     print("Log_noise kern var logGaussian",log_noise_kern_var,'median (log-rad)',np.sqrt(np.exp(log_noise_kern_var[0])))
    print("likelihood var logGaussian",lik_var,'median (rad)',np.exp(lik_var[0]))
    print("tec mean Gaussian",tec_mean_mu*flags.tec_scale, tec_mean_var*flags.tec_scale**2)
    print("tec kern var logGaussian",tec_kern_var,'median (tec)',np.sqrt(np.exp(tec_kern_var[0]))*flags.tec_scale)
    print("tec kern time ls logGaussian",tec_kern_time_ls,'median (sec)',np.exp(tec_kern_time_ls[0])*t_std)
    print("tec kern dir ls logGaussian",tec_kern_dir_ls,'median (deg)',np.exp(tec_kern_dir_ls[0])*d_std)

    tf.reset_default_graph()
    graph=tf.Graph()
    config = tf.ConfigProto()
    config.intra_op_parallelism_threads = flags.intra_op_threads
    config.inter_op_parallelism_threads = flags.inter_op_threads
    sess = tf.Session(graph=graph,config=config)


    try:
        os.makedirs('summaries')
    except:
        pass
    
    import glob

    run_id = len(glob.glob('summaries/summary_{}_*'.format(antenna_labels[i])))
    
    
    with graph.as_default(), sess.as_default(), tf.summary.FileWriter('summaries/summary_{}_{}'.format(antenna_labels[i],run_id), graph) as writer:
        model = make_hetero_model(X,Y,freqs,M=M,minibatch_size=minibatch_size, Z=Z, eval_freq=freqs[freq_l])

        run_with_adam_and_nat(model,learning_rate,iterations,SendSummary(model,writer), gamma = 0.000)

    #     run_with_adam_and_nat(model,1e-3,iterations,SendSummary(model,writer), gamma = 0.0001)
        if False:
            Xstar = make_coord_array(X_t,X_d_,freqs[:1,None])[:,:-1]

            ystar,varstar = model.predict_y(Xstar,10)#at 140MHz
            y_star = ystar.mean(0).reshape([Nt,Nd_])
            y_star += y_mean
            varstar = varstar.mean(0).reshape([Nt,Nd_])

            dtec_ystar,dtec_varstar = model.predict_dtec(Xstar,10)
            dtec_ystar = dtec_ystar.mean(0).reshape([Nt,Nd_])
            dtec_varstar = dtec_varstar.mean(0).reshape([Nt,Nd_])

            hetero_noise = model.likelihood.compute_hetero_noise(Xstar,10)
            hetero_noise = hetero_noise.mean(0).reshape([Nt,Nd_])
        else: 
            y_star,varstar,dtec_ystar, dtec_varstar, hetero_noise = [],[],[],[],[]
            for k in range(Nd_):
                Xstar = make_coord_array(X_t,X_d_[k:k+1,:],freqs[:1,None])[:,:-1]

                ystar_,varstar_ = model.predict_y(Xstar,10)#at 140MHz
                y_star_ = ystar_.mean(0).reshape([Nt,1])
                y_star_ += y_mean
                varstar_ = varstar_.mean(0).reshape([Nt,1])

                dtec_ystar_,dtec_varstar_ = model.predict_dtec(Xstar,10)
                dtec_ystar_ = dtec_ystar_.mean(0).reshape([Nt,1])
                dtec_varstar_ = dtec_varstar_.mean(0).reshape([Nt,1])

                hetero_noise_ = model.likelihood.compute_hetero_noise(Xstar,10)
                hetero_noise_ = hetero_noise_.mean(0).reshape([Nt,1])
                y_star.append(y_star_)
                varstar.append(varstar_)
                dtec_ystar.append(dtec_ystar_)
                dtec_varstar.append(dtec_varstar_)
                hetero_noise.append(hetero_noise_)
            y_star = np.concatenate(y_star,axis=-1)
            varstar = np.concatenate(varstar,axis=-1)
            dtec_ystar = np.concatenate(dtec_ystar,axis=-1)
            dtec_varstar = np.concatenate(dtec_varstar,axis=-1)
            hetero_noise = np.concatenate(hetero_noise,axis=-1)
            
            
    if flags.plot:
        try:
            os.makedirs('{}/{}/time_diagnostics'.format(flags.plot_dir,antenna_labels[i]))
            os.makedirs('{}/{}/inferred_phase_diff'.format(flags.plot_dir,antenna_labels[i]))
            os.makedirs('{}/{}/inferred_phase'.format(flags.plot_dir,antenna_labels[i]))
            os.makedirs('{}/{}/inferred_phase_faceted'.format(flags.plot_dir,antenna_labels[i]))
            os.makedirs('{}/{}/observed'.format(flags.plot_dir,antenna_labels[i]))
            os.makedirs('{}/{}/hetero_noise'.format(flags.plot_dir,antenna_labels[i]))
            os.makedirs('{}/{}/inferred_phase_error'.format(flags.plot_dir,antenna_labels[i]))
            os.makedirs('{}/{}/inferred_dtec'.format(flags.plot_dir,antenna_labels[i]))
            os.makedirs('{}/{}/inferred_dtec_error'.format(flags.plot_dir,antenna_labels[i]))
        except:
            pass
        
#         with h5py.File('{}/{}/solution.hdf5'.format(flags.plot_dir,antenna_labels[i]),'a') as f:
#             if 'tec' not in f.keys():
#                 f['tec'] = np.zeros((Na,Nt,Nd_),dtype=np.float64)
#             if 'tec_var' not in f.keys():
#                 f['tec_var'] = np.zeros((Na,Nt,Nd_),dtype=np.float64)
#             f['tec'][i,time_idx,:] = dtec_ystar
#             f['tec_var'][i,time_idx,:] = dtec_varstar

        dataplotter = DatapackPlotter(dp)
        fignames = []
        for t in time_idx:
            fignames.append('{}/{}/observed/fig{:04d}.png'.format(flags.plot_dir,antenna_labels[i],t))
        dataplotter.plot(ant_idx=[i], time_idx = time_idx, dir_idx=-1, 
                                 freq_idx=[freq_l], vmin=None,vmax=None,mode='perantenna',observable='phase',fignames=fignames,
                                 phase_wrap=True, plot_crosses=True,plot_facet_idx=False,plot_patchnames=False,
                                 labels_in_radec=True,show=False)

        plt.close('all')

        fignames = []
        for t in time_idx:
            fignames.append('{}/{}/inferred_phase_diff/fig{:04d}.png'.format(flags.plot_dir,antenna_labels[i],t))
        with graph.as_default(), sess.as_default():
            Xstar = make_coord_array(X_t,X_d,freqs[:1,None])[:,:-1]
            ystar,_ = model.predict_y(Xstar,10)#at 140MHz
            ystar = ystar.mean(0).reshape([Nt,Nd])
            ystar += y_mean
            
            dtec_facet_ystar, dtec_facet_varstar =  model.predict_dtec(Xstar,10)#at 140MHz
            dtec_facet_ystar = dtec_facet_ystar.mean(0).reshape([Nt,Nd])
            dtec_facet_varstar = dtec_facet_varstar.mean(0).reshape([Nt,Nd])
            
            
        with h5py.File(flags.solution_file,'a') as f:
            f['dtec'][i,time_idx,:] += dtec_ystar/dtec_varstar
            f['dtec_variance'][i,time_idx,:] += dtec_varstar/dtec_varstar
            f['count'][i,time_idx,:] += 1./dtec_varstar
            
            f['dtec_facets'][i,time_idx,:] += dtec_facet_ystar/dtec_facet_varstar
            f['dtec_facets_variance'][i,time_idx,:] += dtec_facet_varstar/dtec_facet_varstar
            f['count_facets'][i,time_idx,:] += 1./dtec_facet_varstar
            
            
        
        
        for k in range(Nd):
            
            plt.plot(X_t[:,0], w(dp.phase[i,time_idx,k,freq_l]),label='data')
            plt.plot(X_t[:,0], ystar[:,k],label='inferred')
            plt.legend()
            plt.savefig('{}/{}/time_diagnostics/direction_{:02d}_{:04d}_{:04d}.png'.format(flags.plot_dir,antenna_labels[i],k,flags.start_time,flags.end_time))
            plt.close('all')


        dp.phase[i,time_idx,:,freq_l] = w(w(dp.phase[i,time_idx,:,freq_l]) - w(ystar))
        dataplotter.plot(ant_idx=[i], time_idx = time_idx, dir_idx=-1, 
                                 freq_idx=[freq_l], vmin=None,vmax=None,mode='perantenna',observable='phase',fignames=fignames,
                                 phase_wrap=True, plot_crosses=False,plot_facet_idx=False,plot_patchnames=False,
                                 labels_in_radec=True,show=False)
        plt.close('all')

        fignames = []
        for t in time_idx:
            fignames.append('{}/{}/inferred_phase_faceted/fig{:04d}.png'.format(flags.plot_dir,antenna_labels[i],t))
        
        dp.phase[i,time_idx,:,freq_l] = ystar
        dataplotter.plot(ant_idx=[i], time_idx = time_idx, dir_idx=-1, 
                                 freq_idx=[freq_l], vmin=None,vmax=None,mode='perantenna',observable='phase',fignames=fignames,
                                 phase_wrap=True, plot_crosses=False,plot_facet_idx=False,plot_patchnames=False,
                                 labels_in_radec=True,show=False)

        plt.close('all')
        
        

        phase_screen_dp.set_reference_antenna(dp.ref_ant)
        dataplotter = DatapackPlotter(phase_screen_dp)
        fignames = []
        for t in time_idx:
            fignames.append('{}/{}/inferred_phase/fig{:04d}.png'.format(flags.plot_dir,antenna_labels[i],t))
        phase_screen_dp.phase[i,:,:,freq_l] = y_star
        dataplotter.plot(ant_idx=[i], time_idx = -1, dir_idx=-1, 
                                 freq_idx=[freq_l], vmin=None,vmax=None,mode='perantenna',observable='phase',fignames=fignames,
                                 phase_wrap=True, plot_crosses=False,plot_facet_idx=False,plot_patchnames=False,
                                 labels_in_radec=True,show=False)

        plt.close('all')

        fignames = []
        for t in time_idx:
            fignames.append('{}/{}/hetero_noise/fig{:04d}.png'.format(flags.plot_dir,antenna_labels[i],t))
        phase_screen_dp.variance[i,:,:,freq_l] = hetero_noise**2
        dataplotter.plot(ant_idx=[i], time_idx = -1, dir_idx=-1, 
                                 freq_idx=[freq_l], vmin=None,vmax=None,mode='perantenna',observable='std',fignames=fignames,
                                 phase_wrap=False, plot_crosses=False,plot_facet_idx=False,plot_patchnames=False,
                                 labels_in_radec=True,show=False)

        plt.close('all')

        fignames = []
        for t in time_idx:
            fignames.append('{}/{}/inferred_phase_error/fig{:04d}.png'.format(flags.plot_dir,antenna_labels[i],t))
        phase_screen_dp.variance[i,:,:,freq_l] = varstar
        dataplotter.plot(ant_idx=[i], time_idx = -1, dir_idx=-1, 
                                 freq_idx=[freq_l], vmin=None,vmax=None,mode='perantenna',observable='std',fignames=fignames,
                                 phase_wrap=False, plot_crosses=False,plot_facet_idx=False,plot_patchnames=False,
                                 labels_in_radec=True,show=False)

        plt.close('all')

        fignames = []
        for t in time_idx:
            fignames.append('{}/{}/inferred_dtec/fig{:04d}.png'.format(flags.plot_dir,antenna_labels[i],t))
        phase_screen_dp.phase[i,:,:,freq_l] = dtec_ystar
        dataplotter.plot(ant_idx=[i], time_idx = -1, dir_idx=-1, 
                                 freq_idx=[freq_l], vmin=None,vmax=None,mode='perantenna',observable='phase',fignames=fignames,
                                 phase_wrap=False, plot_crosses=False,plot_facet_idx=False,plot_patchnames=False,
                                 labels_in_radec=True,show=False)

        plt.close('all')

        fignames = []
        for t in time_idx:
            fignames.append('{}/{}/inferred_dtec_error/fig{:04d}.png'.format(flags.plot_dir,antenna_labels[i],t))
        phase_screen_dp.variance[i,:,:,freq_l] = dtec_varstar
        dataplotter.plot(ant_idx=[i], time_idx = -1, dir_idx=-1, 
                                 freq_idx=[freq_l], vmin=None,vmax=None,mode='perantenna',observable='std',fignames=fignames,
                                 phase_wrap=False, plot_crosses=False,plot_facet_idx=False,plot_patchnames=False,
                                 labels_in_radec=True,show=False)

        plt.close('all')


# In[14]:


import argparse

def add_args(parser):
    antenna_labels = ['CS001HBA0', 'CS001HBA1', 'CS002HBA0', 'CS002HBA1', 'CS003HBA0',
       'CS003HBA1', 'CS004HBA0', 'CS004HBA1', 'CS005HBA0', 'CS005HBA1',
       'CS006HBA0', 'CS006HBA1', 'CS007HBA0', 'CS007HBA1', 'CS011HBA0',
       'CS011HBA1', 'CS013HBA0', 'CS013HBA1', 'CS017HBA0', 'CS017HBA1',
       'CS021HBA0', 'CS021HBA1', 'CS024HBA0', 'CS024HBA1', 'CS026HBA0',
       'CS026HBA1', 'CS028HBA0', 'CS028HBA1', 'CS030HBA0', 'CS030HBA1',
       'CS031HBA0', 'CS031HBA1', 'CS032HBA0', 'CS032HBA1', 'CS101HBA0',
       'CS101HBA1', 'CS103HBA0', 'CS103HBA1', 'CS201HBA0', 'CS201HBA1',
       'CS301HBA0', 'CS301HBA1', 'CS302HBA0', 'CS302HBA1', 'CS401HBA0',
       'CS401HBA1', 'CS501HBA0', 'CS501HBA1', 'RS106HBA', 'RS205HBA',
       'RS208HBA', 'RS210HBA', 'RS305HBA', 'RS306HBA', 'RS307HBA',
       'RS310HBA', 'RS406HBA', 'RS407HBA', 'RS409HBA', 'RS503HBA',
       'RS508HBA', 'RS509HBA']
    def _antenna_type(s):
        try:
            idx = int(s)
            assert idx < len(antenna_labels)
            return idx
        except:
            idx = None
            for idx in range(len(antenna_labels)):
                if antenna_labels[idx].lower() == s.lower():
                    return idx
        raise ValueError("{} invalid antenna".format(s))
    def _inducing(s):
        if s.lower().strip() == 'none':
            return None
        else:
            return float(s)
            
        
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.register("type", "antenna", _antenna_type)
    parser.register('type','inducing',_inducing)

    # network
    parser.add_argument("--antenna", type="antenna", default=51, 
                        help="""The index or name of antenna.\n{}""".format(list(zip(range(len(antenna_labels)),antenna_labels))))
    parser.add_argument("--start_time", type=int, default=0,
                      help="Start time index")
    parser.add_argument("--end_time", type=int, default=20,
                      help="End time index")
    parser.add_argument("--time_skip", type=int, default=2,
                      help="Time skip")
    parser.add_argument("--inducing", type='inducing', default='None',
                       help="""The number of inducing point if > 1.0 else the fraction of total number of points. If None then use time_skip instead.""")
    parser.add_argument("--minibatch_size", type=int, default=256,
                      help="Size of minibatch")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                      help="learning rate")
    parser.add_argument("--plot", type="bool", default=True, const=True,nargs='?',
                      help="Whether to plot results")
    parser.add_argument("--plot_dir", type=str, default='./figs', 
                      help="Where to plot results")
    parser.add_argument("--iterations", type=int, default=10000, 
                      help="How many iterations to run")
    parser.add_argument("--jitter", type=float, default=1e-6, 
                      help="Jitter for stability")
    parser.add_argument("--eval_freq", type=float, default=144e6, 
                      help="Eval frequency")
    parser.add_argument("--inter_op_threads", type=int, default=0,
                       help="""The max number of concurrent threads""")
    parser.add_argument("--intra_op_threads", type=int, default=0,
                       help="""The number threads allowed for multi-threaded ops.""")
    parser.add_argument("--tec_scale", type=float, default=0.01,
                       help="""The relative tec scale used for scaling the GP model for computational stability.""")
    parser.add_argument("--solution_file", type=str, default='solution.hdf5',
                       help="""solution file path ending in .hdf5""")


# In[ ]:




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    flags, unparsed = parser.parse_known_args()
    print(flags)
    run_solve(flags)
#     tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

