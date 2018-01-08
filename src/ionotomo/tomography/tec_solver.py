import tensorflow as tf
import numpy as np
import pylab as plt
from ionotomo import *
import time as tm
import h5py
from ionotomo.settings import TFSettings
import ionotomo.utils.gaussian_process as gp
#import gpflow as gp

from gpflow.transforms import positive
from gpflow import priors
from gpflow.decors import params_as_tensors
from gpflow.params import Parameter
from gpflow.likelihoods import Likelihood

from ionotomo.tomography.simulate import simulate_tec
from ionotomo.tomography.pipeline import Pipeline
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


def hermgauss(n):
    x, w = np.polynomial.hermite.hermgauss(n)
    x, w = x.astype(float), w.astype(float)
    return x, w

def variational_expectations(Fmu, Fvar, Y, calc_logp, num_gauss_hermite_points = 20):
    """
    Compute the expected log density of the data, given a Gaussian
    distribution for the function values.
    if
        q(f) = N(Fmu, Fvar)
    and this object represents
        p(y|f)
    then this method computes
       \int (\log p(y|f)) q(f) df.
    Here, we implement a default Gauss-Hermite quadrature routine, but some
    likelihoods (Gaussian, Poisson) will implement specific cases.
    calc_logp : callable
        Computes the log-likelihood \log p(y|f)
    """

    gh_x, gh_w = hermgauss(num_gauss_hermite_points)
    gh_x = gh_x.reshape(1, -1)
    gh_w = gh_w.reshape(-1, 1) / np.sqrt(np.pi)
    shape = tf.shape(Fmu)
    Fmu, Fvar, Y = [tf.reshape(e, (-1, 1)) for e in (Fmu, Fvar, Y)]
    X = gh_x * tf.sqrt(2.0 * Fvar) + Fmu
    Y = tf.tile(Y, [1, num_gauss_hermite_points])  # broadcast Y to match X

    logp = calc_logp(X, Y)
    return tf.reshape(tf.matmul(logp, gh_w), shape)

def reduce_var(x, axis=None, keep_dims=False):
    """Variance of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the variance.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the variance of elements of `x`.
    """
    m = tf.reduce_mean(x, axis=axis, keep_dims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keep_dims=keep_dims)

def reduce_std(x, axis=None, keep_dims=False):
    """Standard deviation of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: An integer, the axis to compute the standard deviation.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1. If `keepdims` is `True`,
            the reduced dimension is retained with length 1.

    # Returns
        A tensor with the standard deviation of elements of `x`.
    """
    return tf.sqrt(reduce_var(x, axis=axis, keep_dims=keep_dims))

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


def pdist(x):
    """do pdist
    x : Tensor (batch_size,num_points,ndims)"""
    #D[:,i,j] = a[:,i] a[:,i]' - a[:,i] a[:,j]' -a[:,j] a[:,i]' + a[:,j] a[:,j]'
    #       =   a[:,i,p] a[:,i,p]' - a[:,i,p] a[:,j,p]' - a[:,j,p] a[:,i,p]' + a[:,j,p] a[:,j,p]'
    # batch_size,num_points,1
    r = tf.reduce_sum(x*x,axis=-1,keep_dims=True)
    #batch_size,num_points,num_points
    A = tf.matmul(x,x,transpose_b=True)
    B = r - 2*A
    out = B + tf.transpose(r,perm=[0,2,1])
    return out

def _neg_log_mar_like(mu,y,sigma_y,K, use_cholesky = True):
    """Return log mar likelihood.
    mu : tensor (B1, ... Bb, N)
    y : tensor (... , N)
    sigma_y : tensor (..., N)
    K : tensor (..., N, N)
    """
    with tf.variable_scope("neg_log_mar_like"):
        #batch_size
        n = tf.cast(tf.shape(y)[1],TFSettings.tf_float)
        
        # batch_size, n,n
        Kf = K + tf.matrix_diag(sigma_y**2,name='var_y_diag')
        # batch_size, n, 1
        dy = (mu - y)[...,tf.newaxis]
        # batch_size, num_hp, n, n
        
        def _cho():
            # batch_size, n, n
            L = tf.cholesky(Kf,name='L')
            # batch_size, n,1
            alpha = tf.cholesky_solve(L, dy, name='alpha')
            data_fit = 0.5 * tf.reduce_sum(dy*alpha,axis=-1)[...,0]
            complexity = tf.trace(tf.log(L))
            scale = 0.5*n*np.log(2.*np.pi)
            return data_fit + complexity + scale

        def _no_cho():
            Kf = (Kf + tf.transpose(Kf,perm=[0,2,1]))/2.
            e,v = tf.self_adjoint_eig(Kf)
            e = tf.where(e > 1e-14, e, 1e-14*tf.ones_like(e))
            Kf = tf.matmul(tf.matmul(v,tf.matrix_diag(e),transpose_a=True),v)

            logdet = tf.reduce_sum(tf.where(e > 1e-14, tf.log(e), tf.zeros_like(e)),axis=-1,name='logdet')

            #batch_size, n, 1
            alpha = tf.matrix_solve(Kf,tf.expand_dims(y,-1),name='solve_alpha')
            neg_log_mar_like = (tf.reduce_sum(y*tf.squeeze(alpha,axis=2),axis=1) + logdet + n*np.log(2.*np.pi))/2.
            return neg_log_mar_like
        return _cho()
        return tf.cond(tf.constant(use_cholesky),_cho,_no_cho)


def _level2_optimize(x,y,sigma_y,K,use_cholesky,learning_rate):
    with tf.variable_scope("level2_solve"):
        optimizer = tf.train.AdamOptimizer(learning_rate)
#        neg_log_mar_like, grad =_neg_log_mar_like_and_grad(x,y,sigma_y,K,use_cholesky)
#        grad = [(tf.expand_dims(tf.expand_dims(grad[name],-1),-1),K.get_variables_()[name]) for name in grad] 
#        print(grad)
        neg_log_mar_like =_neg_log_mar_like(x,y,sigma_y,K,use_cholesky)
        out = optimizer.minimize(tf.reduce_sum(neg_log_mar_like))
        #out = optimizer.apply_gradients(grad)
        return out, neg_log_mar_like



def tec_solve(V,V_mod,freqs,ant1, ant2, time, antenna_labels, ra, dec, weight_spectrum, flags,timestep=[0], freq_step = 10, tec_prior=None, c_prior=None):

    ###
    # Filter the data
    ###

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

    ###
    # Create priors
    ###

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
        tec_prior = simulate_tec(antennas,directions,times,spacing=5.,res_n=301)
    if c_prior is None:
        c_prior = np.zeros([Na,Nt,Nd],dtype=float)
    
    ### 
    # Observing params
    ###

    phase_center = ac.SkyCoord(np.mean(directions.ra.deg)*au.deg,
            np.mean(directions.dec.deg)*au.deg,frame='icrs')
    array_center = ac.SkyCoord(np.mean(antennas.x.to(au.m).value)*au.m,
                               np.mean(antennas.y.to(au.m).value)*au.m,
                               np.mean(antennas.z.to(au.m).value)*au.m,frame='itrs')
    fixtime = times[0]
    uvw = Pointing(location = array_center.earth_location,obstime = times[0],fixtime=fixtime, phase = phase_center)
    ants_uvw = antennas.transform_to(uvw)
    dirs_uvw = directions.transform_to(uvw)

    ###
    # Bayesian coords
    ###
    
    # Nd, 2
    X = angular_space(dirs_uvw)
    X_mean, X_std = np.mean(X), np.std(X)
    X -= X_mean
    X /= X_std + 1e-15#np.where(X_std == 0, np.ones_like(X_std), X_std)
        
    ###
    # solve graph
    ###
    
    p = Pipeline()

    graph = p.get_graph()
    
    with graph.as_default():
        with tf.name_scope("tec_solver"):
            # num_baselines*Nt, Nf, 4
            _V = p.add_variable("V",init_value=V,remove_nans=True,dtype=TFSettings.tf_complex)
            # num_baselines*Nt, Nf, 4, Nd
            _V_mod = p.add_variable("V_mod",init_value=V_mod,remove_nans=True,dtype=TFSettings.tf_complex)
            # num_baselines*Nt, Nf, 4
            _weight_spectrum = p.add_variable("weight_spectrum",init_value=weight_spectrum)
            # num_baselines*Nt, Nf, 4
            _flags = p.add_variable("flags",init_value=flags)
            # num_baselines*Nt
            _ant1 = p.add_variable("ant1",init_value=ant1,dtype=TFSettings.tf_int)
            _ant2 = p.add_variable("ant2",init_value=ant2,dtype=TFSettings.tf_int)
            _time_idx = p.add_variable("time_idx",init_value=time_idx,dtype=TFSettings.tf_int)
            # Nf
            _freqs = p.add_variable("freqs",init_value=freqs)
            # Na, Nt, Nd
            #_tec = p.add_variable("tec",init_value = tec_prior,trainable=True)
            _flat_tec = p.add_variable("tec",init_value = tec_prior.flatten(),trainable=True)
            _tec = tf.reshape(_flat_tec,tec_prior.shape)
            _tec_sync = tf.placeholder(shape=None,dtype=_tec.dtype)
            # Na, Nt
            #_c = p.add_variable("c",init_value = c_prior, trainable=True)
            _flat_c = p.add_variable("c",init_value = c_prior.flatten(), trainable=True)
            _c = tf.reshape(_flat_c,c_prior.shape)
            _c_sync = tf.placeholder(shape=None,dtype=_c.dtype)
#            # Na, Nt, Nd, Nf
#            _g_i_real = p.add_variable('g_i_real',init_value=np.ones([Na,Nt,Nd,Nf],dtype=np.float64),
#                    dtype=TFSettings.tf_float, trainable=True)
#            _g_i_imag = p.add_variable('g_i_imag',init_value=np.ones([Na,Nt,Nd,Nf],dtype=np.float64),
#                    dtype=TFSettings.tf_float, trainable=True)

            tec_sync_op = tf.assign(_flat_tec,_tec_sync)
            c_sync_op = tf.assign(_flat_c,_c_sync)

            _dropout = p.add_variable("dropout",shape=(),persistent=False)
            _learning_rate = p.add_variable("learning_rate",shape=(),persistent=False)

            ###
            # optimizer
            ###   
            
            optimizer = tf.train.AdamOptimizer(_learning_rate)

            ###
            # Solve gains
            ###

            phi_ik = _c[:,:,:, tf.newaxis] - 8.4480e9/_freqs[tf.newaxis,tf.newaxis,tf.newaxis,:] * _tec[:,:,:,tf.newaxis]

            g_i_real = tf.cos(phi_ik)
            g_i_imag = tf.sin(phi_ik)
            with tf.control_dependencies([g_i_real,g_i_imag]):
                g_i = tf.complex(g_i_real,g_i_imag)
                g_j_conj = tf.conj(g_i)
                
                indices = tf.stack([_ant1, _time_idx],axis=1)
                #[num_baselines * Nt, Nf, 1, Nd]
                g_i = tf.transpose(tf.gather_nd(g_i,indices)[:,:,tf.newaxis,:],perm=[0,3,2,1])
                indices = tf.stack([_ant2, _time_idx],axis=1)
                #[num_baselines * Nt, Nf, 1, Nd]
                g_j_conj = tf.transpose(tf.gather_nd(g_j_conj,indices)[:,:,tf.newaxis,:],perm=[0,3,2,1])
                g_ij = g_i * g_j_conj

                ###
                # data
                ###

                d = tf.reduce_sum(_V_mod * g_ij, axis=-1)

                ###
                # likelihood
                ###
                
                weights = tf.where(tf.equal(_flags,1), tf.zeros_like(_flags), _weight_spectrum)
                weights = tf.nn.dropout(weights,_dropout)
                dd = _V - d
                wdd_real = tf.real(dd)
                wdd_imag = tf.imag(dd)

                neg_log_likelihood_real = tf.reduce_mean(tf.square(wdd_real)*weights)
                neg_log_likelihood_imag = tf.reduce_mean(tf.square(wdd_imag)*weights)
                _neg_log_likelihood_c = tf.reduce_mean(tf.square(_c)/2.)

                _neg_log_likelihood_data = (neg_log_likelihood_real + neg_log_likelihood_imag)/2.

                loss = _neg_log_likelihood_data + _neg_log_likelihood_c

                
                wrt_variables = [_flat_c,_flat_tec]
                grads = tf.gradients(loss, wrt_variables)

                hess = tf.hessians(loss, wrt_variables)

                def _solve(h,g):
                    diag = tf.matrix_diag_part(h)
                    return tf.where(diag > 1e-10, g/diag, tf.zeros_like(g))
 
                
                #update_directions = [(tf.reshape(tf.matrix_solve_ls(h,g[:,tf.newaxis])[:,0], t.shape), t) for h,g,t in zip(hess,grads,[_flat_c,_flat_tec])]
                update_directions = [(tf.reshape(_solve(h,g), t.shape), t) for h,g,t in zip(hess,grads,[_flat_c,_flat_tec])]


                gain_optimize_op = optimizer.apply_gradients(update_directions)#(_neg_log_likelihood_data, var_list=[_tec,_c])

            ###
            # TEC fitter
            ###
            def _wrap(a):
                return tf.cast(tf.angle(tf.exp(1j*tf.cast(a,TFSettings.tf_complex))),TFSettings.tf_float)

            with tf.control_dependencies([gain_optimize_op]):

                gain_phases = tf.atan2(g_i_imag, g_i_real)
                dgain = _wrap(gain_phases[..., 1:] - gain_phases[...,:-1])


                #[Na, Nt, Nd, Nf]
                #phi_ik = _c[:,:,:, tf.newaxis] - 8.4480e9/_freqs[tf.newaxis,tf.newaxis,tf.newaxis,:] * _tec[:,:,:,tf.newaxis]
                dphase = phi_ik[...,1:] - phi_ik[...,:-1]
                
                dphase = tf.sqrt(1. + tf.square(dgain - dphase))
                _neg_log_like_tec = tf.reduce_mean(tf.nn.dropout(dphase,_dropout))
                
                #tec_optimize_op = optimizer.minimize(_neg_log_like_tec+_neg_log_likelihood_c,
                #        var_list=[_c,_tec])

            ###
            # Bayesian smoothing part
            ###
            with tf.control_dependencies([gain_optimize_op]):

                # Nd, 2
                _X = p.add_variable("X",init_value = X)
                # Na,1,1,1
                _length_scale = p.add_variable("length_scale",init_value = 0.2*np.ones([Na,1,1,1],
                    dtype=float),trainable=True)
                # Na, 1,1
                _var_scale = p.add_variable("var_scale",init_value = 0.01*np.ones([Na,1,1], dtype=float),
                        trainable=True)
                # Na, 1, 1
                _white_var = p.add_variable("white_var",init_value = 0.001*np.ones([Na,1,1], dtype=float),
                        trainable=True)

                l_min, l_max = 0.05,1.
                _length_scale = l_min + (l_max - l_min) / (1. + tf.exp(-_length_scale))

                l_min, l_max = 0.0001, 1.
                _var_scale = l_min + (l_max - l_min) / (1. + tf.exp(-_var_scale))
                _white_var = l_min + (l_max - l_min) / (1. + tf.exp(-_white_var))

                # Define the kernel 

                # (scaled) coords
                coords = tf.tile(_X[tf.newaxis,tf.newaxis,:,:]/_length_scale, (1,Nt,1,1))

                # Na, Nt*Nd, 2
                coords = tf.reshape(coords,(Na,Nt*Nd,2))

                
                # Na, N, N
                X2 = pdist(coords)

                #Na, N, N
                kern = _var_scale * tf.exp(- 0.5 * X2) + _white_var * tf.eye(Nt*Nd,batch_shape=(Na,),
                        dtype=TFSettings.tf_float)

                #Na, Nt * Nd
                y = tf.reshape(_tec,(Na,Nt*Nd))
                sigma_y = tf.abs(y)*0.1
                sigma_y = tf.where(sigma_y < 0.01, 0.01*tf.ones_like(y), sigma_y)
                y -= tf.reduce_mean(y,axis=-1,keep_dims=True)
                y_std = reduce_std(y,axis=-1,keep_dims=True)
                y /= y_std
                sigma_y /= y_std
                y = tf.where(tf.is_nan(y),tf.zeros_like(y),y)
                sigma_y = tf.where(tf.is_nan(sigma_y),tf.zeros_like(sigma_y),sigma_y)
                
                #Na, 1
                mu = tf.zeros_like(y)#tf.reduce_mean(y,axis=1,keep_dims=True)
                
                _neg_log_mar_like_spatial = tf.reduce_mean(_neg_log_mar_like(mu,y,sigma_y,kern, use_cholesky = True))
                ###
                bayes_optimize = optimizer.minimize(_neg_log_mar_like_spatial)

    sess = tf.Session(graph=graph)
    p.initialize_graph(sess)

    epoch = 0
    max_iter = 100
    loss_array = []
    train_time = []

    ###
    # Bayes setup
    ###
    coords = np.tile(X[None,None,:,:], (Na,Nt,1,1))
    # Na, Nt*Nd, 3
    coords = np.reshape(coords,(Na,Nt*Nd,2))
    coords = [coords[i,:,:] for i in range(Na)]
    K1 = gp.SquaredExponential(2,l=0.25,sigma=0.1)
    K1.set_hyperparams_bounds([0.01,2.],name='l')
    K1.set_hyperparams_bounds([0.001,2.],name='sigma')
    K2 = gp.Diagonal(2,sigma=0.001)
    K2.set_hyperparams_bounds([0.00001,0.1],name='sigma')

    K = K1+K2

    print("Epoch | gains | c_prior")
    while epoch < max_iter:
        lr = 0.5 * np.exp(-epoch / 10) + 0.0001
        if epoch > 20:
            lr = 0.01
        if epoch > 50:
            lr = 0.001
        if epoch > 70:
            lr = 0.0008
        dp = 1. - 0.25 * np.exp(-epoch/50)

        _losses = [_neg_log_likelihood_data, _neg_log_likelihood_c]
        
        losses, _ = sess.run([_losses, gain_optimize_op], 
                feed_dict = {_dropout : dp, _learning_rate : lr})
        print("{} | {}".format(epoch, losses))
        if (epoch+1) % 30 == 0:
            tec = sess.run(_tec)
            tec_mean = np.mean(tec,axis=-1,keepdims=True)
            tec_std = np.std(tec,axis=-1,keepdims=True)
            
            tec -= tec_mean
            tec /= tec_std + 1e-10

            y = [tec[i,...].flatten() for i in range(Na)]
            sigma_y = [np.where(0.1*np.abs(y[i]) < 0.01, 0.01, 0.1*np.abs(y[i])) for i in range(Na)]
            
            tec_smooth = np.zeros([Na,Nt,Nd])
            for i in range(Na):
                K.hyperparams = gp.level2_solve(coords[i],y[i],sigma_y[i],K,n_random_start=10)
                print(K)
                ystar,covstar, lml = gp.level1_solve(coords[i],y[i],sigma_y[i],coords[i],K)
                tec_smooth[i,:,:] = np.reshape(ystar,(Nt,Nd))
            tec_smooth *= tec_std
            tec_smooth += tec_mean
            sess.run([tec_sync_op],feed_dict={_tec_sync:tec_smooth.flatten()})
                
        loss_array.append(losses)
        train_time.append(tm.mktime(tm.gmtime()))
        epoch += 1
    c_res,tec_res = sess.run([_c, _tec])

    loss_array = np.array(loss_array)
    train_time = np.array(train_time)
    loss_names = ['neg_log_likelihood_data', 'neg_log_likelihood_c']
    plt.figure(figsize=(12,12))
    for i,(loss,name) in enumerate(zip(loss_array.T,loss_names)):
        plt.plot(train_time,loss, label=name)
    plt.ylabel("loss")
    plt.xlabel("time (s)")
    plt.yscale("log")
    plt.legend(frameon=False)
    plt.savefig("training_curve.png")
    plt.show()
    return c_res, tec_res, antennas, directions, times[timestep]

    
def plot_res(c,tec,antennas, directions, times):
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
    phase = c + tec*(-8.4480e9)/140e6 
    print(phase.shape)
    phase_nearest = np.angle(np.exp(1j*interp_nearest(X_angular[:,0],X_angular[:,1],phase[-1,0,:],Xstar[:,0],Xstar[:,1]).reshape([25,25])))
    import cmocean
    tec_cm = cmocean.cm.phase

    fig = plt.figure(figsize=(6,6))
    
    extent = (np.min(X[:,0]), np.max(X[:,0]),np.min(X[:,1]), np.max(X[:,1]))

    ax = fig.add_subplot(1,1,1)
    #plot first time slot for example
    #print(X.shape,y.shape,sigma_y.shape)
    sc = ax.imshow(phase_nearest,origin='lower',
                    extent=extent,
                    cmap=tec_cm,
                    vmin = -np.pi,vmax=np.pi)#X[0][:,0],X[0][:,1],c=y[0],marker='+')
    plt.colorbar(sc)
    ax.set_title("phase")
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
                                    f['flags'],timestep=[0], freq_step = 10)
        tec -= tec[0,...]
        plot_res(c,tec,antennas, directions, time)
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
                                    f['flags'],timestep=[0], freq_step = 1,c_prior=c,tec_prior=tec)
        tec -= tec[0,...]
        plot_res(c,tec,antennas, directions, time)

        


        f = h5py.File("res0.hdf5","w")
        f['tec'] = tec
        f['c'] = c
        f['ra'] = directions.ra.deg
        f['dec'] = directions.dec.deg
        f.close()

            
if __name__ == '__main__':
    main("vis_data.hdf5")
