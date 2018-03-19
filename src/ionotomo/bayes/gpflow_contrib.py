from __future__ import absolute_import

import tensorflow as tf
import numpy as np

from gpflow import settings,autoflow
from gpflow import densities
from gpflow.densities import multivariate_normal
from gpflow import transforms
from gpflow.decors import (params_as_tensors, name_scope)
from gpflow.params import (Parameter,DataHolder,Minibatch)
from gpflow.likelihoods import Likelihood

from gpflow.models import GPModel

class Gaussian_v2(Likelihood):
    def __init__(self, Y_var=None,var=1.0, trainable=True,minibatch_size=None):
        super().__init__()
        self.variance = Parameter(
                var, transform=transforms.positive, dtype=settings.float_type, trainable=trainable)
        
        if Y_var is None:
            self.relative_variance = 1.0
        else:
            if minibatch_size is None:
                self.relative_variance = DataHolder(Y_var[:,None]+1e-3)
            else:
                self.relative_variance = Minibatch(Y_var[:,None]+1e-3, batch_size=minibatch_size, shuffle=True, seed=0)


    @params_as_tensors
    def logp(self, F, Y):
        if self.relative_variance == 1.0:
            return densities.gaussian(F, Y, self.variance)
        else:
            # var should be [:,1]
            return densities.gaussian(F, Y, self.variance*self.relative_variance)


    @params_as_tensors
    def conditional_mean(self, F):  # pylint: disable=R0201
        return tf.identity(F)

    @params_as_tensors
    def conditional_variance(self, F):
        return self.relative_variance*self.variance
        #return tf.fill(tf.shape(F), tf.squeeze(self.relative_variance*self.variance))

    @params_as_tensors
    def predict_mean_and_var(self, Fmu, Fvar):
        return tf.identity(Fmu), Fvar + self.relative_variance*self.variance

    @params_as_tensors
    def predict_density(self, Fmu, Fvar, Y):
        return densities.gaussian(Fmu, Y, Fvar + self.relative_variance*self.variance)

    @params_as_tensors
    def variational_expectations(self, Fmu, Fvar, Y):
        return -0.5 * np.log(2 * np.pi) - 0.5 * tf.log(self.relative_variance*self.variance) \
               - 0.5 * (tf.square(Y - Fmu) + Fvar) / (self.relative_variance*self.variance)

class GPR_v2(GPModel):
    """
    Gaussian Process Regression.

    This is a vanilla implementation of GP regression with a Gaussian
    likelihood.  Multiple columns of Y are treated independently.

    The log likelihood i this models is sometimes referred to as the 'marginal log likelihood', and is given by

    .. math::

       \\log p(\\mathbf y \\,|\\, \\mathbf f) = \\mathcal N\\left(\\mathbf y\,|\, 0, \\mathbf K + \\sigma_n \\mathbf I\\right)
    """
    def __init__(self, X, Y,  kern, Z=None, Zy=None,Zvar = None,mean_function=None, minibatch_size=None, var = 1.0, shuffle=True, trainable_var=True,**kwargs):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        kern, mean_function are appropriate GPflow objects
        minibatch_size, if not None, turns on mini-batching with that size.
        vector_obs_variance if not None (default) is vectorized measurement variance
        """
        Z = DataHolder(Z) if (Z is not None) and (Zy is not None) and (Zvar is not None) else None
        Zy = DataHolder(Zy) if (Z is not None) and (Zy is not None) and (Zvar is not None) else None
        Zvar = DataHolder(Zvar) if (Z is not None) and (Zy is not None) and (Zvar is not None) else None

        if minibatch_size is None:
            X = DataHolder(X)
            Y = DataHolder(Y)
            Y_var = DataHolder(var)
        else:
            X = Minibatch(X, batch_size=minibatch_size, shuffle=shuffle, seed=0)
            Y = Minibatch(Y, batch_size=minibatch_size, shuffle=shuffle, seed=0)
            Y_var = Minibatch(var, batch_size=minibatch_size, shuffle=shuffle, seed=0)
        likelihood = Gaussian_v2(var=1.0,trainable=trainable_var)
        likelihood.relative_variance = Y_var

        GPModel.__init__(self, X, Y, kern, likelihood, mean_function, **kwargs)
        self.Z = Z
        self.Zy = Zy
        self.Zvar = Zvar

    def set_batch_size(self,size):
        self.X.set_batch_size(size)
        self.Y.set_batch_size(size)
        self.likelihood.relative_variance.set_batch_size(size)

    @property
    @params_as_tensors
    def XZ(self):
        if self.Z is not None:
            return tf.concat([self.Z,self.X],axis=0)
        else:
            return self.X

    @property
    @params_as_tensors
    def YZy(self):
        if self.Zy is not None:
            return tf.concat([self.Zy,self.Y],axis=0)
        else:
            return self.Y

    @property
    @params_as_tensors
    def relative_variance(self):
        if self.Zvar is not None:
            return tf.concat([self.Zvar,self.likelihood.relative_variance],axis=0)
        else:
            return self.likelihood.relative_variance


    

    @params_as_tensors
    @autoflow()
    def eval(self):
        return self.X,self.Y,self.relative_variance

    @name_scope('likelihood')
    @params_as_tensors
    def _build_likelihood(self):
        """
        Construct a tensorflow function to compute the likelihood.

            \log p(Y | theta).
        """
        K = self.kern.K(self.XZ) + tf.eye(tf.shape(self.XZ)[0], dtype=settings.float_type) * (self.likelihood.variance * self.relative_variance)
        L = tf.cholesky(K)
        m = self.mean_function(self.XZ)
        logpdf = multivariate_normal(self.YZy, m, L)  # (R,) log-likelihoods for each independent dimension of Y

        return tf.reduce_sum(logpdf)

    @name_scope('predict')
    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False):
        """
        Xnew is a data matrix, point at which we want to predict

        This method computes

            p(F* | Y )

        where F* are points on the GP at Xnew, Y are noisy observations at X.

        """
        Kx = self.kern.K(self.XZ, Xnew)
        K = self.kern.K(self.XZ) + tf.eye(tf.shape(self.XZ)[0], dtype=settings.float_type) * (self.likelihood.variance * self.relative_variance)
        L = tf.cholesky(K)
        A = tf.matrix_triangular_solve(L, Kx, lower=True)
        V = tf.matrix_triangular_solve(L, self.YZy - self.mean_function(self.XZ))
        fmean = tf.matmul(A, V, transpose_a=True) + self.mean_function(Xnew)
        if full_cov:
            fvar = self.kern.K(Xnew) - tf.matmul(A, A, transpose_a=True)
            shape = tf.stack([1, 1, tf.shape(self.YZy)[1]])
            fvar = tf.tile(tf.expand_dims(fvar, 2), shape)
        else:
            fvar = self.kern.Kdiag(Xnew) - tf.reduce_sum(tf.square(A), 0)
            fvar = tf.tile(tf.reshape(fvar, (-1, 1)), [1, tf.shape(self.YZy)[1]])
        return fmean, fvar

def test2():
    import gpflow as gp
    X=np.arange(100)[:,None]
    Y=np.arange(100)[:,None]
    var = np.arange(100)
    kern = gp.kernels.RBF(1,lengthscales=[0.05])#*gp.kernels.Periodic(1)
    mean = gp.mean_functions.Zero()
    m = GPR_v2(X.astype(float),Y.astype(float),kern,var=var.astype(float),minibatch_size=1,shuffle=False)
    o = gp.train.ScipyOptimizer(method='BFGS')
    o.minimize(m,maxiter=1)

    print(m.eval())
    y,var = m.predict_y(X)
    print(m.eval())


def test():
    import gpflow as gp
    import pylab as plt
    np.random.seed(0)
    X = np.linspace(0,1,1000)[:,None]
    sigma_y = np.random.uniform(size=1000)
    sigma_y[sigma_y < 0.5] = 0.01
    #sigma_y = 0.5*np.ones([100])
    #sigma_y[50:] = 0.
    F = np.exp(-X)*np.sin(50*X) + X**2*np.sin(10*X)
    Y = F + sigma_y*np.random.normal(size=X.shape)

    k = gp.kernels.RBF(1,lengthscales=[0.1])#*gp.kernels.Periodic(1)
    kern = k
    mean = gp.mean_functions.Zero()
    m = GPR_v2(X,Y,kern,Z=X[500:550,:],Zy=Y[500:550,:],Zvar=(sigma_y[500:550])**2,var=(sigma_y)**2,minibatch_size=100,trainable_var=False)
    o = gp.train.ScipyOptimizer(method='BFGS')
    sess = m.enquire_session()
    with sess.as_default():
        print(m.objective.eval())
        o.minimize(m,maxiter=100)
        print(m.objective.eval())
        o.minimize(m,maxiter=100)
        print(m.objective.eval())
    #print(m)
#    o = gp.train.AdamOptimizer(0.001)
#    o.minimize(m,maxiter=1000)
#    #     o = gp.train.HMC()
#    #     print(o.sample(m,1000,0.001))
    print(m)
    m.set_batch_size(1000)
    y,var = m.predict_y(X)
    print(y)
    plt.plot(X[:,0],Y[:,0])
    #plt.fill_between(X[:,0],Y[:,0]+sigma_y,Y[:,0] - sigma_y,alpha=0.25)
    plt.plot(X[:,0],y[:,0])
    plt.fill_between(X[:,0],y[:,0]+np.sqrt(var[:,0]),y[:,0] - np.sqrt(var[:,0]),alpha=0.25)
    plt.show()

def test_svgp():
    from gpflow.kernels import RBF, White
    from gpflow.models.svgp import SVGP
    from gpflow.training import AdamOptimizer

    from scipy.stats import mode
    from scipy.cluster.vq import kmeans2

    import gpflow as gp
    import pylab as plt
    plt.style.use('ggplot')

    np.random.seed(0)
    X = np.linspace(0,1,10000)[:,None]
    sigma_y = np.random.uniform(size=10000)
    sigma_y[sigma_y < 0.5] = 0.01
    #sigma_y = 0.5*np.ones([100])
    #sigma_y[50:] = 0.
    F = np.exp(-X)*np.sin(50*X) + X**2*np.sin(10*X)
    Y = F + sigma_y[:,None]*np.random.normal(size=X.shape)

    k = gp.kernels.RBF(1,lengthscales=[0.1])#*gp.kernels.Periodic(1)
    kern = k
    mean = gp.mean_functions.Zero()

    M = 100
    Z = kmeans2(X, M, minit='points')[0]
    with gp.defer_build():
        m_sgp = SVGP(X, Y, RBF(1, lengthscales=0.05, variance=1.)*gp.kernels.Periodic(1), Gaussian_v2(Y_var=sigma_y**2+1e-3, trainable=False, minibatch_size=100), 
             Z=Z, num_latent=1, minibatch_size=100, whiten=True)
        m_sgp.feature.set_trainable(False)
        #m_sgp.kern.lengthscales.prior = gp.priors.Gaussian(0,0.3)
        m_sgp.compile()
    iterations = 1000
    AdamOptimizer(0.1).minimize(m_sgp, maxiter=iterations)

    def assess_model_sgp(model, X_batch, Y_batch):
        m, v = model.predict_y(X_batch)
        l = model.predict_density(X_batch, Y_batch)
        a = np.abs(m.reshape(Y_batch.shape) - Y_batch).mean()
        return l, a

    def batch_assess(model, assess_model, X, Y,minibatch_size=100):
        n_batches = max(int(len(X)/minibatch_size), 1)
        lik, acc = [], []
        for X_batch, Y_batch in zip(np.split(X, n_batches), np.split(Y, n_batches)):
            l, a = assess_model(model, X_batch, Y_batch)
            lik.append(l)
            acc.append(a)
#        lik = np.concatenate(lik, 0)
#        acc = np.array(np.concatenate(acc, 0), dtype=float)
        return np.average(lik), np.average(acc)
    
#    l,a = batch_assess(m_sgp, assess_model_sgp, X, Y, 100)
#    print("Likelihood: {}, MAE: {}".format(l,a))
    print(m_sgp)

    Xs = np.linspace(0,1,100)[:,None]
    m, v = m_sgp.predict_y(Xs)
    plt.scatter(X[:,0],Y[:,0],alpha=0.1)
    plt.plot(Xs,m[:,0])
    plt.fill_between(Xs[:,0],m[:,0] + np.sqrt(v)[:,0],m[:,0] - np.sqrt(v)[:,0],alpha=0.3)
    plt.plot(m_sgp.feature.Z.value, np.zeros(m_sgp.feature.Z.value.shape),'k|',mew=2)
    plt.plot(Z, np.zeros(Z.shape),'g|',mew=2)

    plt.show()
    
if __name__=='__main__':
    test_svgp()
