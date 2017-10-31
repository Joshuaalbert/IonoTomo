import tensorflow as tf
import numpy as np


import numpy as np
from scipy.linalg import cho_solve
from ionotomo.utils.cho_solver import cho_back_substitution
from sympy import lambdify, Matrix, symbols, exp, sqrt, Rational, factorial, sin,pi
from scipy.optimize import fmin_l_bfgs_b
from scipy.spatial.distance import (pdist,cdist,squareform)
import sys
import logging as log


class KernelND(object):
    '''Base class for kernels in ND.
    ndims : int
        the number of dimensions of input
    '''
    def __init__(self,_hyperparams={},_hyperparams_bounds={},**kwargs):
        assert isinstance(_hyperparams,dict)
        assert isinstance(_hyperparams_bounds,dict)
        _hyperparams.update(kwargs.get("hyperparams",{}))
        _hyperparams_bounds.update(kwargs.get("hyperparams_bounds",{}))
        self.hyperparams = {}
        self.fixed = {}
        self.hyperparams_bounds = {}
        self.built = False
        for name in _hyperparams:
            self._add_hyperparam(name,_hyperparams[name],bounds = _hyperparams_bounds.get(name,None))

    def _add_hyperparam(self,name,value,bounds=None):
        self.hyperparams[name] = value
        self.fixed[name] = False
        if bounds is None:
            self.hyperparams_bounds[name] = [1e-5,1e5]
        else:
            self.hyperparams_bounds[name] = bounds

    def _log_normal_initializer(self,lower,upper,seed=None):
        def _initializer(shape, dtype, partition_info=None):
            return tf.exp(tf.random_uniform(shape,lower,upper,dtype,seed=None))
        return _initializer

    def build(self,batch_size,multi_dataset=False,use_initializer=True,seed=None):
        """Set up the variables (hyperparams)"""
        if self.built:
            return
        self.variables = {}
        self.sync_ops = []
        self.sync_placeholders = {}
        with tf.variable_scope("{}_hyperparams".format(type(self).__name__)):
            #self.batch_size = tf.placeholder(tf.int32,shape=(), name='batch_size')
            self.variables = {}
            for name in self.hyperparams.keys():
                bounds = self.hyperparams_bounds[name]
                value = [self.hyperparams[name]]
                if multi_dataset:
                    shape=[1,1,1]
                else:
                    shape=[batch_size,1,1]
                if use_initializer and not self.fixed[name]:
                    if bounds[0] > 0 and bounds[1] > 0:
                        self.variables[name] = tf.get_variable(\
                                name,
                                shape,
                                dtype=tf.float64,
                                initializer=self._log_normal_initializer(np.log(bounds[0]),np.log(bounds[1]),seed=seed),
                                trainable=True)
                    else:
                        self.variables[name] = tf.get_variable(\
                                name,
                                shape,
                                dtype=tf.float64,
                                initializer=tf.random_uniform_initializer(bounds[0],bounds[1],seed=seed),
                                trainable=True)
                else:
                    self.variables[name] = tf.get_variable(\
                            name,
                            initializer=tf.ones(shape,tf.float64)*tf.constant(value,dtype=tf.float64),
                            trainable=not self.fixed[name]) 
                self.variables[name] = tf.clip_by_value(self.variables[name],bounds[0],bounds[1])
                #batch_size , 1, 1
                #self.variables[name] = tf.expand_dims(tf.expand_dims(self.variables[name],axis=-1),axis=-1)
#                if multi_dataset:
#                     self.variables[name] = tf.expand_dims(self.variables[name],axis=0)


#                self.sync_placeholders[name] = tf.placeholder(tf.float64,shape=[batch_size,1,1],name='sync_{}'.format(name))
#                self.sync_ops.append(tf.assign(self.variables[name],self.sync_placeholders[name]))
        self.built = True

#    def _sync_variables(self,sess):
#        """assign self.hyperparams to self.variables"""
#        feed_dict = {}
#        for name in self.hyperparams.keys():
#            feed_dict[self.sync_placeholders[name]] = self.hyperparams[name]
#        sess.run(self.sync_ops,feed_dict=feed_dict)

    def call(self,X,Y=None,share_x=False):
        """Construct the sub graph defining this kernel.
        Return an output tensor"""
        raise NotImplementedError("Setup in subclass")

    def fix(self,name):
        '''Sets the given hyperparam to be fixed.
        # Example
        K = SquaredExponential(1)
        K.fix("l")
        #results in symbol "l" having zero derivative
        '''
        assert name in self.hyperparams.keys()
        self.fixed[name] = True

    def unfix(self,name):
        '''Sets the given hyperparam to be trainable.
        # Example
        K = SquaredExponential(1)
        K.fix("l")
        #results in symbol "l" having zero derivative
        '''
        assert name in self.hyperparams.keys()
        self.fixed[name] = False
        
    def set_hyperparams_bounds(self,name,bounds):
        assert name in self.hyperparams.keys(),"{} is not a valid name".format(name)
        self.hyperparams_bounds[name] = bounds

    def set_hyperparams(self,hp):
        assert len(hp) == len(self.hyperparams)
        assert isinstance(hp,dict)
        self.hyperparams.update(hp)
    def get_hyperparams(self):
        return self.hyperparams
    def get_variables(self):
        return self.variables
                
    def __add__(self,K):
        '''Add another Kernel or SumKernel. Creates a SumKernel object'''
        assert isinstance(K,KernelND), "Can only add kernels to kernels"
        return SumKernel([self,K])

    def __mul__(self,K):
        """Multiply the input kernel by this kernel and return a ProdKernel"""
        assert isinstance(K,KernelND), "Can only multiply kernels by kernels"
        return ProductKernel([self,K])

    def __pow__(self,b):
        """Exponentiate the input kernel by this kernel and return a ExpKernel"""
        return PowKernel([self],b)

    def __repr__(self):
        """Get the string repr of this kernel, for pretty print"""
        s = "** Kernel {} **\n".format(type(self).__name__)
        for i,name in enumerate(self.hyperparams.keys()):
            if self.fixed[name]:
                s += "{} : {} in [{} , {}] (fixed)\n".format(name,np.ravel(self.hyperparams[name]),*self.hyperparams_bounds[name])
            else:
                s += "{} : {} in [{} , {}]\n".format(name,np.ravel(self.hyperparams[name]),*self.hyperparams_bounds[name])
        return s

    

class MultiKernel(KernelND):
    def __init__(self,kernels,**kwargs):
        kernels = list(kernels)
        #
        super(MultiKernel,self).__init__(**kwargs)
        self.kernels = kernels

    def build(self,batch_size=1,multi_dataset=False,use_initializer=True,**kwargs):
        """Set up the variables (hyperparams)"""
        for K in self.kernels:
            K.build(batch_size=batch_size,
                    multi_dataset=multi_dataset,
                    use_initializer=use_initializer,
                    **kwargs)
        
    @property
    def kernels(self):
        return self._kernels
    @kernels.setter
    def kernels(self,kernels):
        self._kernels = kernels
#        for K in kernels:
#            if not isinstance(K,KernelND):
#                raise TypeError("Only add KernelND, {}".format(type(K)))
#            assert K.ndims == kernels[0].ndims, "Only add like dim kernels"
#        self._kernels = kernels
#
    def set_hyperparams(self,hp):
        assert isinstance(hp,(list,tuple))
        assert len(hp) == len(self.kernels)
        for i in range(len(self.kernels)):
            self.kernels[i].set_hyperparams(hp[i])

    def get_hyperparams(self):
        hp = []
        for K in self.kernels:
            hp.append(K.get_hyperparams())
        return hp
    def get_variables(self):
        var = []
        for K in self.kernels:
            var.append(K.get_variables())
        return var
        
class SumKernel(MultiKernel):
    def __init__(self,kernels,**kwargs):
        super(SumKernel,self).__init__(kernels,**kwargs)    
        assert len(self.kernels) > 1
    def call(self,X,Y=None,share_x=False):
        """Construct the sub graph defining this kernel.
        Return an output tensor"""
        output = self.kernels[0].call(X,Y,share_x)
        for i in range(1,len(self.kernels)):
            output += self.kernels[i].call(X,Y,share_x)
        return output
    def __repr__(self):
        s = "**************\n"
        s += self.kernels[0].__repr__()
        for i in range(1,len(self.kernels)):
            s += "** + **\n"
            s += self.kernels[i].__repr__()
        s += "**************\n"

        return s

class ProductKernel(MultiKernel):
    def __init__(self,kernels,**kwargs):
        super(ProductKernel,self).__init__(kernels,**kwargs)
        assert len(self.kernels) > 1
    def call(self,X,Y=None,share_x=False):
        output = self.kernels[0].call(X,Y,share_x)
        for i in range(1,len(self.kernels)):
            output *= self.kernels[i].call(X,Y,share_x)
        return output
    def __repr__(self):
        s = "**************\n"
        s += self.kernels[0].__repr__()
        for i in range(1,len(self.kernels)):
            s += "** x **\n"
            s += self.kernels[i].__repr__()
        s += "**************\n"
        return s

class PowKernel(MultiKernel):
    def __init__(self,kernels,b,**kwargs):
        super(PowKernel,self).__init__(kernels,**kwargs)
        assert int(b) == b, "only integer powers are valid kernels"
        self.b = int(b)
        assert len(self.kernels) == 1
    def call(self,X,Y=None,share_x=False):
        output = self.kernels[0].call(X,Y,share_x)**self.b
        return output
    def __repr__(self):
        s = "*****POW({})******\n".format(self.b)
        s += self.kernels[0].__repr__()
        s += "**************\n"
        return s

def cdist(x,y):
    """do pdist
    x : Tensor (batch_size,num_points,ndims)"""
    #D[:,i,j] = (a[:,i] - b[:,j]) (a[:,i] - b[:,j])'
    #=   a[:,i,p] a[:,i,p]' - b[:,j,p] a[:,i,p]' - a[:,i,p] b[:,j,p]' + b[:,j,p] b[:,j,p]'
    # batch_size,num_points,1
    r1 = tf.reduce_sum(x*x,axis=-1,keep_dims=True)
    r2 = tf.reduce_sum(y*y,axis=-1,keep_dims=True)
    out = r1 - 2*tf.matmul(x,y,transpose_b=True) +  tf.transpose(r2,perm=[0,2,1])
    
    return out

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


class SquaredExponential(KernelND):
    def __init__(self,_hyperparams={'l':1.,'sigma':1.},**kwargs):
        super(SquaredExponential,self).__init__(_hyperparams=_hyperparams,**kwargs)
    def call(self,X,Y=None,share_x=False):
        """return SE kernel.
        inputs : Tenor (batch_size,ndims)
            Input coordinates in ndims
        returns kernel evaluated at all pair computations
        """
        # batch_size,N,M
        if share_x:
            X = tf.expand_dims(X,0)
        if Y is None:
            x2 = pdist(X)
        else:
            x2 = cdist(X,Y)
        out = self.variables['sigma']**2 * tf.exp(-x2/(2*self.variables['l']**2))
        return out

class SquaredExponentialSep(KernelND):
    def __init__(self,dim,_hyperparams={'l':1.,'sigma':1.},**kwargs):
        super(SquaredExponentialSep,self).__init__(_hyperparams=_hyperparams,**kwargs)
        self.dim = int(dim)
    def call(self,X,Y=None,share_x=False):
        """return SE kernel.
        inputs : Tenor (batch_size,ndims)
            Input coordinates in ndims
        returns kernel evaluated at all pair computations
        """
        # batch_size,ndims, 1
        if share_x:
            X = tf.expand_dims(X,0)
        if Y is None:
            x2 = pdist(X[:,:,self.dim:self.dim+1])
        else:
            x2 = cdist(X[:,:,self.dim:self.dim+1],Y[:,:,self.dim:self.dim+1])
        out = self.variables['sigma']**2 * tf.exp(-x2/(2*self.variables['l']**2))
        return out

class GammaExponential(KernelND):
    def __init__(self,_hyperparams={'l':1.,'gamma':1.,'sigma':1.},**kwargs):
        super(GammaExponential,self).__init__(_hyperparams=_hyperparams,_hyperparams_bounds={'gamma':[1e-5,2.]},**kwargs)
    def call(self,X,Y=None,share_x=False):
        # batch_size,ndims, 1
        if share_x:
            X = tf.expand_dims(X,0)
        if Y is None:
            x2 = pdist(X)
        else:
            x2 = cdist(X,Y)
        out = self.variables['sigma']**2 * tf.exp(-(x2/(2*self.variables['l']**2)**(self.variables['gamma']/2.)))
        return out

class GammaExponentialSep(KernelND):
    def __init__(self,dim,_hyperparams={'l':1.,'gamma':1.,'sigma':1.},**kwargs):
        super(GammaExponentialSep,self).__init__(_hyperparams=_hyperparams,_hyperparams_bounds={'gamma':[1e-5,2.]},**kwargs)
        self.dim = int(dim)
    def call(self,X,Y=None,share_x=False):
        # batch_size,ndims, 1
        if share_x:
            X = tf.expand_dims(X,0)
        if Y is None:
            x2 = pdist(X[:,:,self.dim:self.dim+1])
        else:
            x2 = cdist(X[:,:,self.dim:self.dim+1],Y[:,:,self.dim:self.dim+1])
        out = self.variables['sigma']**2 * tf.exp(-(x2/(2*self.variables['l']**2)**(self.variables['gamma']/2.)))
        return out

class MaternP(KernelND):
    def __init__(self,p=1,_hyperparams={'l':1.,'sigma':1.},**kwargs):
        super(MaternP,self).__init__(_hyperparams=_hyperparams,**kwargs)
        self.p=int(p)
    def call(self,X,Y=None,share_x=False):
        from scipy.misc import factorial
        # batch_size,ndims, 1
        if share_x:
            X = tf.expand_dims(X,0)
        if Y is None:
            x2 = pdist(X)
        else:
            x2 = cdist(X,Y)
        r = tf.sqrt(x2)/self.variables['l']
        nu = self.p + 1./2.
        out = [factorial(self.p)/(factorial(self.p)) * \
                    (np.sqrt(8.*nu) * r )**self.p]
        for i in range(1,self.p+1): 
            out.append(factorial(self.p+i)/(factorial(i)*factorial(self.p-i)) * \
                    (np.sqrt(8.*nu) * r )**(self.p-i))
        out = tf.stack(out,axis=0)
        out = tf.reduce_sum(out,axis=0)

        out *= self.variables['sigma']**2 * tf.exp(-np.sqrt(2 * nu) * r) * factorial(self.p) / factorial(2*self.p)
        return out

class MaternPSep(KernelND):
    def __init__(self,dim,p=1,_hyperparams={'l':1.,'sigma':1.},**kwargs):
        super(MaternPSep,self).__init__(_hyperparams=_hyperparams,**kwargs)
        self.dim=int(dim)
        self.p=int(p)
    def call(self,X,Y=None,share_x=False):
        from scipy.misc import factorial
        # batch_size,ndims, 1
        if share_x:
            X = tf.expand_dims(X,0)
        if Y is None:
            x2 = pdist(X[:,:,self.dim:self.dim+1])
        else:
            x2 = cdist(X[:,:,self.dim:self.dim+1],Y[:,:,self.dim:self.dim+1])
        r = tf.sqrt(x2)/self.variables['l']
        nu = self.p + 1./2.
        out = [factorial(self.p)/(factorial(self.p)) * \
                    (np.sqrt(8.*nu) * r )**self.p]
        for i in range(1,self.p+1): 
            out.append(factorial(self.p+i)/(factorial(i)*factorial(self.p-i)) * \
                    (np.sqrt(8.*nu) * r )**(self.p-i))
        out = tf.stack(out,axis=0)
        out = tf.reduce_sum(out,axis=0)

        out *= self.variables['sigma']**2 * tf.exp(-np.sqrt(2 * nu) * r) * factorial(self.p) / factorial(2*self.p)
        return out


class Periodic(KernelND):
    def __init__(self,_hyperparams={'l':1.,'p':1.,'sigma':1.},**kwargs):
        super(Periodic,self).__init__(_hyperparams=_hyperparams,**kwargs)
    def call(self,X,Y=None,share_x=False):
        # batch_size,ndims, 1
#        if share_x:
#            X = tf.expand_dims(X,0)
        if Y is None:
            x2 = pdist(X)
        else:
            x2 = cdist(X,Y)
        r = tf.sqrt(x2)/self.variables['p']
        out = self.variables['sigma']**2 * tf.exp(-2*(tf.sin(np.pi * r) / self.variables['l'])**2)
        return out

class PeriodicSep(KernelND):
    def __init__(self,dim,_hyperparams={'l':1.,'p':1.,'sigma':1.},**kwargs):
        super(PeriodicSep,self).__init__(_hyperparams=_hyperparams,**kwargs)
        self.dim = int(dim)
    def call(self,X,Y=None,share_x=False):
        # batch_size,ndims, 1
#        if share_x:
#            X = tf.expand_dims(X,0)
        if Y is None:
            x2 = pdist(X[:,:,self.dim:self.dim+1])
        else:
            x2 = cdist(X[:,:,self.dim:self.dim+1],Y[:,:,self.dim:self.dim+1])
        r = tf.sqrt(x2)/self.variables['p']
        out = self.variables['sigma']**2 * tf.exp(-2*(tf.sin(np.pi * r) / self.variables['l'])**2)
        return out

class Diagonal(KernelND):
    def __init__(self,_hyperparams={'sigma':1.},**kwargs):
        super(Diagonal,self).__init__(_hyperparams=_hyperparams,**kwargs)
    def call(self,X,Y=None,share_x=False):
        # batch_size,ndims, 1
#        if share_x:
#            X = tf.expand_dims(X,0)
        if Y is None:
            xshape = tf.shape(X)
            I = tf.eye(xshape[1],batch_shape=[xshape[0]])
        else:
            yshape = tf.shape(Y)
            I = tf.eye(num_rows=xshape[1],num_columns=yshape[1],batch_shape=[xshape[0]])
        out = self.variables['sigma']**2 * I
        return out


class RationalQuadratic(KernelND):
    def __init__(self,_hyperparams={'l':1.,'alpha':1.,'sigma':1.},**kwargs):
        super(RationalQuadratic,self).__init__(_hyperparams=_hyperparams,**kwargs)
    def call(self,X,Y=None,share_x=False):
        # batch_size,ndims, 1
#        if share_x:
#            X = tf.expand_dims(X,0)
        if Y is None:
            x2 = pdist(X)
        else:
            x2 = cdist(X,Y)
        r = x2/(2 * self.variables['l']**2 * self.variables['alpha'])
        out = self.variables['sigma']**2 * (1 + r)**self.variables['alpha']
        return out

class RationalQuadraticSep(KernelND):
    def __init__(self,dim,_hyperparams={'l':1.,'alpha':1.,'sigma':1.},**kwargs):
        super(RationalQuadraticSep,self).__init__(_hyperparams=_hyperparams,**kwargs)
        self.dim = int(dim)
    def call(self,X,Y=None,share_x=False):
        # batch_size,ndims, 1
#        if share_x:
#            X = tf.expand_dims(X,0)
        if Y is None:
            x2 = pdist(X[:,:,self.dim:self.dim+1])
        else:
            x2 = cdist(X[:,:,self.dim:self.dim+1],Y[:,:,self.dim:self.dim+1])
        r = x2/(2 * self.variables['l']**2 * self.variables['alpha'])
        out = self.variables['sigma']**2 * (1 + r)**self.variables['alpha']
        return out

class DotProduct(KernelND):
    def __init__(self,_hyperparams={'c':0,'sigma_b':1.,'sigma_v':1.},**kwargs):
        super(DotProduct,self).__init__(_hyperparams=_hyperparams,
                _hyperparams_bounds={'c':[-1e5,1e5]},**kwargs)
    def call(self,X,Y=None,share_x=False):
        # batch_size,ndims, 1
#        if share_x:
#            X = tf.expand_dims(X,0)
        X -= self.variables['c']
        if Y is None:
            #batch_size, num_points, ndim 
            x2 = tf.matmul(X,X,transpose_b=True)
        else:
            Y -= self.variables['c']
            x2 = tf.matmul(X,Y,transpose_b=True)
        out = x2 * self.variables['sigma_v']**2 + self.variables['sigma_b']**2
        return out

class DotProductSep(KernelND):
    def __init__(self,dim,_hyperparams={'c':0,'sigma_b':1.,'sigma_v':1.},**kwargs):
        super(DotProductSep,self).__init__(_hyperparams=_hyperparams,
                _hyperparams_bounds={'c':[-1e5,1e5]},**kwargs)
        self.dim = int(dim)
    def call(self,X,Y=None,share_x=False):
        # batch_size,ndims, 1
#        if share_x:
#            X = tf.expand_dims(X,0)
        X = X[:,:,self.dim:self.dim+1] - self.variables['c']
        if Y is None:
            #batch_size, num_points, ndim 
            x2 = tf.matmul(X,X,transpose_b=True)
        else:
            Y = Y[:,:,self.dim:self.dim+1] - self.variables['c']
            x2 = tf.matmul(X,Y,transpose_b=True)
        out = x2 * self.variables['sigma_v']**2 + self.variables['sigma_b']**2
        return out

class PhaseScreen(KernelND):
    """
    D(t,x,y,a,b) + SE(tau_slow)SE(tau_long)
    + RQ(1/6,l_inertia)SE(L_outer)
    + PE(gamma)
    + SE(freq
    """
    def __init__(self,_hyperparams={'sigma_D':1.,
        'sigma_temporal':1.,'tau_slow': 1., 'tau_quick':1.,
        'sigma_turb':1.,'l_inertia':1.,'L_outer':1., 
        'sigma_angular':1., 'gamma':1.,
        'sigma_freq':1., 'l_freq':1.},
        _hyperparams_bounds={'sigma_D':[1e-5,10],
        'sigma_temporal':[1e-5,10],'tau_slow': [100,1000], 'tau_quick':[16,100],
        'sigma_turb':[1e-5,10],'l_inertia':[0.1,50],'L_outer':[20,100], 
        'sigma_angular':[1e-5,10], 'gamma':[0.01,20],
        'sigma_freq':[1e-5,10], 'l_freq':[0.2e6,50e5]},**kwargs):
        super(PhaseScreen,self).__init__(_hyperparams=_hyperparams,
                _hyperparams_bounds=_hyperparams_bounds,**kwargs)
    def call(self,X,Y=None):
        # batch_size,npoints, (ant_u, ant_v, ant_w, time, dir_u, dir_v, dir_w, freq)
        if Y is None:
            xshape = tf.shape(X)
            I = tf.eye(xshape[1],batch_shape=[xshape[0]])
            t2 = pdist(X[:,:,3:4])
            x2 = pdist(X[:,:,0:2])
            #batch_size, num_points, 3
            direction = X[:,:,4:7]
            cos = tf.matmul(direction,direction,transpose_b=True)
            sin2 = 1. - cos**2
            f2 = pdist(X[:,:,7:8])
        else:
            yshape = tf.shape(Y)
            I = tf.eye(num_rows=xshape[1],num_columns=yshape[1],batch_shape=[xshape[0]])
            t2 = cdist(X[:,:,3:4],Y[:,:,3:4])
            x2 = cdist(X[:,:,0:2],Y[:,:,0:2])
            #batch_size, num_points, 3
            direction_X = X[:,:,4:7]
            direction_Y = Y[:,:,4:7]
            cos = tf.matmul(direction_X,direction_Y,transpose_b=True)
            sin2 = 1. - cos**2
            f2 = cdist(X[:,:,7:8],Y[:,:,7:8])

        
        uncorrelated = self.variables['sigma_D']**2 * I
        temporal = self.variables['sigma_temporal']**2 * tf.exp(-t2*(1./self.variables['tau_slow']**2 + 1./self.variables['tau_quick']**2)/2.)
        alpha = 1./6.
        r = x2/(2 * self.variables['l_inertial']**2 * alpha)
        spatial = self.variables['sigma_turb']**2 * (1. + r)**alpha * tf.exp(-x2*(1./self.variables['L_outer']**2)/2.)
        angular = self.variables['sigma_angular']**2 * tf.exp(-2.*sin2/self.variables['gamma']**2)
        freq = self.variables['sigma_freq']**2 * tf.exp(-f2/(2.*self.variables['l_freq']**2))

        out = uncorrelated + temporal + spatial + angular + freq

        return out 

def is_singular(A):
    return np.linalg.cond(A) > 1/sys.float_info.epsilon

def _level1_solve(x,y,sigma_y,xstar,K,use_cholesky):
    with tf.variable_scope("level1_solve"):
        #batch_size
        n = tf.to_double(tf.shape(y)[1])
        Knn = K.call(x)
        Knm = K.call(x,xstar)
        Kmm = K.call(xstar)
        # batch_size, n,n
        Kf = Knn + tf.matrix_diag(sigma_y,name='sigma_y_diag')
        def _cho():
            # batch_size, n, n
            L = tf.cholesky(Kf,name='L')
            # batch_size, n
            alpha = tf.squeeze(tf.cholesky_solve(L, tf.expand_dims(y,-1), name='alpha'),axis=-1)
            # batch_size, n, m
            # batch_size, n
            # batch_size, m
            fstar = tf.matmul(Knm,tf.expand_dims(alpha,-1),transpose_a=True)
            #tf.einsum("bnm,bn->bm",Knm,alpha)
            cov = Kmm
            cov -= tf.matmul(Knm,tf.cholesky_solve(L,Knm),transpose_a=True)
            #tf.einsum("bnm,bnl->bml",Knm,tf.cholesky_solve(L,Knm))
            log_mar_like = -tf.reduce_sum(y*alpha,1)/2. - tf.reduce_sum(tf.log(tf.matrix_diag_part(L)),axis=1) - n*(np.log(2.*np.pi)/2.)
            return fstar,cov,log_mar_like

        def _no_cho():
            #Kf^-1 = Kf' (Kf Kf')^-1
            det = tf.matrix_determinant(Kf,name='detKf')
            det = tf.clip_by_value(det,1e-13, det)
            logdet = tf.log(det)
            #logdet = -tf.log(tf.matrix_determinant(tf.matrix_solve(Kf,tf.eye(n,batch_shape=[tf.shape(Kf)[0]],dtype=tf.float64))))
            #det(Kf) = 1/det(Kf^-1)
            #log(det(Kf)) = log(1./det(Kf^-1)) = -log(det(Kf^-1))
            #batch_size, n, 1
            alpha = tf.matrix_solve(Kf,tf.expand_dims(y,-1),name='solve_alpha')
            fstar = tf.matmul(Knm,tf.expand_dims(alpha,-1),transpose_a=True)
            cov = Kmm
            cov -= tf.matmul(Knm,tf.matrix_solve(Kf,Knm),transpose_a=True)
            log_mar_like = (-tf.reduce_sum(y*alpha,axis=1) - logdet - n*np.log(2.*np.pi))/2.
            return fstar,cov,log_mar_like
        return tf.cond(use_cholesky,_cho,_no_cho)

def _neg_log_mar_like(x,y,sigma_y,K,use_cholesky):
    with tf.variable_scope("neg_log_mar_like"):
        #batch_size
        n = tf.to_double(tf.shape(y)[1])
        Knn = K.call(x,x)
        # batch_size, n,n
        Kf = Knn + tf.matrix_diag(sigma_y,name='sigma_y_diag')
        def _cho(y=y):
            # batch_size, n, n
            L = tf.cholesky(Kf,name='L')
            # batch_size, n
            alpha = tf.squeeze(tf.cholesky_solve(L, tf.expand_dims(y,-1), name='alpha'),axis=-1)
            neg_log_mar_like = tf.reduce_sum(y*alpha,1)/2. + tf.reduce_sum(tf.log(tf.matrix_diag_part(L)),axis=1) + n*(np.log(2.*np.pi)/2.)
            return neg_log_mar_like

        def _no_cho(y=y):
            det = tf.matrix_determinant(Kf,name='detKf')
            det = tf.clip_by_value(det,1e-13, det)
            logdet = tf.log(det)
            #logdet = -tf.log(tf.matrix_determinant(tf.matrix_solve(Kf,tf.eye(n,batch_shape=[tf.shape(Kf)[0]],dtype=tf.float64))))

            #batch_size, n, 1
            alpha = tf.matrix_solve(Kf,tf.expand_dims(y,-1),name='solve_alpha')
            neg_log_mar_like = (tf.reduce_sum(y*alpha,axis=1) + logdet + n*np.log(2.*np.pi))/2.
            return neg_log_mar_like
        return tf.cond(use_cholesky,_cho,_no_cho)
        
def _level2_optimize(x,y,sigma_y,K,use_cholesky,learning_rate):
    with tf.variable_scope("level2_solve"):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        neg_log_mar_like =_neg_log_mar_like(x,y,sigma_y,K,use_cholesky)
        out = optimizer.minimize(tf.reduce_sum(neg_log_mar_like))
        return out, neg_log_mar_like

class Pipeline(object):
    """This class defines the problems that are to be solved using Gaussian processes.
    In general many problems can be solved at once using batching so long 
    as the dimensions are the same."""
    def __init__(self,batch_size,K,multi_dataset=False,share_x=False):
        assert isinstance(K,KernelND)
        self.K = K
        self.K.build(batch_size=batch_size,multi_dataset=multi_dataset,use_initializer=True)
        self.batch_size = batch_size
        self.multi_dataset = multi_dataset
        self.share_x = share_x
        self.sess = tf.Session()
        self._build()
        self.sess.run(tf.global_variables_initializer())
        
    def _build(self):
        with tf.variable_scope("pipeline"):
            self.X = tf.placeholder(tf.float64,shape=None, name='X')
            self.y = tf.placeholder(tf.float64,shape=None, name='y')
            self.sigma_y = tf.placeholder(tf.float64,shape=None, name='sigma_y')
            self.Xstar = tf.placeholder(tf.float64,shape=None, name='Xstar')
            self.use_cholesky = tf.placeholder(tf.bool,shape=(),name='use_cholesky')

            self.ystar, self.cov, self.lml = _level1_solve(self.X,self.y,self.sigma_y,self.Xstar,self.K,self.use_cholesky)
            self.learning_rate = tf.placeholder(tf.float64,shape=None, name='learning_rate')
            self.level2_op, self.neg_log_mar_like = _level2_optimize(self.X,self.y,self.sigma_y,self.K,self.use_cholesky,self.learning_rate)
        
        
    def level2_optimize(self,X,y,sigma_y,delta=0.,patience=20,epochs=1000):
        if self.share_x:
            if len(X.shape) == 2:
                X = np.expand_dims(X,0)
            if len(y.shape) == 1:
                y = np.expand_dims(y,0)
                y = np.tile(y,(self.batch_size,1))
            if len(sigma_y.shape) == 1:
                sigma_y = np.expand_dims(sigma_y,0)
                sigma_y = np.tile(sigma_y,(self.batch_size,1))
        else:
            assert len(X.shape) == 3
            assert len(y.shape) == 2
            assert y.shape[0] == X.shape[0]
        assert y.shape[1] == X.shape[1]
        assert y.shape == sigma_y.shape

        feed_dict = {self.X : X,
                self.y : y,
                self.sigma_y : sigma_y,
                self.learning_rate : 0.01}
        
        neg_log_mar_lik_last = np.inf
        patience_count = 0
        epoch_count = 0
        while epoch_count < epochs:
            
            epoch_count += 1
            try:
                feed_dict[self.use_cholesky] = True
                _, neg_log_mar_lik = self.sess.run([self.level2_op,self.neg_log_mar_like],feed_dict=feed_dict)
            except:
                feed_dict[self.use_cholesky] = False
                _, neg_log_mar_lik = self.sess.run([self.level2_op,self.neg_log_mar_like],feed_dict=feed_dict)
            print(neg_log_mar_lik)
            if (np.sum(neg_log_mar_lik)/np.sum(neg_log_mar_lik_last) - 1) > -delta:
                patience_count += 1
                feed_dict[self.learning_rate] /= 2.
                feed_dict[self.learning_rate] = max(0.0001,feed_dict[self.learning_rate])
                if patience_count > patience:
                    break
            else:
                neg_log_mar_lik_last = neg_log_mar_lik
                patience_count = 0
                feed_dict[self.learning_rate] *= 1.5
                feed_dict[self.learning_rate] = min(0.1,feed_dict[self.learning_rate])

        hp = self.sess.run(self.K.get_variables())
        self.K.set_hyperparams(hp)

        return neg_log_mar_lik

    def level1_predict(self,X,y,sigma_y, Xstar=None, smooth=False):
        '''
        Predictive distribution.
        X : array (batch_size,num_points, ndims)
            training input coords
        y : array (batch_size, num_points)
            training outputs
        sigma_y : array (batch_size, num_points)
            uncertainty for each y point
        Xstar : array (batch_size,num_points_test, ndims)
            test input coords.
        smooth : bool
            if True smooth using Xstar = X
        '''

        if self.share_x:
            if len(X.shape) == 2:
                X = np.expand_dims(X,0)
            if len(y.shape) == 1:
                y = np.expand_dims(y,0)
            if len(sigma_y.shape) == 1:
                sigma_y = np.expand_dims(sigma_y,0)
        else:
            assert len(X.shape) == 3
            assert len(y.shape) == 2
            assert y.shape[0] == X.shape[0]
        assert y.shape[1] == X.shape[1]
        assert y.shape == sigma_y.shape

        if Xstar is None and smooth:
            Xstar = X
        assert Xstar is not None
        assert Xstar.shape == X.shape
        
        feed_dict = {self.X : X,
                self.y : y,
                self.sigma_y : sigma_y,
                self.Xstar : Xstar}

        try:
            feed_dict[self.use_cholesky] = True
            ystar, cov, lml = self.sess.run([self.ystar, self.cov, self.lml],feed_dict=feed_dict)
        except:
            feed_dict[self.use_cholesky] = False
            ystar, cov, lml = self.sess.run([self.ystar, self.cov, self.lml],feed_dict=feed_dict)

        
        return ystar,cov,lml

def test_build():
    X = np.random.uniform(size=[10,250,1])
#    Xstar = np.linspace(-1,2,100)
#    Xstar,Ystar = np.meshgrid(xstar,xstar)
#    Xstar = np.expand_dims(np.array([Xstar.flatten(),Ystar.flatten()]).T,0)
#    y = np.sin(X[:,:,0]*2*np.pi/0.5) *np.cos( X[:,:,1]*np.pi/0.5*2.) + np.random.normal(size=X.shape[0:2])*0.1
    y = np.sin(X[:,:,0]*2*np.pi/0.5) + np.random.normal(size=X.shape[0:2])*0.1
    sigma_y = np.ones_like(y)*0.1

    K1 = SquaredExponential()
    K1.set_hyperparams_bounds('l',[1e-2, 5])
    K1.set_hyperparams_bounds('sigma',[1e-5,10])
    K2 = DotProduct()
    #K3 = RationalQuadratic(hyperparams={'alpha':1./6.})
    #K3.fix('alpha')
    K = K1# * K3
    p = Pipeline(10,K,multi_dataset=True,share_x = False)
    #print(p.level1_predict(X,y,sigma_y,smooth=True))
    print(K)
    print(p.level2_optimize(X,y,sigma_y,patience=20))
    print(K)
    ystar,cov,lml = p.level1_predict(X,y,sigma_y,smooth=True)
    import pylab as plt
    for i in range(10):
        plt.scatter(X[i,:,0],y[i,:])
        plt.scatter(X[i,:,0],ystar[i,:])
        plt.show()
    
if __name__=='__main__':
    test_build()
