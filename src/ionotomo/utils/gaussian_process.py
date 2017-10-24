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
    no_create : bool
        whether to run the create method
    '''
    def __init__(self,ndims,no_create=False,**kwargs):
        self.ndims = ndims
        self.symbolic_func = None
        self.func = None
        self.func_diff = None
        self.symbols = None
        self.positional_symbols = None
        self.positional_args = None
#        self.hyperparams = []
        if not no_create:
            self.create()
    def fix_hyperparams(self,name=None,index=None):
        """Fix the given hyperparam by name or index.
        {depreciated} use K.fixed = symbol_name
        """
        if index is None and name is None:
            raise ValueError("index or name must not be None")
        if name is not None:
            self.fixed = name
        if index is not None:
            assert index < len(self.symbols)
            self.fixed = self.symbols[i].name
    @property
    def fixed(self):
        '''Return array of type bool.
        element i is True iif hyperparam i is fixed
        '''
        return self._fixed
    @fixed.setter
    def fixed(self,name):
        '''Sets the given hyperparam to be fixed.
        Setting None will unfix all params of this kernel.
        
        # Example
        K = SquaredExponential(1)
        K.fixed = "l"
        #results in symbol "l" having zero derivative
        '''
        if name is None:
            self._fixed = [False for sym in self.symbols]
            return
        for i,sym in enumerate(self.symbols):
            if name == sym.name:
                self._fixed[i] = True
                return
        raise ValueError("{} is not a valid hyperparameter".format(name))
    def __add__(self,K):
        '''Add another Kernel or SumKernel. Creates a SumKernel object'''
        assert isinstance(K,KernelND), "Can only add kernels to kernels"
        assert self.ndims == K.ndims
        return SumKernel([self,K])
    def __mul__(self,K):
        """Multiply the input kernel by this kernel and return a ProdKernel"""
        assert isinstance(K,KernelND), "Can only multiply kernels by kernels"
        assert self.ndims == K.ndims
        return ProductKernel([self,K])
    def __exp__(self,K):
        raise NotImplementedError("Kernel exponentiation not implemented")

    def __repr__(self):
        """Get the string repr of this kernel, for pretty print"""
        s = "** Kernel {} **\n".format(type(self).__name__)
        for i,(sym,hp) in enumerate(zip(self.symbols,self.hyperparams)):
            if self.fixed[i]:
                s += "{} : {} in [{} , {}] (fixed)\n".format(sym.name,hp,*self.hyperparams_bounds[i])
            else:
                s += "{} : {} in [{} , {}]\n".format(sym.name,hp,*self.hyperparams_bounds[i])
        return s

    def create(self):
        """The super class create. 
        Will be called after children's create method.
        Expects several objects to exist."""
        assert self.symbols is not None, "Set symbols first in self.create"
        self._fixed = [False for sym in self.symbols]
        assert self.positional_symbols is not None, "set dimensional symbols in self.create"
        args = list(self.symbols) + list(self.positional_symbols)
#        from sympy import fourier_transform
#        fourier_symbols = [symbols("k_{:d}".format(i)) for i in range(self.ndims)]
#        fourier = self.func
#        for x,k in zip(self.positional_symbols,fourier_symbols):
#            fourier = fourier_transform(fourier,x,k,simplify=True,noconds=True)
#            print(fourier)
        try:
            func_diff = Matrix([self.symbolic_func.diff(s) for s in self.symbols])
            self.func = lambdify(args ,self.symbolic_func,modules=['numpy'])
            self.func_diff = lambdify(args ,func_diff,modules=['numpy'])
            #print(func_diff)
        except:
            raise NotImplementedError("Set up functions in self.create")
    @property
    def ndims(self):
        return self._ndims
    @ndims.setter
    def ndims(self,ndims):
        try:
            self._ndims = int(ndims)
        except:
            raise ValueError("ndims but be int castable")
    @property
    def hyperparams(self):
        assert self._hyperparams is not None, "No kernel defined"
        return self._hyperparams
    @hyperparams.setter
    def hyperparams(self,hp):
        self._hyperparams = list(hp)
    @property
    def hyperparams_bounds(self):
        if self._hyperparams_bounds is None:
            raise NotImplementedError("Setup in subclass")
        else:
            return self._hyperparams_bounds
    @hyperparams_bounds.setter
    def hyperparams_bounds(self,bounds):
        self.set_hyperparams_bounds(bounds)
        
    def set_hyperparams_bounds(self,bounds,index=None,name=None):
        hp_bounds = []
        if name is None and index is None:
            for i in range(len(self.hyperparams)):
                hp_bounds.append(bounds[i])
            self._hyperparams_bounds = hp_bounds
            return
        assert self._hyperparams_bounds is not None, "Set hyperparam bounds first"
        assert len(self._hyperparams_bounds) == len(self.hyperparams)
        if name is not None:
            i = 0
            while i < len(self.hyperparams):
                if name == self.symbols[i].name:
                    self._hyperparams_bounds[i] = bounds
                    return
                i += 1
            raise ValueError("{} is not a valid name".format(name))
        if index is not None:
            index = int(index)
            assert index < len(self._hyperparams_bounds)
            self._hyperparams_bounds[index] = bounds               
        
    @property
    def func(self):
        assert self._func is not None, "No kernel defined"
        return self._func
    @func.setter
    def func(self,f):
        self._func = f
    
    @property
    def symbolic_func(self):
        assert self._symbolic_func is not None, "No kernel defined"
        return self._symbolic_func
    @symbolic_func.setter
    def symbolic_func(self,f):
        self._symbolic_func = f

    @property
    def symbols(self):
        assert self._symbols is not None, "No kernel defined"
        return self._symbols
    @symbols.setter
    def symbols(self,s):
        self._symbols = s
    @property
    def positional_symbols(self):
        assert self._positional_symbols is not None, "No kernel defined"
        return self._positional_symbols
    @positional_symbols.setter
    def positional_symbols(self,s):
        self._positional_symbols = s
    @property
    def positional_args(self):
        assert self._positional_args is not None, "No kernel defined"
        return self._positional_args
    @positional_args.setter
    def positional_args(self,s):
        self._positional_args = s
    @property
    def func(self):
        assert self._func is not None, "No kernel defined"
        return self._func
    @func.setter
    def func(self,f):
        self._func = f
    @property
    def symbols(self):
        assert self._symbols is not None, "No kernel defined"
        return self._symbols
    @symbols.setter
    def symbols(self,s):
        self._symbols = s
    
    @property
    def func_diff(self):
        assert self._func_diff is not None, "No kernel defined"
        return self._func_diff
    @func_diff.setter
    def func_diff(self,f):
        self._func_diff = f
    def __call__(self,X,Y=None,eval_gradient=False):
        '''X is m by ndims
        Y is n by ndims, if None use K(X,X) else K(X,Y)
        if eval_derivative then return also derivatives [num_symbols,K.shape]'''
        assert not (eval_gradient and (Y is not None)), "Cannot use Y and do gradient"
        positional_args = self.eval_positional_args(X,Y=Y)
        res = self.eval(positional_args)
        #print(res)
        res[np.isnan(res)] = 0.
        if eval_gradient:
            grad = self.eval_derivative(positional_args)
            #zero out fixed
            grad[self.fixed,...] = 0.
            #print(grad)
            grad[np.isnan(grad)] = 0.
            return res, grad[:,0,...]
        return res
    def eval(self,positional_args):
        arg = [hp for hp in self.hyperparams] + positional_args
        res = self.func(*arg)
        return res
    def eval_derivative(self,positional_args):
        arg = [hp for hp in self.hyperparams] + positional_args
        res = self.func_diff(*arg)
        return res
    def eval_positional_args(self,*xyz):
        '''set self.positional_args'''
        raise NotImplementedError("Set up eval_positional_args")
    def sample_hyperparameters(self):
        bounds = self.hyperparams_bounds
        s = np.zeros(len(bounds))
        for i,b in enumerate(bounds):
#            if b[0] > 0 and b[1] > 0:
#                b0 = np.log(b[0])
#                b1 = np.log(b[1])
#                s[i] = np.exp(np.random.uniform(low=b0,high=b1))
#            else:
            b0,b1 = b
            if np.isinf(b[0]):
                b0 = np.sign(b[0])*1e5
            if np.isinf(b[1]):
                b1 = np.sign(b[1])*1e5
            s[i] = np.random.uniform(low=b0,high=b1)
        return s

class SumKernel(KernelND):
    def __init__(self,kernels):
        kernels = list(kernels)
        assert len(kernels) == 2
        self.kernels = list(kernels)
        super(SumKernel,self).__init__(self.kernels[0].ndims,no_create=True)
    @property
    def kernels(self):
        return self._kernels
    @kernels.setter
    def kernels(self,kernels):
        for K in kernels:
            if not isinstance(K,KernelND):
                raise TypeError("Only add KernelND, {}".format(type(K)))
            assert K.ndims == kernels[0].ndims, "Only add like dim kernels"
        self._kernels = kernels

    def __repr__(self):
        s = "**************\n"
        s += self.kernels[0].__repr__()
        s += "** + **\n"
        s += self.kernels[1].__repr__()
        s += "**************\n"

        return s

    def __call__(self,X,Y=None,eval_gradient=False):
        if eval_gradient:
            K0,grad0 = self.kernels[0](X,Y,True)
            K1,grad1 = self.kernels[1](X,Y,True)
            return K0+K1, np.concatenate([grad0,grad1],axis=0)
        else:
            K0 = self.kernels[0](X,Y,False)
            K1 = self.kernels[1](X,Y,False)
            return K0+K1
    @property
    def hyperparams(self):
        return np.concatenate([self.kernels[0].hyperparams,self.kernels[1].hyperparams],axis=0)
    @hyperparams.setter
    def hyperparams(self,hp):
        self.kernels[0].hyperparams = hp[:len(self.kernels[0].hyperparams)]
        self.kernels[1].hyperparams = hp[len(self.kernels[0].hyperparams):]
    @property
    def hyperparams_bounds(self):
        bounds = []
        for kernel in self.kernels:
            for b in kernel.hyperparams_bounds:
                bounds.append(b)
        return bounds
    @property
    def fixed(self):
        fixed = list(self.kernels[0].fixed) + list(self.kernels[1].fixed)
        return fixed
    @fixed.setter
    def fixed(self,name):
        '''Setting None will clear the fixed'''
        raise RuntimeError("Only use fixed on base kernel objects")

class ProductKernel(KernelND):
    def __init__(self,kernels):
        kernels = list(kernels)
        assert len(kernels) == 2
        self.kernels = list(kernels)
        super(ProductKernel,self).__init__(self.kernels[0].ndims,no_create=True)
    @property
    def kernels(self):
        return self._kernels
    @kernels.setter
    def kernels(self,kernels):
        for K in kernels:
            if not isinstance(K,KernelND):
                raise TypeError("Only multiply KernelND, {}".format(type(K)))
            assert K.ndims == kernels[0].ndims, "Only multiply like dim kernels"
        self._kernels = kernels
    def __repr__(self):
        s = self.kernels[0].__repr__()
        s += "** x **\n"
        s += self.kernels[1].__repr__()
        return s

    def __call__(self,X,Y=None,eval_gradient=False):
        '''d/dxi K0*K1 = (d/dxi K0)*K1 + (d/dxi K1)*K0'''
        if eval_gradient:
            K0,grad0 = self.kernels[0](X,Y,True)
            K1,grad1 = self.kernels[1](X,Y,True)
            return K0*K1, np.concatenate([grad0*K1,grad1*K0],axis=0)
        else:
            K0 = self.kernels[0](X,Y,False)
            K1 = self.kernels[1](X,Y,False)
            return K0*K1

    @property
    def hyperparams(self):
        return np.concatenate([self.kernels[0].hyperparams,self.kernels[1].hyperparams],axis=0)
    @hyperparams.setter
    def hyperparams(self,hp):
        self.kernels[0].hyperparams = hp[:len(self.kernels[0].hyperparams)]
        self.kernels[1].hyperparams = hp[len(self.kernels[0].hyperparams):]
    @property
    def hyperparams_bounds(self):
        bounds = []
        for kernel in self.kernels:
            for b in kernel.hyperparams_bounds:
                bounds.append(b)
        return bounds
    @property
    def fixed(self):
        fixed = list(self.kernels[0].fixed) + list(self.kernels[1].fixed)
        return fixed
    @fixed.setter
    def fixed(self,name):
        '''Setting None will clear the fixed'''
        raise RuntimeError("Only use fixed on base kernel objects")




class GammaExponential(KernelND):
    def __init__(self,ndims,l=1.,gamma=1.,sigma=1.):
        super(GammaExponential,self).__init__(ndims)
        self.hyperparams = [l,gamma,sigma]
        self.hyperparams_bounds = [[1e-5,1e5],[1e-5,2.],[1e-5,1e5]]
    def create(self):
        l,gamma,sigma,r = symbols('l gamma sigma r')
        self.symbols = [l,gamma,sigma]
        self.positional_symbols = [r]
        self.symbolic_func = sigma**Rational(2)*exp(-(r/l)**gamma)
        super(GammaExponential,self).create()

    def eval_positional_args(self,X,Y=None):
        if Y is None:
            dist = squareform(pdist(X,metric='euclidean'))#r
        else:
            dist = cdist(X,Y,metric='euclidean')#r
        return [dist]
                
#class MaternNuIso(Kernel2D):
#    def __init__(self,l,nu,sigma):
#        super(MaternNuIso,self).__init__()
#        self.hyperparams = [l,nu,sigma]
#    def create(self):
#        l,nu,sigma,r = symbols('l nu sigma r')
#        self.symbols = [l,nu,sigma]
#        from sympy.functions.special.gamma_functions import gamma
#        from sympy.functions.special.bessel import besselk
#        from scipy.special import kv
#        from scipy.special import gamma as gamma_scipy
#        func = Rational(2)**(Rational(1) - nu)/gamma(nu) * (sqrt(Rational(2)*nu)*r/(l+1e-15))**nu * besselk(nu,sqrt(Rational(2)*nu)*r/(l+1e-15))
#        self.func = lambdify((l,nu,sigma,r) ,func,modules=['numpy',{'gamma':gamma_scipy,'besselk':kv}])
#        func_diff = Matrix([func.diff(s) for s in self.symbols])
#        self.func_diff = lambdify((l,nu,sigma,r) ,func_diff,modules=['numpy'])
#    def hyperparams_bounds(self):
#        return [[0.,np.inf],[0.,np.inf],[0.,np.inf]]
#    def eval(self,x,y):
#        dx2 = np.subtract.outer(x,x)
#        dx2 *= dx2
#        dy2 = np.subtract.outer(y,y)
#        dy2 *= dy2
#        dy2 += dx2
#        self.r = np.sqrt(dy2)
#        arg = [hp for hp in self.hyperparams]
#        arg.append(self.r)
#        res = self.func(*arg)
#        return res
#
#    def eval_derivative(self,x,y):
#        if x is None or y is None:
#            r = self.r
#        else:
#            dx2 = np.subtract.outer(x,x)
#            dx2 *= dx2
#            dy2 = np.subtract.outer(y,y)
#            dy2 *= dy2
#            dy2 += dx2
#            self.r = np.sqrt(dy2)
#        arg = [hp for hp in self.hyperparams]
#        arg.append(self.r)
#        res = self.func_diff(*arg)
#        return res[:,0,:,:]

class MaternPSep(KernelND):
    def __init__(self,ndims,dim, l=1.,sigma=1.,p=1):
        self.p = int(p)#nu = p + 1/2
        super(MaternPSep,self).__init__(ndims)
        assert dim < ndims and dim >= 0
        self.dim = int(dim)
        self.hyperparams = [l,sigma]
        self.hyperparams_bounds = [[1e-5,1e5],[1e-5,1e5]]
        

    def create(self):
        l,sigma,r = symbols('l sigma r')
        self.symbols = [l,sigma]
        self.positional_symbols = [r]
        nu = Rational(self.p) + Rational(1,2)
        self.symbolic_func = Rational(0)
        for i in range(self.p+1):
            self.symbolic_func += factorial(Rational(self.p+i))/factorial(Rational(i))/factorial(Rational(self.p-i)) * (sqrt(Rational(8)*nu) * r / (l))**Rational(self.p-i)
        self.symbolic_func *= sigma**Rational(2)*exp(-sqrt(Rational(2)*nu)*r/(l)) * factorial(Rational(self.p))/factorial(Rational(2*self.p))
        super(MaternPSep,self).create()

    def eval_positional_args(self,X,Y=None):
        if Y is None:
            dist = squareform(pdist(X[:,[self.dim]],metric='euclidean'))#r
        else:
            dist = cdist(X[:,[self.dim]],Y[:,[self.dim]],metric='euclidean')#r
        return [dist]


class MaternPIso(KernelND):
    def __init__(self,ndims,l=1.,sigma=1.,p=1):
        self.p = int(p)#nu = p + 1/2
        super(MaternPIso,self).__init__(ndims)
        self.hyperparams = [l,sigma]
        self.hyperparams_bounds = [[1e-5,1e5],[1e-5,1e5]]
        

    def create(self):
        l,sigma,r = symbols('l sigma r')
        self.symbols = [l,sigma]
        self.positional_symbols = [r]
        nu = Rational(self.p) + Rational(1,2)
        self.symbolic_func = Rational(0)
        for i in range(self.p+1):
            self.symbolic_func += factorial(Rational(self.p+i))/factorial(Rational(i))/factorial(Rational(self.p-i)) * (sqrt(Rational(8)*nu) * r / (l))**Rational(self.p-i)
        self.symbolic_func *= sigma**Rational(2)*exp(-sqrt(Rational(2)*nu)*r/(l)) * factorial(Rational(self.p))/factorial(Rational(2*self.p))
        super(MaternPIso,self).create()

    def eval_positional_args(self,X,Y=None):
        if Y is None:
            dist = squareform(pdist(X,metric='euclidean'))#r
        else:
            dist = cdist(X,Y,metric='euclidean')#r
        return [dist]

class SquaredExponential(KernelND):
    def __init__(self,ndims,l=1.,sigma=1.):
        super(SquaredExponential,self).__init__(ndims)
        self.hyperparams = [l,sigma]
        self.hyperparams_bounds = [[1e-5,1e5],[1e-5,1e5]]
    def create(self):
        l,sigma,r2 = symbols('l sigma r2')
        self.symbols = [l,sigma]
        self.positional_symbols = [r2]
        self.symbolic_func = sigma**Rational(2)*exp(-(r2/(l)**Rational(2))/Rational(2))
        super(SquaredExponential,self).create()

    def eval_positional_args(self,X,Y=None):
        if Y is None:
            dist = squareform(pdist(X,metric='sqeuclidean'))#r2
        else:
            dist = cdist(X,Y,metric='sqeuclidean')#r2
        return [dist]

class PeriodicSep(KernelND):
    def __init__(self,ndims,dim,p=1.,l=1.,sigma=1.):
        super(PeriodicSep,self).__init__(ndims)
        assert dim < ndims and dim >= 0
        self.dim = int(dim)
        self.hyperparams = [p,l,sigma]
        self.hyperparams_bounds = [[1e-5,1e5],[1e-5,1e5],[1e-5,1e5]]
    def create(self):
        p,l,sigma,dx = symbols('p l sigma dx')
        self.symbols = [p,l,sigma]
        self.positional_symbols = [dx]
        self.symbolic_func = sigma**Rational(2)*exp(-Rational(2)*(sin(pi*sqrt(dx)/p)/l)**Rational(2))
        super(PeriodicSep,self).create()
    def eval_positional_args(self,X,Y=None):
        if Y is None:
            dist = squareform(pdist(X[:,[self.dim]],metric='euclidean'))#r
        else:
            dist = cdist(X[:,[self.dim]],Y[:,[self.dim]],metric='euclidean')#r
        return [dist]

    
class SquaredExponentialSep(KernelND):
    def __init__(self,ndims,dim,l=1.,sigma=1.):
        super(SquaredExponentialSep,self).__init__(ndims)
        assert dim < ndims and dim >= 0
        self.dim = int(dim)
        self.hyperparams = [l,sigma]
        self.hyperparams_bounds = [[1e-5,1e5],[1e-5,1e5]]
    def create(self):
        l,sigma,dx2 = symbols('l sigma dx2')
        self.symbols = [l,sigma]
        self.positional_symbols = [dx2]
        self.symbolic_func = sigma**Rational(2)*exp(-dx2/l**Rational(2)/Rational(2))
        super(SquaredExponentialSep,self).create()
    def eval_positional_args(self,X,Y=None):
        if Y is None:
            dist = squareform(pdist(X[:,[self.dim]],metric='sqeuclidean'))#r2
        else:
            dist = cdist(X[:,[self.dim]],Y[:,[self.dim]],metric='sqeuclidean')#r2
        return [dist]

class Diagonal(KernelND):
    def __init__(self,ndims,sigma=1.):
        super(Diagonal,self).__init__(ndims)
        self.hyperparams = [sigma]
        self.hyperparams_bounds = [[1e-5,1e5]]
    def create(self):
        sigma,I = symbols('sigma I')
        self.symbols = [sigma]
        self.positional_symbols = [I]
        self.symbolic_func = sigma**Rational(2)*I
        super(Diagonal,self).create()
    def eval_positional_args(self,X,Y=None):
        if Y is None:
            return [np.eye(X.shape[0])]
        else:
            dist = cdist(X,Y,metric='sqeuclidean')#r2
            mask = dist == 0.
            dist *= 0.
            dist[mask] = 1.
            return [dist]


class RationalQuadratic(KernelND):
    def __init__(self,ndims,l=1.,alpha=1.,sigma=1.):
        super(RationalQuadratic,self).__init__(ndims)
        self.hyperparams = [l,alpha,sigma]
        self.hyperparams_bounds = [[1e-5,1e5],[1e-5,1e5],[1e-5,1e5]]
    def create(self):
        l,alpha,sigma,r2 = symbols('l alpha sigma r2')
        self.symbols = [l,alpha,sigma]
        self.positional_symbols = [r2]
        self.symbolic_func = sigma**Rational(2)*(Rational(1) + r2/Rational(2)/(alpha)/(l)**2)**(-alpha)
        super(RationalQuadratic,self).create()
        
    def eval_positional_args(self,X,Y=None):
        if Y is None:
            dist = squareform(pdist(X,metric='sqeuclidean'))#r2
        else:
            dist = cdist(X,Y,metric='sqeuclidean')#r2
        return [dist]

class RationalQuadraticSep(KernelND):
    def __init__(self,ndims,dim,l=1.,alpha=1.,sigma=1.):
        super(RationalQuadraticSep,self).__init__(ndims)
        assert dim < ndims and dim >= 0
        self.dim = int(dim)
        self.hyperparams = [l,alpha,sigma]
        self.hyperparams_bounds = [[1e-5,1e5],[1e-5,1e5],[1e-5,1e5]]
    def create(self):
        l,alpha,sigma,dx2 = symbols('l alpha sigma dx2')
        self.symbols = [l,alpha,sigma]
        self.positional_symbols = [dx2]
        self.symbolic_func = sigma**Rational(2)*(Rational(1) + dx2/Rational(2)/(alpha)/(l)**2)**(-alpha)
        super(RationalQuadraticSep,self).create()

    def eval_positional_args(self,X,Y=None):
        if Y is None:
            dist = squareform(pdist(X[:,[self.dim]],metric='sqeuclidean'))#r2
        else:
            dist = cdist(X[:,[self.dim]],Y[:,[self.dim]],metric='sqeuclidean')#r2
        return [dist]

class DotProduct(KernelND):
    """The linear kernel or else dot-product kernel.
    k(x,x') = sigma_b^2 + sigma_v^2 (x - c).(x' - c).
    Equivalent to linear regression with N(0,sigma_v) prior on coeffs
    and N(0,sigma_b) on bias"""
    def __init__(self,ndims,c=0., sigma_b = 1. , sigma_v = 1.):
        super(DotProduct,self).__init__(ndims)
        self.hyperparams = [c, sigma_b, sigma_v]
        self.hyperparams_bounds = [[-np.inf, np.inf],[1e-5,1e5],[1e-5,1e5]]
    def create(self):
        c, sigma_b, sigma_v, x2, x_1, x_2, I = symbols('c sigma_b sigma_v x2 x_1 x_2 I')
        self.symbols = [c, sigma_b, sigma_v]
        self.positional_symbols = [x2, x_1, x_2, I]
        self.symbolic_func = I*sigma_b**Rational(2) + sigma_v**Rational(2) * (x2 - c*x_1 - c*x_2 + c**Rational(2))
        super(DotProduct,self).create()

    def eval_positional_args(self,X,Y=None):
        if Y is None:
            x2 = np.inner(X,X)
            x_1 = np.inner(X,np.ones_like(X))
            x_2 = x_1.T
            I = np.ones_like(x2)
        else:
            x2 = np.inner(X,Y)
            x_1 = np.inner(X,np.ones_like(Y))
            x_2 = np.inner(np.ones_like(X), Y)
            I = np.ones_like(x2)
        return [x2,x_1,x_2,I]

def solve_equation(K,y):
    L = np.linalg.cholesky(K)
    alpha = cho_solve((L,True),y)
    return alpha

def is_singular(A):
    return np.linalg.cond(A) > 1/sys.float_info.epsilon

def level1_solve(x,y,sigma_y,xstar,K):
    if np.size(sigma_y) == 1:
        sigma_y = np.ones_like(y)*sigma_y
    else:
        assert sigma_y.shape == y.shape
    xconc = np.concatenate([x,xstar],axis=0)
    K_matrix = K(xconc)
    n = x.shape[0]
    m = xstar.shape[0]
    Knn = K_matrix[:n,:n]
    Knm = K_matrix[:n,n:n+m]
    Kmm = K_matrix[n:n+m,n:n+m]
    Kf = Knn + np.diag(sigma_y)
    try:
        L = np.linalg.cholesky(Kf)
        alpha = cho_solve((L,True),y)
        fstar = Knm.T.dot(alpha)
        V = cho_back_substitution(L,Knm,lower=True,modify=True)
        cov = Kmm - V.T.dot(V)
        log_mar_like = -y.dot(alpha)/2. - np.sum(np.log(np.diag(L))) - n/2.*np.log(2.*np.pi)
    except:
        u,s,v = np.linalg.svd(Kf)
        sinv = 1./s
        sinv[s < 1e-14] == 0.
        sinv = np.diag(sinv)
        Kinv = v.T.dot(sinv).dot(u.T)
        det = np.prod(s[s>1e-14])
        alpha = Kinv.dot(y)
        fstar = Knm.T.dot(alpha)
        cov = Kmm - Knm.T.dot(Kinv).dot(Knm)
        log_mar_like = - y.dot(alpha)/2. - np.log(det)/2. - n*np.log(2.*np.pi)/2.

    return fstar,cov,log_mar_like

def neg_log_mar_like_and_derivative(hyperparams,x,y,sigma_y,K):
    if np.size(sigma_y) == 1:
        sigma_y = np.ones_like(y)*sigma_y
    else:
        assert sigma_y.shape == y.shape
    K.hyperparams = hyperparams
    K_matrix, K_diff = K(x,eval_gradient=True)
    n = x.shape[0]
    Kf = K_matrix + np.diag(sigma_y)
    #sing = is_singular(Kf)
    try:#if pos_def and not sing:
        L = np.linalg.cholesky(Kf)
        alpha = cho_solve((L,True),y)
        log_mar_like = -y.dot(alpha)/2. - np.sum(np.log(np.diag(L))) - n/2.*np.log(2.*np.pi)
        grad = np.zeros(len(hyperparams))
        aa = np.outer(alpha,alpha)    
        for i in range(len(hyperparams)):
            aaK = aa.dot(K_diff[i,:,:])
            KK = cho_solve((L,True),K_diff[i,:,:])
            grad[i] = (np.trace(aaK) - np.trace(KK))/2.
    except:#else:
        u,s,v = np.linalg.svd(Kf)
        sinv = 1./s
        sinv[s < 1e-14] == 0.
        sinv = np.diag(sinv)
        Kinv = v.T.dot(sinv).dot(u.T)
        det = np.prod(s[s>1e-14])
        alpha = Kinv.dot(y)
        log_mar_like = - y.dot(alpha)/2. - np.log(det)/2. - n*np.log(2.*np.pi)/2.
        grad = np.zeros(len(hyperparams))
        aa = np.outer(alpha,alpha)   
        for i in range(len(hyperparams)):
            aaK = aa.dot(K_diff[i,:,:])
            KK = Kinv.dot(K_diff[i,:,:])
            grad[i] = (np.trace(aaK) - np.trace(KK))/2.
    if np.isinf(log_mar_like) or np.isnan(log_mar_like):
        log_mar_like = 0.
    grad[np.isnan(grad)] = 0.
    grad[np.isinf(grad)] = 0.
    #print(log_mar_like, grad, hyperparams)

    return -log_mar_like,-grad

def log_mar_like(hyperparams,x,y,sigma_y,K):
    if np.size(sigma_y) == 1:
        sigma_y = np.ones_like(y)*sigma_y
    else:
        assert sigma_y.shape == y.shape
    K.hyperparams = hyperparams
    K_matrix = K(x)
    n = x.shape[0]
    Kf = K_matrix + np.diag(sigma_y)
    try:
        L = np.linalg.cholesky(Kf)
        alpha = cho_solve((L,True),y)
        log_mar_like = -y.dot(alpha)/2. - np.sum(np.log(np.diag(L))) - n/2.*np.log(2.*np.pi)
    except:
        u,s,v = np.linalg.svd(Kf)
        sinv = 1./s
        sinv[s < 1e-14] == 0.
        sinv = np.diag(sinv)
        Kinv = v.T.dot(sinv).dot(u.T)
        det = np.prod(s[s>1e-14])
        alpha = Kinv.dot(y)
        log_mar_like = - y.dot(alpha)/2. - np.log(det)/2. - n*np.log(2.*np.pi)/2.

    return log_mar_like




def level2_solve(x,y,sigma_y,K,n_random_start=0):
    '''Solve the ML covariance hyperparameters based on data and gradient descent.
    x is shape (num_points,2), y is shape(num_points,)'''
#    #sample randomly first
#    hp_ = np.array(K.hyperparams)
#    lml_ = log_mar_like(hp_,x,y,sigma_y,K)
#    for i in range(1000):
#        hp = K.sample_hyperparameters()
#        K.hyperparams = hp
#        lml = log_mar_like(hp,x,y,sigma_y,K)
#        if lml > lml_:
#            lml_ = lml
#            hp_ = hp
#    K.hyperparams = hp_
#    print(K)
    bounds = K.hyperparams_bounds
    res = fmin_l_bfgs_b(neg_log_mar_like_and_derivative,K.hyperparams,None,args=(x,y,sigma_y,K),bounds=bounds)
    param_min = res[0]
    fmin = res[1]

    for n in range(n_random_start):
        K.hyperparams = K.sample_hyperparameters()
        res = fmin_l_bfgs_b(neg_log_mar_like_and_derivative,K.hyperparams,None,args=(x,y,sigma_y,K),bounds=bounds)
        if res[1] < fmin and res[2]['warnflag'] == 0:
            param_min = res[0]
            fmin = res[1]

    hyperparams = param_min
    log.info("Final log marginal likelihood {}".format(-fmin))
    return hyperparams

def level2_multidataset_solve(x,y,sigma_y,K,n_random_start=0):
    '''x is a list of arrays of shape (num_points_i,2) where i is the i-th dataset.
    y is list of shape (num_points_i,)
    sigma_y is array of (num_points)'''
    def neg_log_mar_like_and_derivative_multidataset(hyperparams,x,y,sigma_y,K):
        ngrad = np.zeros(len(hyperparams))
        nlml = 0.
        for xi, yi,sigma_yi in zip(x,y,sigma_y):
            nlml_, ngrad_ = neg_log_mar_like_and_derivative(hyperparams,xi,yi,sigma_yi,K)
            nlml += nlml_
            ngrad += ngrad_
        return nlml,ngrad
    bounds = K.hyperparams_bounds
    res = fmin_l_bfgs_b(neg_log_mar_like_and_derivative_multidataset,
            K.hyperparams,None,
            args=(x,y,sigma_y,K),bounds=bounds)
    param_min = res[0]
    fmin = res[1]

    for n in range(n_random_start):
        K.hyperparams = K.sample_hyperparameters()
        res = fmin_l_bfgs_b(neg_log_mar_like_and_derivative_multidataset,K.hyperparams,None,args=(x,y,sigma_y,K),bounds=bounds)
        if res[1] < fmin and res[2]['warnflag'] == 0:
            param_min = res[0]
            fmin = res[1]

    hyperparams = param_min
    log.info("Final log marginal likelihood {}".format(-fmin))

    return hyperparams
    
def example_level2_solve():
    np.random.seed(1234)
    K1 = SquaredExponential(2,l=0.29,sigma=3.7)
    #K1.fixed = 'l'
    #K1.fixed = 'sigma'
    K2 = Diagonal(2,sigma=1e-5)
    K2.fixed = 'sigma'
    K3 = RationalQuadratic(2,sigma=1.)
    K4 = MaternPIso(2,p=2)
    K6 = GammaExponential(2)
    K7 = PeriodicSep(2,0,l=0.5)
    K7.fixed = 'l'
    K8 = PeriodicSep(2,1,l=0.5)
    K8.fixed='l'
    K9 = DotProduct(2)
    K = K3 *K1*K9+K2
    hp = K.hyperparams 
    x = np.random.uniform(size=[250,2])
    xstar = np.linspace(-1,2,100)
    Xstar,Ystar = np.meshgrid(xstar,xstar)
    xstar = np.array([Xstar.flatten(),Ystar.flatten()]).T
    y = np.sin(x[:,0]*2*np.pi/0.5) *np.cos( x[:,1]*np.pi/0.5*2.) + np.random.normal(size=x.shape[0])*0.1
    m_y = np.mean(y)
    y -= m_y
    sigma_y = 0.1
    hyperparams = level2_solve(x,y,sigma_y,K,n_random_start=10)
    K.hyperparams = hyperparams
    print(K)
    fstar,cov,log_mar_like = level1_solve(x,y,sigma_y,xstar,K)
    import pylab as plt
    vmin = np.min(y) + m_y
    vmax = np.max(y) + m_y
    plt.imshow(fstar.reshape(Xstar.shape)+m_y,extent=(-1,2,-1,2),origin='lower',vmin=vmin,vmax=vmax)
    plt.scatter(x[:,0],x[:,1],c=y+m_y)
    #plt.scatter(xstar[:,0],xstar[:,1],c=fstar,marker='+')
    plt.show()

    
if __name__=='__main__':
    example_level2_solve()

