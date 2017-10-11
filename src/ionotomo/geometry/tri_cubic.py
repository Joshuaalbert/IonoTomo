'''Based on the paper doi=10.1.1.89.7835
the tricubic interpolation of a regular possibly non uniform grid can be seen as a computation of 21 cubic splines.
A cubic spline is a special case of cubic interpolation, and in general these 21 cubic splines perform many redundant
calulations. Here we formulate the full tricubic interpolation where the value of a scalar function defined on a 3d
grid can be reconstructed to allow full C1, and thus langrangian structures to persist.'''

import numpy as np
import h5py

from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import simps

class TriCubic(object):
    def __init__(self,xvec=None,yvec=None,zvec=None,M=None,filename=None):
        if filename is not None:
            self.load(filename)
        else:
            self.xvec = xvec
            self.yvec = yvec
            self.zvec = zvec
            self.M = M
            self.rgi = RegularGridInterpolator((self.xvec,self.yvec,self.zvec),self.M, bounds_error=True)
    
    @property
    def xvec(self):
        return self._xvec
    @xvec.setter
    def xvec(self,val):
        self._xvec = np.array(val)
        self.nx = int(np.size(self.xvec))
    @property
    def yvec(self):
        return self._yvec
    @yvec.setter
    def yvec(self,val):
        self._yvec = np.array(val)
        self.ny = int(np.size(self.yvec))
    @property
    def zvec(self):
        return self._zvec
    @zvec.setter
    def zvec(self,val):
        self._zvec = np.array(val)
        self.nz = int(np.size(self.zvec))
    @property
    def M(self):
        return self._M
        
    @M.setter
    def M(self,val):
        assert not np.any(np.isnan(val)) and not np.any(np.isinf(val))
        if len(val.shape) == 1:
            self.M = val.reshape((len(self.xvec),len(self.yvec),len(self.zvec)))
            return
        assert val.shape[0] == len(self.xvec)
        assert val.shape[1] == len(self.yvec)
        assert val.shape[2] == len(self.zvec)
        self._M = val
        self.rgi = RegularGridInterpolator((self.xvec,self.yvec,self.zvec),self.M,bounds_error=True)  

    def inner(self,M,inplace=False):
        '''Do the inner product of self.M with M'''
        if not inplace:
            M = M*self.M
        else:
            M *= self.M
        return simps(simps(simps(M,self.zvec,axis=2),self.yvec,axis=1),self.xvec,axis=0)
        
    def interp(self,x,y,z):
        return np.reshape(self.rgi(np.array([x,y,z]).T),np.shape(x))
    def extrapolate(self,x,y,z):
        self.rgi = RegularGridInterpolator((self.xvec,self.yvec,self.zvec),self.M,bounds_error=False,fill_value=None)
        res = self.interp(x,y,z)
        self.rgi = RegularGridInterpolator((self.xvec,self.yvec,self.zvec),self.M,bounds_error=True)
        return res

    def copy(self,**kwargs):
        '''Return a copy of the TriCubic object by essentially creating a copy of all the data. 
        ``kwargs`` are the same as constructor.'''
        return TriCubic(self.xvec.copy(),self.yvec.copy(),self.zvec.copy(),self.M.copy(),**kwargs)
    def load(self,filename,**kwargs):
        f = h5py.File(filename,'r')
        xvec = f["TCI/xvec"][:]
        yvec = f["TCI/yvec"][:]
        zvec = f["TCI/zvec"][:]
        M = f["TCI/M"][:,:,:]
        self.__init__(xvec,yvec,zvec,M,**kwargs)
        f.close()
    def save(self,filename):
        f = h5py.File(filename,'w')
        xvec = f.create_dataset("TCI/xvec",(self.nx,),dtype=np.double)
        yvec = f.create_dataset("TCI/yvec",(self.ny,),dtype=np.double)
        zvec = f.create_dataset("TCI/zvec",(self.nz,),dtype=np.double)
        M = f.create_dataset("TCI/M",(self.nx,self.ny,self.nz),dtype=np.double)
        xvec[...] = self.xvec
        yvec[...] = self.yvec
        zvec[...] = self.zvec
        M[...] = self.M
        f.close()
    
    def get_model_coordinates(self):
        X,Y,Z = np.meshgrid(self.xvec,self.yvec,self.zvec,indexing='ij')
        return X.flatten(order='C'),Y.flatten(order='C'),Z.flatten(order='C')
       
def bisection(array,value):
    '''Given an ``array`` , and given a ``value`` , returns an index j such that ``value`` is between array[j]
    and array[j+1]. ``array`` must be monotonic increasing. j=-1 or j=len(array) is returned
    to indicate that ``value`` is out of range below and above respectively.'''
    #return bisection(array,value)
    n = len(array)
    if (value < array[0]):
        return -1
        res = -1# Then set the output
    elif (value > array[n-1]):
        return n
    #array = np.append(np.append(-np.inf,array),np.inf)
    jl = 0# Initialize lower
    ju = n-1# and upper limits.
    while (ju-jl > 1):# If we are not yet done,
        jm=(ju+jl) >> 1# compute a midpoint,
        if (value >= array[jm]):
            jl=jm# and replace either the lower limit
        else:
            ju=jm# or the upper limit, as appropriate.
        # Repeat until the test condition is satisfied.
    if (value == array[0]):
        return 0
        res = -1# Then set the output
    elif (value == array[n-1]):
        return n-1
    else:
        return jl

