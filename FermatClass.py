
# coding: utf-8

# In[ ]:

import numpy as np
from scipy.integrate import odeint
from TricubicInterpolation import TriCubic

class Fermat(object):
    def __init__(self,neTCI=None,frequency = 120e6,type='s',straightLineApprox=True):
        '''Fermat principle. type = "s" means arch length is the indepedent variable
        type="z" means z coordinate is the independent variable.'''
        self.type = type
        self.frequency = frequency#Hz
        self.straightLineApprox = straightLineApprox
        if neTCI is not None:
            self.ne2n(neTCI)  
            return
        
    def loadFunc(self,file):
        '''Load the model given in `file`'''
        data = np.load(file)
        if 'ne' in data.keys():
            ne = data['ne']
            xvec = data['xvec']
            yvec = data['yvec']
            zvec = data['zvec']
            self.ne2n(TriCubic(xvec,yvec,zvec,ne,useCache=True))
            return
        if 'n' in data.keys():
            ne = data['n']
            xvec = data['xvec']
            yvec = data['yvec']
            zvec = data['zvec']
            self.n2ne(TriCubic(xvec,yvec,zvec,n,useCache=True))
            return
    
    def saveFunc(self,file):
        np.savez(file,xvec=self.nTCI.xvec,yvec=self.nTCI.yvec,zvec=self.nTCI.zvec,n=self.nTCI.m,ne=self.neTCI.m)
            
    def ne2n(self,neTCI):
        '''Analytically turn electron density to refractive index. Assume ne in m^-3'''
        self.neTCI = neTCI
        #copy object
        self.nTCI = neTCI.copy(default=1.)
        #inplace change to refractive index
        self.nTCI.m *= -8.980**2/self.frequency**2
        self.nTCI.m += 1.
        self.nTCI.m = np.sqrt(self.nTCI.m)
        #wp = 5.63e4*np.sqrt(ne/1e6)/2pi#Hz^2 m^3 lightman p 226
        return self.nTCI
    
    def n2ne(self,nTCI):
        """Get electron density in m^-3 from refractive index"""
        self.nTCI = nTCI
        #convert to 
        self.neTCI = nTCI.copy()
        self.neTCI.m *= -self.neTCI.m
        self.neTCI.m += 1.
        self.neTCI.m *= self.frequency**2/8.980**2
        #wp = 5.63e4*np.sqrt(ne/1e6)/2pi#Hz^2 m^3 lightman p 226
        return self.neTCI
    
    def eulerODE(self,y,t,*args):
        '''return pxdot,pydot,pzdot,xdot,ydot,zdot,sdot'''
        #print(y)
        px,py,pz,x,y,z,s = y
        if self.straightLineApprox:
            n,nx,ny,nz = 1.,0,0,0
        else:
            n,nx,ny,nz,nxy,nxz,nyz,nxyz = self.nTCI.interp(x,y,z,doDiff=True)
        #from ne
        #ne,nex,ney,nez,nexy,nexz,neyz,nexyz = self.neTCI.interp(x,y,z,doDiff=True)
        #A = - 8.98**2/self.frequency**2
        #n = math.sqrt(1. + A*ne)
        #ndot = A/(2.*n)
        #nx = ndot * nex
        #ny = ndot * ney
        #nz = ndot * nez
        if self.type == 'z':
            sdot = n / pz
            pxdot = nx*n/pz
            pydot = ny*n/pz
            pzdot = nz*n/pz

            xdot = px / pz
            ydot = py / pz
            zdot = 1.
        
        if self.type == 's':
            sdot = 1.
            pxdot = nx
            pydot = ny
            pzdot = nz

            xdot = px / n
            ydot = py / n
            zdot = pz / n
        
        return [pxdot,pydot,pzdot,xdot,ydot,zdot,sdot]
    
    def jacODE(self,y,t,*args):
        '''return d ydot / d y, with derivatives down columns for speed'''
        px,py,pz,x,y,z,s = y
        if self.straightLineApprox:
            n,nx,ny,nz,nxy,nxz,nyz = 1.,0,0,0,0,0,0
        else:
            n,nx,ny,nz,nxy,nxz,nyz,nxyz = self.nTCI.interp(x,y,z,doDiff=True)
        #TCI only gaurentees C1 and C2 information is lost, second order anyways
        nxx,nyy,nzz = 0.,0.,0.
        #from electron density
        #ne,nex,ney,nez,nexy,nexz,neyz,nexyz = self.neTCI.interp(x,y,z,doDiff=True)
        #A = - 8.98**2/self.frequency**2
        #n = math.sqrt(1. + A*ne)
        #ndot = A/(2.*n)
        #nx = ndot * nex
        #ny = ndot * ney
        #nz = ndot * nez
        #ndotdot = -(A * ndot)/(2. * n**2)
        #nxy = ndotdot * nex*ney + ndot * nexy
        #nxz = ndotdot * nex * nez + ndot * nexz
        #nyz = ndotdot * ney * nez + ndot * neyz 
        if self.type == 'z':
            x0 = n
            x1 = nx
            x2 = pz**(-2)
            x3 = x0*x2
            x4 = 1./pz
            x5 = ny
            x6 = x4*(x0*nxy + x1*x5)
            x7 = nz
            x8 = x4*(x0*nxz + x1*x7)
            x9 = x4*(x0*nyz + x5*x7)
            jac = np.array([[ 0,  0, -x1*x3, x4*(x0*nxx + x1**2),x6, x8, 0.],
                            [ 0,  0, -x3*x5,x6, x4*(x0*nyy + x5**2), x9, 0.],
                            [ 0,  0, -x3*x7,x8, x9, x4*(x0*nzz + x7**2), 0.],
                            [x4,  0, -px*x2, 0, 0,  0, 0.],
                            [ 0, x4, -py*x2, 0, 0, 0, 0.],
                            [ 0,  0, 0, 0, 0, 0, 0.],
                            [ 0,  0,-x3,x1*x4, x4*x5, x4*x7, 0.]])
        
        if self.type == 's':
            x0 = n
            x1 = nxy
            x2 = nxz
            x3 = nyz
            x4 = 1./x0
            x5 = nx
            x6 = x0**(-2)
            x7 = px*x6
            x8 = ny
            x9 = nz
            x10 = py*x6
            x11 = pz*x6
            jac = np.array([[ 0,  0,  0, nxx, x1, x2, 0.],
                            [ 0,  0,  0, x1, nyy, x3, 0.],
                            [ 0,  0,  0, x2, x3, nzz, 0.],
                            [x4,  0,  0, -x5*x7, -x7*x8, -x7*x9, 0.],
                            [ 0, x4,  0, -x10*x5, -x10*x8, -x10*x9, 0.],
                            [ 0,  0, x4, -x11*x5, -x11*x8, -x11*x9, 0.],
                            [ 0,  0,  0, 0, 0, 0, 0.]])
        return jac
        
    def integrateRay(self,origin,direction,tmax,N=100):
        '''Integrate ray defined by the ``origin`` and ``direction`` along the independent variable (s or z)
        until tmax. 
        ``N`` - the number of partitions along the ray to save ray trajectory.'''
        x0,y0,z0 = origin
        xdot0,ydot0,zdot0 = direction
        sdot = np.sqrt(xdot0**2 + ydot0**2 + zdot0**2)
        #momentum
        px0 = xdot0/sdot
        py0 = ydot0/sdot
        pz0 = zdot0/sdot
        #px,py,pz,x,y,z,s
        init = [px0,py0,pz0,x0,y0,z0,0]
        if self.type == 'z':
            tarray = np.linspace(z0,tmax,N)
        if self.type == 's':
            tarray = np.linspace(0,tmax,N)
        Y,info =  odeint(self.eulerODE, init, tarray,Dfun = self.jacODE, col_deriv = True, full_output=1)
        #print(info['hu'].shape,np.sum(info['hu']),info['hu'])
        #print(Y)
        x = Y[:,3]
        y = Y[:,4]
        z = Y[:,5]
        s = Y[:,6]
        return x,y,z,s   


