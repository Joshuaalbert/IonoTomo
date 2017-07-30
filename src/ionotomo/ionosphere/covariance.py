'''Handle covariance operations'''
import numpy as np
from scipy.special import gamma

class Covariance(object):
    '''Use for repeated use of covariance.
    `TCI` : `tomoiono.geometry.tri_cubic.TriCubic`
        TricubicCubic interpolator that contains the geometry of the volume
    `sigma` : `float`
    variance of diagonal terms (sigma_1)
    `corr` : `float`
        correlation length of covariance (sigma_2)
    `nu` : `float`
        smoothness parameter 1./2. results in exponential, 3./2. to 7./2. more 
        smooth realistic ionosphere, as nu -> inf it approaches 
        square-exponential covariance (too smooth to be real)'''
    def __init__(self,TCI,sigma,corr,nu):
        self.sigma = sigma
        self.corr = corr
        self.nu = nu
        self.nx = TCI.nx
        self.ny = TCI.ny
        self.nz = TCI.nz
        self.dx = TCI.xvec[1] - TCI.xvec[0]
        lvec = np.fft.fftfreq(TCI.nx,d=self.dx)
        self.dy = TCI.yvec[1] - TCI.yvec[0]
        mvec = np.fft.fftfreq(TCI.ny,d=self.dy)
        self.dz = TCI.zvec[1] - TCI.zvec[0]
        self.dV = self.dx*self.dy*self.dz
        nvec = np.fft.fftfreq(TCI.nz,d=self.dz)
        L,M,N = np.meshgrid(lvec,mvec,nvec,indexing='ij')
        self.r = L**2
        self.r += M**2
        self.r += N**2
        np.sqrt(self.r,out=self.r)
        n = 3.
        self.f  = sigma**2*2**(n) * np.pi**(n/2.) * gamma(nu+n/2.) * (2*nu)**(nu) / gamma(nu) / corr**(2*nu) * (2*nu/corr**2 + 4*np.pi**2*self.r**2)**(-nu - n/2.)

    def realization(self):
        '''Generate a Gaussian Random field with given covariance'''
        B = np.random.normal(size=[self.nx,self.ny, self.nz])
        A = np.fft.fftn(B)
        A *= np.sqrt(self.f/self.dV)/4.
        B = (np.fft.ifftn(A)).real
        return B
    
    def contract(self,phi):
        '''Do Cm^{-1}.phi using ffts'''
        Phi = np.fft.fftn(phi)
        Phi /= self.f
        #factor which lines up with theory. Not sure where it comes from
        #Phi /= 2.
        #Phi /= self.dV*4.
        phihat = (np.fft.ifftn(Phi)).real
        return phihat
