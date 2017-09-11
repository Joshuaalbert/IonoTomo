'''Handle covariance operations'''
import numpy as np
from scipy.special import gamma
from ionotomo.utils.gaussian_process import *

class Covariance(object):
    '''Use for repeated use of covariance.
    `tci` : `tomoiono.geometry.tri_cubic.TriCubic`
        TricubicCubic interpolator that contains the geometry of the volume
    `sigma` : `float`
    variance of diagonal terms (sigma_1)
    `corr` : `float`
        correlation length of covariance (sigma_2)
    `nu` : `float`
        smoothness parameter 1./2. results in exponential, 3./2. to 7./2. more 
        smooth realistic ionosphere, as nu -> inf it approaches 
        square-exponential covariance (too smooth to be real)'''
    def __init__(self,K=None, tci = None):
        if K is None:
            self.K = MaternPSep(3,0,l=30.,sigma=1.,p=2)*MaternPSep(3,1,l=30.,sigma=1.,p=2)*MaternPSep(3,2,l=30.,sigma=1.,p=2) + Diagonal(3,sigma=1.)
        else:
            assert isinstance(K,KernelND)
            assert K.ndims == 3
            self.K = K
        if tci is not None:
            xvec = tci.xvec - np.mean(tci.xvec)
            yvec = tci.yvec - np.mean(tci.yvec)
            zvec = tci.zvec - np.mean(tci.zvec)
            dx = xvec[1] - xvec[0]
            dy = yvec[1] - yvec[0]
            dz = zvec[1] - zvec[0]
            self.dV = dx*dy*dz
            self.nx,self.ny,self.nz = tci.M.shape
            self.N = np.prod(tci.M.shape)
            x,y,z = np.meshgrid(xvec,yvec,zvec,indexing='ij')
            x = x.flatten()
            y = y.flatten()
            z = z.flatten()
            X = np.array([x,y,z]).T
            self.K_kernel = self.K(np.zeros([1,3]),X).reshape(tci.M.shape)
            print(np.any(self.K_kernel <= 0))
            import pylab as plt
            plt.imshow(self.K_kernel[:,50,:])
            plt.show()
            self.spectral_density = np.fft.fftn(self.K_kernel,norm='ortho')
            print(np.min(self.spectral_density),np.max(self.spectral_density))
        else:
            self.K_kernel = None
            self.spectral_density = None
#            self.dx = tci.xvec[1] - tci.xvec[0]
#            lvec = np.fft.fftfreq(tci.nx,d=self.dx)
#            self.dy = tci.yvec[1] - tci.yvec[0]
#            mvec = np.fft.fftfreq(tci.ny,d=self.dy)
#            self.dz = tci.zvec[1] - tci.zvec[0]
#            self.dV = self.dx*self.dy*self.dz
#            nvec = np.fft.fftfreq(tci.nz,d=self.dz)
#            L,M,N = np.meshgrid(lvec,mvec,nvec,indexing='ij')
#            self.r = L**2

    def __call__(self,X,Y=None):
        '''Return the covariance between all pairs in X or between X and Y if Y is not None.
        X : (M1, self.ndim)
        Y : (M2, self.ndim)
        Return : (M1, M2)'''
        return self.K(X,Y=Y)

    def realization(self):
        '''Generate a Gaussian Random field with given covariance'''
        assert self.spectral_density is not None
        B = np.random.normal(size=[self.nx,self.ny, self.nz])
        A = np.fft.fftn(B,norm='ortho')
        A *= np.sqrt(self.spectral_density)
        B = np.fft.ifftn(A,norm='ortho').real
        return B
    
    def contract(self,phi):
        '''Do Cm^{-1}.phi using ffts'''
        Phi = np.fft.fftn(phi,norm='ortho')
        Phi /= (self.spectral_density + 1e-15)

        #factor which lines up with theory. Not sure where it comes from
        #Phi /= 2.
        #Phi /= self.dV*4.
        phihat = np.fft.ifftn(Phi,norm='ortho').real/self.dV
        return phihat
