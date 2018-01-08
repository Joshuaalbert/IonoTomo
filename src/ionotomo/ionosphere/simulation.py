""" This module is concerned with being able to realize realistic *enough* 
ionospheres. We wish to take to be able to take into account turbulent
structure"""
import numpy as np
from scipy.special import gamma, kv


class IonosphereSimulation_OLD(object):
    """Simulate a realisation with a Matern kernel and Chapmen layers.
    tci : TriCubic
        defines the volume
    sigma : the overall magnatude of kernel
    corr : the correlation length
    nu : smoothness"""
    def __init__(self,tci,sigma,corr, nu):
        self.sigma = sigma
        self.corr = corr
        self.nu = nu
        self.nx = tci.nx
        self.ny = tci.ny
        self.nz = tci.nz
        self.dx = tci.xvec[1] - tci.xvec[0]
        lvec = np.fft.fftfreq(tci.nx,d=self.dx)
        self.dy = tci.yvec[1] - tci.yvec[0]
        mvec = np.fft.fftfreq(tci.ny,d=self.dy)
        self.dz = tci.zvec[1] - tci.zvec[0]
        self.dV = self.dx*self.dy*self.dz
        nvec = np.fft.fftfreq(tci.nz,d=self.dz)
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

class IonosphereSimulation(object):
    """Simulate a realisation with a Matern kernel and Chapmen layers.
    The grid.
    sigma : the overall magnatude of kernel
    corr : the correlation length
    nu : smoothness"""
    def __init__(self,xvec,yvec,zvec,sigma,corr, type='m52'): 
        assert type in ['m52']
        self.nx = np.size(xvec)
        self.ny = np.size(yvec)
        self.nz = np.size(zvec)
        self.dx = xvec[1] - xvec[0]
        self.dy = yvec[1] - yvec[0]
        self.dz = zvec[1] - zvec[0]
        self.sigma = sigma
        self.corr = corr

        self.sx = 1./(self.dx*self.nx)
        self.sy = 1./(self.dy*self.ny)
        self.sz = 1./(self.dz*self.nz)

        lvec = np.linspace(0,self.sx*self.nx/2.,self.nx)
        mvec = np.linspace(0,self.sy*self.ny/2.,self.ny)
        nvec = np.linspace(0,self.sz*self.nz/2.,self.nz)


 
#        lvec = np.fft.ifftshift(np.fft.fftfreq(self.nx,d=1))
#        mvec = np.fft.ifftshift(np.fft.fftfreq(self.ny,d=1))
#        nvec = np.fft.ifftshift(np.fft.fftfreq(self.nz,d=1))[:self.nz>>1]
        L,M,N = np.meshgrid(lvec,mvec,nvec,indexing='ij')
        s2 = L**2
        s2 += M**2
        s2 += N**2
        s2 = np.fft.ifftshift(s2)
        s = np.sqrt(s2)
        self.type=type
        if self.type == 'm52':
            #self.S = self.sigma**2 * 8*np.sqrt(5)**5*np.sqrt(np.pi)**3*gamma(4.)/gamma(5./2.)/self.corr**5 / np.sqrt(np.sqrt(5./self.corr**2 + (4.* np.pi**2) * s2))
            n = 3.
            nu = 5/2.
            self.S  = self.sigma**2*2**(n) * np.pi**(n/2.) * gamma(nu+n/2.) * (2*nu)**(nu) / gamma(nu) / self.corr**(2*nu) * (2*nu/self.corr**2 + 4*np.pi**2*s2)**(-nu - n/2.)

        if self.type == 'rq16':
            self.S = self.sigma**2 * (2**(5/2.) * self.corr**2/3./s) * kv(1./3., s * self.corr/np.sqrt(3))/gamma(1./6.)
            self.S[s==0] = 0.

        self.S = np.sqrt(self.S)
        
    def realization(self,seed=None):
        '''Generate a Gaussian Random field with given covariance'''
        if seed is not None:
            np.random.seed(seed)
        Z = np.random.normal(size=self.S.shape)+1j*np.random.normal(size=self.S.shape)
        #Z -= np.mean(Z)
        #Z /= np.std(Z)
        #print(np.mean(Z))
        Y = self.S * Z# * (self.nx*self.ny*self.nz)
        B = (np.fft.ifftn(Y,(self.nx,self.ny,self.nz))).real*(self.sx*self.nx)*(self.sy*self.ny)*(self.sz*self.nz)
        B[::2,:,:] *= -1
        B[:,::2,:] *= -1
        B[:,:,::2] *= -1
        #B -= np.mean(B)
        ###
        # Hack to get scale right
        ###
        B *= self.sigma/np.std(B)
        return B



if __name__=='__main__':
    import pylab as plt
    xvec = np.linspace(0,1,100)
    yvec = np.linspace(0,1,100)
    zvec = np.linspace(0,1,100)
    sim = IonosphereSimulation_(xvec,yvec,zvec,1.,0.4,type='m52')
    dn = sim.realization(seed=1234)
    print(dn.shape)
    fig=plt.figure(figsize=(12,12))
    fig.add_subplot(2,2,1)
    plt.imshow(dn[50,:,:])
    plt.colorbar()
    fig.add_subplot(2,2,2)
    plt.imshow(dn[:,50,:])
    plt.colorbar()
    fig.add_subplot(2,2,3)
    plt.imshow(dn[:,:,50])
    plt.colorbar()
    plt.figure()
    plt.hist(dn.flatten(),bins=25)
    plt.show()
