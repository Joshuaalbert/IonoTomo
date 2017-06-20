
# coding: utf-8

# In[ ]:

'''Handle covariance operations'''
import numpy as np
from scipy.special import gamma

class CovarianceClass(object):
    '''Use for repeated use of covariance'''
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
    
def test_CovarianceClass():
    from TricubicInterpolation import TriCubic
    import pylab as plt
    neTCI = TriCubic(filename='output/test/simulate/simulate_0/neModel.hdf5')
    covC = CovarianceClass(neTCI,np.log(5),25.,7./2.)
    B = covC.realization()
    print("Fluctuations:",(np.max(B) + np.max(-B))/2.)
    #xy slice
    x = neTCI.xvec
    y = neTCI.yvec
    z = neTCI.zvec
    plt.imshow(B[0,:,:],extent=(z[0],z[-1],y[0],y[-1]))
    plt.colorbar()
    plt.show()
    plt.imshow(B[:,0,:],extent=(z[0],z[-1],x[0],x[-1]))
    plt.colorbar()
    plt.show()
    plt.imshow(B[:,:,0],extent=(y[0],y[-1],x[0],x[-1]))
    plt.colorbar()
    plt.show()
    
    phi = np.zeros_like(neTCI.getShapedArray())
    phi[30:40,:,:] = 1.
    phihat = covC.contract(phi)
    #Analytic for exp covariance is 1/(8*np.pi*sigma**2) * (1/L**3 * phi - 2/L * Lap phi + L * Lap Lap phi)
    phih = phi.copy()/covC.corr**3
    from scipy import ndimage
    stencil = np.zeros([3,3,3])
    for i in range(-1,2):
        for j in range(-1,2):
            for k in range(-1,2):
                s = 0
                if i == 0:
                    s += 1
                if j == 0:
                    s += 1
                if k == 0:
                    s += 1
                if s == 3:
                    stencil[i,j,k] = -2*3.
                if s == 3 - 1:
                    stencil[i,j,k] = 1.
    stencil /= covC.dV**(2./3.)
                

    lap = ndimage.convolve(phi,stencil,mode='wrap')
    phih -= 2/covC.corr*lap
    
    laplap = ndimage.convolve(lap,stencil,mode='wrap')
    phih += covC.corr*laplap
    
    phih /= 8*np.pi*covC.sigma**2
    
    
    plt.imshow(phi[0,:,:],extent=(z[0],z[-1],y[0],y[-1]))
    plt.colorbar()
    plt.show()
    plt.imshow(phihat[0,:,:],extent=(z[0],z[-1],y[0],y[-1]))
    plt.colorbar()
    plt.show()
    plt.imshow(phih[0,:,:],extent=(z[0],z[-1],y[0],y[-1]))
    plt.colorbar()
    plt.show()
    
    plt.imshow(phi[:,0,:],extent=(z[0],z[-1],x[0],x[-1]))
    plt.colorbar()
    plt.show()
    plt.imshow(phihat[:,0,:],extent=(z[0],z[-1],x[0],x[-1]))
    plt.colorbar()
    plt.show()
    plt.imshow(phih[:,0,:],extent=(z[0],z[-1],x[0],x[-1]))
    plt.colorbar()
    plt.show()
    
    plt.imshow(phi[:,:,0],extent=(y[0],y[-1],x[0],x[-1]))
    plt.colorbar()
    plt.show()
    plt.imshow(phihat[:,:,0],extent=(y[0],y[-1],x[0],x[-1]))
    plt.colorbar()
    plt.show()
    plt.imshow(phih[:,:,0],extent=(y[0],y[-1],x[0],x[-1]))
    plt.colorbar()
    plt.show() 
       
if __name__ == '__main__':
    test_CovarianceClass()

