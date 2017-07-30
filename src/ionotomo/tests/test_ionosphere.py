import ionotomo.ionosphere.iri as iri
from ionotomo.ionosphere.covariance import Covariance
from ionotomo.geometry.tri_cubic import TriCubic
import numpy as np

def test_chapman_layers(plot=False):

    zenith = 45.
    heights = np.linspace(-10,2000,1000)
    if plot:
        import pylab as plt
        print("Plotting iri with zenith angles 0,20,45,65,90")
        for zenith in [0,20,45,65,90]:
            ne = iri.a_priori_model(heights,zenith)
            plt.plot(heights,ne)    
            plt.xlabel('height (km)')
            plt.ylabel('ne [m^-3]')
            plt.yscale('log')
        plt.show()


def test_turbulent_realisation(plot=False):

    vec = np.linspace(-np.pi,np.pi,100)
    X,Y,Z = np.meshgrid(vec,vec,vec,indexing='ij')
    R = np.sqrt(X**2 + Y**2 + Z**2)
    M = np.zeros([100,100,100])
    TCI = TriCubic(vec,vec,vec,M)
    print("Matern 1/2 covariance 0.4 correlation")
    cov_obj = Covariance(TCI,np.log(5),0.4,1./2.)
    print("Testing spectral density")

    G = cov_obj.f
    Crec = np.fft.fftn(G).real*cov_obj.dV
    expCov = np.log(5)*np.exp(-R/0.4)
    if plot:
        import pylab as plt

        f = plt.figure(figsize=(8,4))

        vmin = np.min(expCov)
        vmax = np.max(expCov)
        ax = f.add_subplot(1,3,1)

        ax.imshow(G[50,:,:])

        ax = f.add_subplot(1,3,2)

        plt.imshow(expCov[50,:,:])

        ax = f.add_subplot(1,3,3)

        im = plt.imshow(Crec[50,:,:])

        #plt.colorbar(im)
        plt.show()


    B = cov_obj.realization()
    print("Fluctuations measured {}".format((np.percentile(B.flatten(),95) + np.percentile(-B.flatten(),95))))

    print("Compared with input {}".format(cov_obj.sigma))

    #xy slice

    x = TCI.xvec
    y = TCI.yvec
    z = TCI.zvec
    if plot:
        f = plt.figure(figsize=(8,4))

        vmin = np.min(B)
        vmax = np.max(B)
        ax = f.add_subplot(1,3,1)

        ax.imshow(B[0,:,:],extent=(z[0],z[-1],y[0],y[-1]),vmin=vmin,vmax=vmax)

        ax = f.add_subplot(1,3,2)

        plt.imshow(B[:,0,:],extent=(z[0],z[-1],x[0],x[-1]),vmin=vmin,vmax=vmax)

        ax = f.add_subplot(1,3,3)

        im = plt.imshow(B[:,:,0],extent=(y[0],y[-1],x[0],x[-1]),vmin=vmin,vmax=vmax)

        plt.colorbar(im)
        plt.show()

        
    print("testing contraction C^{-1}.phi")

    phi = np.zeros_like(TCI.get_shaped_array())

    phi = np.cos(R*4)*np.exp(-R)

    #phi[30:40,:,:] = 1.
    phihat = cov_obj.contract(phi)
    #Analytic for exp covariance is 1/(8*np.pi*sigma**2) * (1/L**3 * phi - 2/L * Lap phi + L * Lap Lap phi)

    phih = phi.copy()/cov_obj.corr**3
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

    stencil /= cov_obj.dV**(2./3.)

                

    lap = ndimage.convolve(phi,stencil,mode='wrap')

    phih -= 2/cov_obj.corr*lap
    
    laplap = ndimage.convolve(lap,stencil,mode='wrap')

    phih += cov_obj.corr*laplap
    
    phih /= 8*np.pi*cov_obj.sigma**2

    if plot:
        f = plt.figure(figsize=(12,12))

        ax = f.add_subplot(3,3,1)

        ax.set_title("phi")

        im = ax.imshow(phi[50,:,:],extent=(z[0],z[-1],y[0],y[-1]))

        plt.colorbar(im)
        ax = f.add_subplot(3,3,2)

        ax.set_title("FFT based")

        im = plt.imshow(phihat[50,:,:],extent=(z[0],z[-1],x[0],x[-1]))

        plt.colorbar(im)
        ax = f.add_subplot(3,3,3)

        ax.set_title("Analytic")

        im = plt.imshow(phih[50,:,:],extent=(y[0],y[-1],x[0],x[-1]))

        plt.colorbar(im)

        ax = f.add_subplot(3,3,4)

        im = ax.imshow(phi[:,20,:],extent=(z[0],z[-1],y[0],y[-1]))

        plt.colorbar(im)
        ax = f.add_subplot(3,3,5)

        im = plt.imshow(phihat[:,20,:],extent=(z[0],z[-1],x[0],x[-1]))

        plt.colorbar(im)
        ax = f.add_subplot(3,3,6)

        im = plt.imshow(phih[:,20,:],extent=(y[0],y[-1],x[0],x[-1]))

        plt.colorbar(im)

        ax = f.add_subplot(3,3,7)

        im = ax.imshow(phi[:,:,70],extent=(z[0],z[-1],y[0],y[-1]))
        plt.colorbar(im)
        ax = f.add_subplot(3,3,8)
        im = plt.imshow(phihat[:,:,70],extent=(z[0],z[-1],x[0],x[-1]))
        plt.colorbar(im)
        ax = f.add_subplot(3,3,9)
        im = plt.imshow(phih[:,:,70],extent=(y[0],y[-1],x[0],x[-1]))
        plt.colorbar(im)
        plt.show()

    
