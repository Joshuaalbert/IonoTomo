from ionotomo import *
import numpy as np
import pylab as plt

def test_turbulent_realisation(plot=True):
    xvec = np.linspace(-100,100,100)
    zvec = np.linspace(0,1000,1000)
    M = np.zeros([100,100,1000])
    TCI = TriCubic(xvec,xvec,zvec,M)
    print("Matern 1/2 kernel")
    cov_obj = Covariance(tci=TCI)
    sigma = 1.
    corr = 30.
    nu = 1./2.
    print("Testing spectral density")

    B = cov_obj.realization()
    print("Fluctuations measured {}".format((np.percentile(B.flatten(),95) + np.percentile(-B.flatten(),95))))

    #xy slice

    x = TCI.xvec
    y = TCI.yvec
    z = TCI.zvec
    X,Y,Z = np.meshgrid(x,y,z,indexing='ij')
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]


    if plot and True:
        f = plt.figure(figsize=(8,4))

        vmin = np.min(B)
        vmax = np.max(B)
        ax = f.add_subplot(1,3,1)

        ax.imshow(B[49,:,:],extent=(z[0],z[-1],y[0],y[-1]),vmin=vmin,vmax=vmax)

        ax = f.add_subplot(1,3,2)

        plt.imshow(B[:,49,:],extent=(z[0],z[-1],x[0],x[-1]),vmin=vmin,vmax=vmax)

        ax = f.add_subplot(1,3,3)

        im = plt.imshow(B[:,:,499],extent=(y[0],y[-1],x[0],x[-1]),vmin=vmin,vmax=vmax)

        plt.colorbar(im)
        plt.show()

        
    print("testing contraction C^{-1}.phi")

    phi = np.zeros_like(TCI.M)

    #phi = np.cos(R*4)*np.exp(-R)

    phi = X**2 + Y**2 + Z**4
    phihat = cov_obj.contract(phi)
    assert not np.any(np.isnan(phihat))
    #Analytic for exp covariance is 1/(8*np.pi*sigma**2) * (1/L**3 * phi - 2/L * Lap phi + L * Lap Lap phi)
    # 1/(8*np.pi*sigma**2) * (1/L**3 * phi + 2/L * sin(2 pi Z / 20)*(2*pi/20)**2 + L * sin(2 pi Z / 20)*(2*pi/20)**4)
    phih = 1./(8*np.pi*sigma**2) * ( 1./corr**3 * phi - 2./corr *(2 + 2 + 2*Z**2) + corr*4)

    if plot:
        f = plt.figure(figsize=(12,12))

        ax = f.add_subplot(3,3,1)

        ax.set_title("phi")

        im = ax.imshow(phi[50,:,:],extent=(z[0],z[-1],y[0],y[-1]))

        plt.colorbar(im)
        ax = f.add_subplot(3,3,2)

        ax.set_title("FFT based")

        im = plt.imshow(phihat[50,:,:],extent=(z[0],z[-1],y[0],y[-1]))

        plt.colorbar(im)
        ax = f.add_subplot(3,3,3)

        ax.set_title("Analytic")

        im = plt.imshow(phih[50,:,:],extent=(z[0],z[-1],y[0],y[-1]))

        plt.colorbar(im)

        ax = f.add_subplot(3,3,4)

        im = ax.imshow(phi[:,20,:],extent=(z[0],z[-1],x[0],x[-1]))

        plt.colorbar(im)
        ax = f.add_subplot(3,3,5)

        im = plt.imshow(phihat[:,20,:],extent=(z[0],z[-1],x[0],x[-1]))

        plt.colorbar(im)
        ax = f.add_subplot(3,3,6)

        im = plt.imshow(phih[:,20,:],extent=(z[0],z[-1],x[0],x[-1]))

        plt.colorbar(im)

        ax = f.add_subplot(3,3,7)

        im = ax.imshow(phi[:,:,70],extent=(y[0],y[-1],x[0],x[-1]))
        plt.colorbar(im)
        ax = f.add_subplot(3,3,8)
        im = plt.imshow(phihat[:,:,70],extent=(y[0],y[-1],x[0],x[-1]))
        plt.colorbar(im)
        ax = f.add_subplot(3,3,9)
        im = plt.imshow(phih[:,:,70],extent=(y[0],y[-1],x[0],x[-1]))
        plt.colorbar(im)
        plt.tight_layout()
        plt.show()
    return

    phih = phi.copy()/corr**3
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

    stencil /= (dx*dy*dz)**(2./3.)

                

    lap = ndimage.convolve(phi,stencil,mode='wrap')

    phih -= 2/corr*lap
    
    laplap = ndimage.convolve(lap,stencil,mode='wrap')

    phih += corr*laplap
    
    phih /= 8*np.pi*sigma**2

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

    
