import ionotomo.ionosphere.iri as iri
from ionotomo.ionosphere.covariance import Covariance
from ionotomo.inversion.initial_model import *
from ionotomo.geometry.tri_cubic import TriCubic
import numpy as np
import pylab as plt

def test_chapman_layers(plot=False):

    zenith = 45.
    heights = np.linspace(-10,2000,1000)
    ne = iri.a_priori_model(heights,zenith)

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
    
def test_turbulent_scale():
    vec = np.linspace(0,100,100)
    X,Y,Z = np.meshgrid(vec,vec,vec,indexing='ij')
    R = np.sqrt(X**2 + Y**2 + Z**2)
    M = np.zeros([len(vec)]*3)
    TCI = TriCubic(vec,vec,vec,M)
    print("Matern 2/3 covariance 20 correlation")
    cov_obj = Covariance(TCI,0.05,20.,2./3.)
    B = cov_obj.realization()
    print("Fluctuations measured {}".format((np.percentile(B.flatten(),95) + np.percentile(-B.flatten(),95))))

    print("Compared with input {}".format(cov_obj.sigma))

    #compute <B(x)B(x+dx)>
    i_,j_,k_ = B.shape[0]>>1,B.shape[1]>>1,B.shape[2]>>1
    dx = []
    C = []
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            for k in range(B.shape[2]):
                dx.append(np.sqrt((X[i,j,k] - X[i_,j_,k_])**2 + (Y[i,j,k] - Y[i_,j_,k_])**2  + (Z[i,j,k] - Z[i_,j_,k_])**2 ))
                C.append(np.abs(B[i,j,k]-B[i_,j_,k_]))
    dx = np.array(dx)
    C = np.array(C)
    levels = []
    for q in [0,10,20,30,40,50,60,70,80,90,100]:
        levels.append(np.percentile(dx,q))
    levels = np.array(levels)
    #plt.plot([0,10,20,30,40,50,60,70,80,90,100],levels)
    levels = np.linspace(np.min(dx),np.max(dx),20)
    #plt.show()
    E = []
    V = []
    x = []
    xbar = []
    ybar = []
    for i in range(1,len(levels)):
        mask = np.bitwise_and((dx > levels[i-1]),(dx <= levels[i]))
        x.append((levels[i] + levels[i-1])/2.)
        E.append(np.mean(C[mask]))
        V.append(np.var(C[mask]))
        xbar.append([x[-1] - levels[i-1], levels[i] - x[-1]])
        ybar.append(np.sqrt(V[-1]))
    xbar = np.array(xbar)
    plt.errorbar(x,E,xerr=[xbar[:,0],xbar[:,1]],yerr=ybar)
    #plt.show()
    #xy slice

    x = TCI.xvec
    y = TCI.yvec
    z = TCI.zvec
    f = plt.figure(figsize=(8,4))

    vmin = np.min(B)
    vmax = np.max(B)
    #ax = f.add_subplot(1,3,1)

    #ax.imshow(B[0,:,:],extent=(z[0],z[-1],y[0],y[-1]),vmin=vmin,vmax=vmax)

    #ax = f.add_subplot(1,3,2)

    #plt.imshow(B[:,0,:],extent=(z[0],z[-1],x[0],x[-1]),vmin=vmin,vmax=vmax)

    #ax = f.add_subplot(1,3,3)

    #im = plt.imshow(B[:,:,0],extent=(y[0],y[-1],x[0],x[-1]),vmin=vmin,vmax=vmax)

    #plt.colorbar(im)
    #plt.show()



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

    phi = np.zeros_like(TCI.M)

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

    
