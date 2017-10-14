from ionotomo import *
import numpy as np
import pylab as plt

def test_covariance():
    c = Covariance(dx=10.,dy=10.,dz=10.)
    c.create_inverse_stencil()
#    #c.create_inverse_stencil()
#    phi = np.zeros([100,100,1000])
##    phi[10,15:25,:] = 1.
#    phi = np.random.normal(size=phi.shape)
#    phi[phi > 0.5] = 100.
#    phi[phi < 0.5] = 0.
##    plt.imshow(phi[10,:,:],aspect='auto')
##    plt.colorbar()
##    plt.show()
##    print("smooth")
#    phih = c.smooth(phi)
##    plt.imshow(phih[10,:,:],aspect='auto')
##    plt.colorbar()
##    plt.show()
#    print("contract")
#    phi0 = c.contract(phih)
#    plt.imshow(phi0[10,:,:],aspect='auto')
#    plt.colorbar()
#    plt.show()
#    plt.imshow((phi0-phi)[10,:,:],aspect='auto')
#    plt.colorbar()
#    plt.show()
#    plt.hist((phi0 - phi).flatten(),bins = 100)
#    plt.show()
