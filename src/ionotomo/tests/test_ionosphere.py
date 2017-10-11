from ionotomo import *
import numpy as np
import pylab as plt

def test_chapman_layers(plot=True):
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
    



