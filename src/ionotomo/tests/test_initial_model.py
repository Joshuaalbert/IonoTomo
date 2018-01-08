import numpy as np
import pylab as plt
import matplotlib.colors as colors
from ionotomo import *

def test_initial_model():
    datapack = generate_example_datapack()
    ne_tci = create_initial_model(datapack)
    pert_tci = create_turbulent_model(datapack,corr=50,factor=10.)
    fig = plt.figure(figsize=(12,12))
    fig.add_subplot(2,2,1)
    im=plt.imshow(pert_tci.M[50,:,:].T,origin='lower',
            norm=colors.LogNorm(vmin=pert_tci.M.min(), vmax=pert_tci.M.max()),
                   cmap='PuBu_r',extent=(pert_tci.yvec[0],pert_tci.yvec[-1],pert_tci.zvec[0],pert_tci.zvec[-1]))
    plt.colorbar(im)
    fig.add_subplot(2,2,2)
    plt.imshow(pert_tci.M[:,50,:].T,origin='lower',
            norm=colors.LogNorm(vmin=pert_tci.M.min(), vmax=pert_tci.M.max()),
                   cmap='PuBu_r',extent=(pert_tci.xvec[0],pert_tci.xvec[-1],pert_tci.zvec[0],pert_tci.zvec[-1]))
    fig.add_subplot(2,2,3)
    plt.imshow(pert_tci.M[:,:,50].T,origin='lower',
            norm=colors.LogNorm(vmin=pert_tci.M.min(), vmax=pert_tci.M.max()),
                   cmap='PuBu_r',extent=(pert_tci.xvec[0],pert_tci.xvec[-1],pert_tci.yvec[0],pert_tci.yvec[-1]))
    fig.add_subplot(2,2,4)
    plt.hist(np.log10(pert_tci.M).flatten(),bins=100)
    plt.show()
            
