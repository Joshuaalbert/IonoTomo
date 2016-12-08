
# coding: utf-8

# In[67]:

import numpy as np
from sympy import symbols,sqrt,sech,Rational,lambdify
from sympy.vector import CoordSysCartesian
import FermatPrinciple as fp
import Inversion as inv
from Geometry import *




def simulateData(rays,octTree=None):
    '''Generate iri model'''
    if octTree is None:
        octTree = OctTree([0,0,1000],dx=100,dy=100,dz=2000)
        octTree = subDivide(octTree)
        octTree = subDivide(octTree)
        octTree = subDivide(octTree)
        octTree = subDivide(octTree)#divide into 16 pieces
    G,mVar,mexact,x = inv.generateModelFromOctree(octTree,0)
    
    d = np.genfromtxt('exampleIRI.txt',names=True)
    profile = d['ne'] + np.mean(d['ne'])*np.exp(-(d['height']-600.)**2/(50**2))
    #plt.plot(d['height'],d['ne'],c='black',label='IRI')
    #plt.plot(d['height'],profile,ls='--',c='red',label='perturbation')
    #plt.legend(frameon=False)
    #plt.xlabel('Height (km)')
    #plt.ylabel(r'Electron density $n_e$ (${\rm m}^{-3}$)')
    #plt.yscale('log')
    #plt.grid()
    #plt.title('International Reference Ionosphere')
    ne = np.interp(x[:,2],d['height'],d['ne'])
    ne += np.mean(d['ne'])*np.exp(-((x[:,0]-20.)**2+(x[:,1])**2+(x[:,2]-600.)**2)/(50**2))
    inv.setOctTreeElectronDensity(octTree,ne,ne*0.01,frequency=120e6)
    print("generating symbolic model")
    s = fp.SmoothVoxel(octTree)
    model_ne = s.smoothifyOctTree_ne()
    print(model_ne)
    
    
def LMSolContinous(model,rays):
    '''Solve in continuous basis'''
    
    f = fp.Fermat(nFunc,frequency = 120e6)
    #forward problem
    for ray in rays:
        x0 = ray.origin
        cosines = x0.dir
        x,y,z,s = integrateRay(x0,cosines,fermat,zinit,zmax)

rays = inv.makeRaysFromSourceAndReciever()
simulateData(rays,octTree=None)


# In[ ]:



