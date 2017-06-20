
# coding: utf-8

# In[ ]:

from dask.multiprocessing import get
import numpy as np
from functools import partial

def precondition(ants_uvw,dirs_uvw,neTCI,L_pre=7.5):
    xvec = neTCI.xvec
    yvec = neTCI.yvec
    zvec = neTCI.zvec
    X,Y = np.meshgrid(xvec,yvec,indexing='ij')
    X_ = X.flatten()
    Y_ = Y.flatten()
    F0 = np.zeros([xvec.size,yvec.size,zvec.size],dtype=np.double)
    i = 0
    while i < len(zvec):
        scale = zvec[i]/dirs_uvw[:,2]
        pir_u = np.add.outer(ants_uvw[:,0],dirs_uvw[:,0]*scale).flatten()
        pir_v = np.add.outer(ants_uvw[:,1],dirs_uvw[:,1]*scale).flatten()
        F0[:,:,i] = np.sum(np.exp(-(np.subtract.outer(X_,pir_u)**2 + np.subtract.outer(Y_,pir_v)**2)/L_pre**2/2.),axis=1).reshape(X.shape)
        i += 1
    F0[F0 > 5.] = 5.
    F0 /= 5.
    return F0

#store inverion
#store a priori TCI, a posteriori TCI
def storeIteration(outputFolder,iteration,currentTCI):
    pass

def createPriorTCI():
    pass

def intertItertation(currentTCI):
    pass

def update(model,mu,gradient):
    return model - mu*gradient
    
def calcStepSize(modelTCI,direction,rays):
    pass

def determineInversionDomain(spacing,antennas, directions, pointing, zmax, padding = 5):
    '''Determine the domain of the inversion'''
    ants = antennas.transform_to(pointing).cartesian.xyz.to(au.km).value.transpose()
    dirs = directions.transform_to(pointing).cartesian.xyz.value.transpose()
    #old
    umin = min(np.min(ants[:,0]),np.min(dirs[:,0]/dirs[:,2]*zmax))-spacing*padding
    umax = max(np.max(ants[:,0]),np.max(dirs[:,0]/dirs[:,2]*zmax))+spacing*padding
    vmin = min(np.min(ants[:,1]),np.min(dirs[:,1]/dirs[:,2]*zmax))-spacing*padding
    vmax = max(np.max(ants[:,1]),np.max(dirs[:,1]/dirs[:,2]*zmax))+spacing*padding
    wmin = min(np.min(ants[:,2]),np.min(dirs[:,2]/dirs[:,2]*zmax))-spacing*padding
    wmax = max(np.max(ants[:,2]),np.max(dirs[:,2]/dirs[:,2]*zmax))+spacing*padding
    
    umin = np.min(ants[:,0]) + np.min(dirs[:,0]/dirs[:,2]*zmax) - spacing*padding
    umax = np.max(ants[:,0]) + np.max(dirs[:,0]/dirs[:,2]*zmax) + spacing*padding
    vmin = (np.min(ants[:,1]) + np.min(dirs[:,1]/dirs[:,2]*zmax)) - spacing*padding
    vmax = (np.max(ants[:,1]) + np.max(dirs[:,1]/dirs[:,2]*zmax)) + spacing*padding
    wmin = (np.min(ants[:,2]) + np.min(dirs[:,2]/dirs[:,2]*zmax)) - spacing*padding
    wmax = (np.max(ants[:,2]) + np.max(dirs[:,2]/dirs[:,2]*zmax)) + spacing*padding
    Nu = np.ceil((umax-umin)/spacing)
    Nv = np.ceil((vmax-vmin)/spacing)
    Nw = np.ceil((wmax-wmin)/spacing)
    uvec = np.linspace(umin,umax,int(Nu))
    vvec = np.linspace(vmin,vmax,int(Nv))
    wvec = np.linspace(wmin,wmax,int(Nw))
    print("Found domain u in {}..{}, v in {}..{}, w in {}..{}".format(umin,umax,vmin,vmax,wmin,wmax))
    return uvec,vvec,wvec
    
    
outputFolder = 'output'
datapackObs = DataPack(filename='.hdf5')

iteration = 0
Nmax = 1
while iteration < Nmax:
    dsk = {'storeIteration':(storeIteration,"{}/iteration-{}/".format(outputFolder,iteration),iteration,'updateModel'),
       'updateModel':(update,'currentModel','stepSize','direction'),
        'direction':(bfgs,'currentModel',)
       'stepSize':(calcStepSize,'currentModel','direction','rays'),
       'rays':(calcRays,'currentModel','dataPack'),
       'dataPack':datapackObs,
       'aPrioriTCI':(createPriorTCI,),
        'antIdx':antIdx,
        'get_antennas':(partial(dataPack.get_antennas,*args, antIdx = 'antIdx')),
        
        'getTCIDomain':(determineInversionDomain,5.,antennas, directions, pointing, zmax, padding = 5))}



get(dsk, 'analyze')  # executes in parallel
from dask.dot import dot_graph
dot_graph(dsk)
from dask.optimize import cull
dsk1, dependencies = cull(dsk, 'analyze')
from dask.optimize import inline
dsk2 = inline(dsk1, dependencies=dependencies)
from dask.optimize import inline_functions
dsk3 = inline_functions(dsk2, 'analyze', [len, str.split], dependencies=dependencies)
from dask.optimize import fuse
dsk4, dependencies = fuse(dsk3)
dot_graph(dsk4)


# In[ ]:



