
# coding: utf-8

# In[ ]:

import dask.array as da
import numpy as np
from dask.dot import dot_graph
#from dask.multiprocessing import get
from dask import get
from functools import partial
from time import sleep, clock
from scipy.integrate import simps

from dask.callbacks import Callback
#from multiprocessing.pool import ThreadPool

from ForwardEquation import forwardEquation, forwardEquation_dask
from Gradient import computeGradient, computeGradient_dask
from TricubicInterpolation import TriCubic
from LineSearch import lineSearch
from InfoCompleteness import precondition
from InitialModel import createInitialModel
from CalcRays import calcRays,calcRays_dask
from RealData import plotDataPack

outputfolder = 'output/test/bfgs_dask_bootes'

def store_Fdot(n1,n2,Fdot,gamma,beta,dm,dgamma,v,sigma_m,L_m):
    filename="{}/F{}gamma{}.hdf5".format(outputfolder,n1,n2)
    if os.path.isfile(filename):
        return filename
    #gamma.save(filename)
    out = Fdot.copy()
    xvec = dm.xvec
    yvec = dm.yvec
    zvec = dm.zvec
    gamma_dm = scalarProduct(gamma.getShapedArray(),dm.getShapedArray(),sigma_m,L_m,xvec,yvec,zvec)
    a = dm.m*(beta*gamma_dm - scalarProduct(dgamma.getShapedArray(),Fdot.getShapedArray(),sigma_m,L_m,xvec,yvec,zvec))
    a -= v.m*gamma_dm
    a /= scalarProduct(dgamma.getShapedArray(),dm.getShapedArray(),sigma_m,L_m,xvec,yvec,zvec)
    out.m + a
    print("Beta: {}".format(beta))
    print("Difference: {}".format(np.dot(out.m,gamma.m)/np.linalg.norm(out.m)/np.linalg.norm(gamma.m)))
    out.save(filename)
    return filename

def pull_Fdot(filename):
    return TriCubic(filename=filename)

def pull_gamma(filename):
    return TriCubic(filename=filename)

def store_gamma(n,rays, g, dobs, i0, K_ne, mTCI, mPrior, CdCt, sigma_m, Nkernel, sizeCell):
    filename='{}/gamma_{}.hdf5'.format(outputfolder,n)
    if os.path.isfile(filename):
        return filename
    gradient = computeGradient_dask(rays, g, dobs, i0, K_ne, mTCI, mPrior.getShapedArray(), CdCt, sigma_m, Nkernel, sizeCell)
    TCI = TriCubic(mTCI.xvec,mTCI.yvec,mTCI.zvec,gradient)
    TCI.save(filename)
    return filename

def store_forwardEq(n,templateDatapack,antIdx,timeIdx,dirIdx,rays,K_ne,mTCI,i0):
    filename = "{}/g_{}.hdf5".format(outputfolder,n)
    if os.path.isfile(filename):
        return filename
    g = forwardEquation_dask(rays,K_ne,mTCI,i0)
    datapack = templateDatapack.clone()
    datapack.set_dtec(g,antIdx=antIdx,timeIdx=timeIdx,dirIdx = dirIdx)
    datapack.save(filename)
    dobs = templateDatapack.get_dtec(antIdx=antIdx,timeIdx=timeIdx,dirIdx = dirIdx)
    vmin = np.min(dobs)
    vmax = np.max(dobs)
    plotDataPack(datapack,antIdx=antIdx,timeIdx=timeIdx,dirIdx = dirIdx,
                figname=filename.split('.')[0], vmin = vmin, vmax = vmax)#replace('hdf5','png'))
    return filename

def pull_forwardEq(filename,antIdx,timeIdx,dirIdx):
    datapack = DataPack(filename=filename)
    g = datapack.get_dtec(antIdx=antIdx,timeIdx=timeIdx,dirIdx = dirIdx)
    return g

def store_m(n,mTCI0,epsilon_n,phi):
    filename = "{}/m_{}.hdf5".format(outputfolder,n)
    if os.path.isfile(filename):
        return filename
    mTCI = mTCI0.copy()
    mTCI.m -= epsilon_n*phi.m
    mTCI.save(filename)
    return filename

def pull_m(filename):
    return TriCubic(filename=filename)

def calcEpsilon(n,phi,mTCI,rays,K_ne,i0,g,dobs,CdCt):
    import pylab as plt
    plt.figure()
    plt.hist(g.flatten(),alpha=0.2,label='g')
    plt.hist(dobs.flatten(),alpha=0.2,label='dobs')
    plt.legend(frameon=False)
    plt.savefig("{}/data-hist-{}.png".format(outputfolder,n))
    plt.clf()
    plt.hist((g-dobs).flatten())
    plt.savefig("{}/datadiff-hist-{}.png".format(outputfolder,n))
    ep = lineSearch(rays,K_ne,mTCI,i0,phi.getShapedArray(),g,dobs,CdCt,figname="{}/lineSearch{}".format(outputfolder,n))
    return ep  

def scalarProduct(a,b,sigma_m,L_m,xvec,yvec,zvec):
    out = a*b
    out = simps(simps(simps(out,zvec,axis=2),yvec,axis=1),xvec,axis=0)
    out /= (np.pi*8.*sigma_m**2 * L_m**3)
    return out

def calcBeta(dgamma, v, dm,sigma_m,L_m):
    xvec = dgamma.xvec
    yvec = dgamma.yvec
    zvec = dgamma.zvec
    beta = 1. + scalarProduct(dgamma.getShapedArray(),v.getShapedArray(),sigma_m,L_m,xvec,yvec,zvec)/(scalarProduct(dgamma.getShapedArray(),dm.getShapedArray(),sigma_m,L_m,xvec,yvec,zvec) + 1e-15)
    print(np.mean(dm.m),np.mean(dgamma.m))
    return beta

def diffTCI(TCI1,TCI2):
    TCI = TCI1.copy()
    TCI.m -= TCI2.m
    return TCI

def store_F0dot(n,F0,gamma):
    filename="{}/F0gamma{}.hdf5".format(outputfolder,n)
    if os.path.isfile(filename):
        return filename
    out = gamma.copy()
    out.m *= F0.m
    out.save(filename)
    return filename

def createBFGSDask(N,datapack,L_ne,sizeCell,i0, antIdx=-1, dirIdx=-1, timeIdx = [0]):
    print("Using output folder: {}".format(outputfolder))
    import pylab as plt
    straightLineApprox = True
    tmax = 1000.
    antennas,antennaLabels = datapack.get_antennas(antIdx = antIdx)
    patches, patchNames = datapack.get_directions(dirIdx = dirIdx)
    times,timestamps = datapack.get_times(timeIdx=timeIdx)
    datapack.setReferenceAntenna(antennaLabels[i0])
    #plotDataPack(datapack,antIdx = antIdx, timeIdx = timeIdx, dirIdx = dirIdx,figname='{}/dobs'.format(outputfolder))
    dobs = datapack.get_dtec(antIdx = antIdx, timeIdx = timeIdx, dirIdx = dirIdx)
    Na = len(antennas)
    Nt = len(times)
    Nd = len(patches) 
    fixtime = times[Nt>>1]
    phase = datapack.getCenterDirection()
    arrayCenter = datapack.radioArray.getCenter()
    #Average time axis down and center on fixtime
    if Nt == 1:
        var = (0.5*np.percentile(dobs[dobs>0],75) + 0.5*np.percentile(-dobs[dobs<0],75))**2
        CdCt = np.ones([Na,1,Nd],dtype=np.double)*var
    else:
        dt = times[1].gps - times[0].gps
        print("Averaging down window of length {} seconds [{} timestamps]".format(dt*Nt, Nt))
        CdCt = np.stack([np.var(dobs,axis=1)],axis=1) 
        dobs = np.stack([np.mean(dobs,axis=1)],axis=1)
        timeIdx = [Nt>>1]
        times,timestamps = datapack.get_times(timeIdx=timeIdx)
        Nt = len(times)
    print("CdCt: {}".format(np.mean(CdCt)))
    vmin = np.min(datapack.get_dtec(antIdx = antIdx, timeIdx = timeIdx, dirIdx = dirIdx))
    vmax = np.max(datapack.get_dtec(antIdx = antIdx, timeIdx = timeIdx, dirIdx = dirIdx))
    plotDataPack(datapack,antIdx=antIdx,timeIdx=timeIdx,dirIdx = dirIdx,
            figname='{}/dobs'.format(outputfolder), vmin = vmin, vmax = vmax)#replace('hdf5','png'))
    neTCI = createInitialModel(datapack,antIdx = antIdx, timeIdx = timeIdx, dirIdx = dirIdx, zmax = tmax,spacing=sizeCell)
    neTCI.save("{}/nePriori.hdf5".format(outputfolder))
    rays = calcRays(antennas,patches,times, arrayCenter, fixtime, phase, neTCI, datapack.radioArray.frequency, 
                    straightLineApprox, tmax, neTCI.nz)
#     bestFit = np.inf
#     bestScale = 0
#     for scale in np.linspace(0.1,10,50):
#         mTCI = neTCI.copy()
#         mTCI.m *= scale
#         K_ne = np.mean(mTCI.m)
#         mTCI.m /= K_ne
#         np.log(mTCI.m,out=mTCI.m)
#         g = forwardEquation_dask(rays,K_ne,mTCI,i0)
#         maxRes = np.max(np.abs(g-dobs))
#         if maxRes < bestFit:
#             bestScale = scale
#             bestFit = maxRes
#     print("Best Scale = {}".format(bestScale))
#     print("Max residual = {}".format(bestFit))
#     neTCI.m *= bestScale
    mTCI = neTCI.copy()
    K_ne = np.mean(mTCI.m)
    mTCI.m /= K_ne
    np.log(mTCI.m,out=mTCI.m)
    
    Nkernel = max(1,int(float(L_ne)/sizeCell))
    sigma_m = np.log(10./0.1)/2.#ne = K*exp(m+dm) = K*exp(m)*exp(dm), exp(dm) in (0.1,10) -> dm = (log(10) - log(0.1))/2.
    #uvw = UVW(location = datapack.radioArray.getCenter().earth_location,obstime = fixtime,phase = phase)
    #ants_uvw = antennas.transform_to(uvw).cartesian.xyz.to(au.km).value.transpose()
    #dirs_uvw = patches.transform_to(uvw).cartesian.xyz.value.transpose()
    F0 = precondition(neTCI, datapack,antIdx=antIdx, dirIdx=dirIdx, timeIdx = timeIdx)
    #
    dsk = {}
    for n in range(N):
        #g_n
        dsk['store_forwardEq{}'.format(n)] = (store_forwardEq,n,'templateDatapack','antIdx','timeIdx','dirIdx','rays',
                                              'K_ne','pull_m{}'.format(n),'i0')
        dsk['pull_forwardEq{}'.format(n)] = (pull_forwardEq,'store_forwardEq{}'.format(n),'antIdx','timeIdx','dirIdx')
        #gradient
        dsk['store_gamma{}'.format(n)] = (store_gamma,n,'rays', 'pull_forwardEq{}'.format(n), 'dobs', 'i0', 'K_ne', 
                                        'pull_m{}'.format(n),'mprior', 'CdCt', 'sigma_m', 'Nkernel', 'sizeCell')
        dsk['pull_gamma{}'.format(n)] = (pull_gamma,'store_gamma{}'.format(n))
        #m update
        dsk['store_m{}'.format(n+1)] = (store_m,n+1,'pull_m{}'.format(n),'ep{}'.format(n),'pull_phi{}'.format(n))
        dsk['pull_m{}'.format(n+1)] = (pull_m,'store_m{}'.format(n+1))
        #phi
        dsk['pull_phi{}'.format(n)] = (pull_Fdot,'store_F{}(gamma{})'.format(n,n))
        dsk['store_F{}(gamma{})'.format(n+1,n+1)] = (store_Fdot, n+1, n+1 ,
                                                     'pull_F{}(gamma{})'.format(n,n+1),
                                                     'pull_gamma{}'.format(n+1),
                                                     'beta{}'.format(n),
                                                     'dm{}'.format(n),
                                                     'dgamma{}'.format(n),
                                                     'v{}'.format(n),
                                                     'sigma_m','L_m'
                                                    )
        for i in range(1,n+1):
            dsk['store_F{}(gamma{})'.format(i,n+1)] = (store_Fdot, i, n+1 ,
                                                     'pull_F{}(gamma{})'.format(i-1,n+1),
                                                     'pull_gamma{}'.format(n+1),
                                                     'beta{}'.format(i-1),
                                                     'dm{}'.format(i-1),
                                                     'dgamma{}'.format(i-1),
                                                     'v{}'.format(i-1),
                                                       'sigma_m','L_m'
                                                    )
            dsk['pull_F{}(gamma{})'.format(i,n+1)] = (pull_Fdot,'store_F{}(gamma{})'.format(i,n+1))
        #should replace for n=0
        dsk['store_F0(gamma{})'.format(n)] = (store_F0dot, n, 'pull_F0','pull_gamma{}'.format(n))
        dsk['pull_F0(gamma{})'.format(n)] = (pull_Fdot,'store_F0(gamma{})'.format(n))
        #epsilon_n       
        dsk['ep{}'.format(n)] = (calcEpsilon,n,'pull_phi{}'.format(n),'pull_m{}'.format(n),'rays',
                                    'K_ne','i0','pull_forwardEq{}'.format(n),'dobs','CdCt')
        #
        dsk['beta{}'.format(n)] = (calcBeta,'dgamma{}'.format(n),'v{}'.format(n),'dm{}'.format(n),'sigma_m','L_m')
        dsk['dgamma{}'.format(n)] = (diffTCI,'pull_gamma{}'.format(n+1),'pull_gamma{}'.format(n))
        dsk['dm{}'.format(n)] = (diffTCI,'pull_m{}'.format(n+1),'pull_m{}'.format(n))
        dsk['v{}'.format(n)] = (diffTCI,'pull_F{}(gamma{})'.format(n,n+1),'pull_phi{}'.format(n))
    dsk['pull_F0'] = F0 
    dsk['templateDatapack'] = datapack
    dsk['antIdx'] = antIdx
    dsk['timeIdx'] = timeIdx
    dsk['dirIdx'] = dirIdx
    dsk['pull_m0'] = 'mprior'
    dsk['i0'] = i0
    dsk['K_ne'] = K_ne
    dsk['dobs'] = dobs
    dsk['mprior'] = mTCI
    dsk['CdCt'] = CdCt
    dsk['sigma_m'] = sigma_m
    dsk['Nkernel'] = Nkernel
    dsk['L_m'] = L_ne
    dsk['sizeCell'] = sizeCell
    #calc rays
    #dsk['rays'] = (calcRays_dask,'antennas','patches','times', 'arrayCenter', 'fixtime', 'phase', 'neTCI', 'frequency',  'straightLineApprox','tmax')
    dsk['rays'] = rays
    return dsk


class TrackingCallbacks(Callback):
    def _start(self,dsk):
        self.startTime = clock()
    def _pretask(self, key, dask, state):
        """Print the key of every task as it's started"""
        self.t1 = clock()
        print('Starting {} at {} seconds'.format(key,self.t1-self.startTime))
    def _posttask(self,key,result,dsk,state,id):
        print("{} took {} seconds".format(repr(key),clock() - self.t1))
    def _finish(self,dsk,state,errored):
        self.endTime = clock()
        dt = (self.endTime - self.startTime)
        print("Approximate time to complete: {} time units".format(dt))
        

    
if __name__=='__main__':
    import os
    try:
        os.makedirs(outputfolder)
    except:
        pass
    from RealData import DataPack
    from AntennaFacetSelection import selectAntennaFacets
    from dask.diagnostics import Profiler, ResourceProfiler, CacheProfiler
    from dask.diagnostics import visualize
    #from InitialModel import createTurbulentlModel
    i0 = 0
    datapack = DataPack(filename="output/test/simulate/simulate_0/datapackSim.hdf5")
    datapack = DataPack(filename="output/test/datapackObs.hdf5")
    #flags = datapack.findFlaggedAntennas()
    #datapack.flagAntennas(flags)
    datapackSel = selectAntennaFacets(15, datapack, antIdx=-1, dirIdx=-1, timeIdx = np.arange(4))
    #pertTCI = createTurbulentlModel(datapackSel,antIdx = -1, timeIdx = -1, dirIdx = -1, zmax = 1000.)
    L_ne = 50.
    sizeCell = 5.
    dsk = createBFGSDask(10,datapackSel,L_ne,sizeCell,i0, antIdx=-1, dirIdx=-1, timeIdx = np.arange(4))
    #dot_graph(dsk,filename="{}/BFGS_graph".format(outputfolder),format='png')
    #dot_graph(dsk,filename="{}/BFGS_graph".format(outputfolder),format='svg')
    #with TrackingCallbacks():
    #with Profiler() as prof, ResourceProfiler(dt=0.25) as rprof, CacheProfiler() as cprof:
    get(dsk,'pull_m10')
    #visualize([prof,rprof,cprof])
    


# In[ ]:




# In[ ]:



