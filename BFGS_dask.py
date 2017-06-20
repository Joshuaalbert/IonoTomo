
# coding: utf-8

# In[1]:

import dask.array as da
import os
import numpy as np
import pylab as plt
import h5py
from dask.dot import dot_graph
#from dask.multiprocessing import get
from dask import get
from functools import partial
from time import sleep, clock
from scipy.integrate import simps

from dask.callbacks import Callback
from distributed import Client
#from multiprocessing.pool import ThreadPool

from ForwardEquation import forwardEquation, forwardEquation_dask
from Gradient import computeGradient, computeGradient_dask
from TricubicInterpolation import TriCubic
from LineSearch import lineSearch
from InfoCompleteness import precondition
from InitialModel import createInitialModel
from CalcRays import calcRays,calcRays_dask
from RealData import plotDataPack
from PlotTools import animateTCISlices
from Covariance import CovarianceClass


def store_Fdot(resettable,outputfolder,n1,n2,Fdot,gamma,beta,dm,dgamma,v,sigma_m,L_m):
    filename="{}/F{}gamma{}.hdf5".format(outputfolder,n1,n2)
    if os.path.isfile(filename) and resettable:
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
    
    if resettable:
        out.save(filename)
        return filename
    else:
        return out

def pull_Fdot(resettable,filename):
    if resettable:
        return TriCubic(filename=filename)
    else:
        return filename

def pull_gamma(resettable,filename):
    if resettable:
        return TriCubic(filename=filename)
    else:
        return filename

def store_gamma(resettable,outputfolder,n,rays, g, dobs, i0, K_ne, mTCI, mPrior, CdCt, sigma_m, Nkernel, sizeCell,covC):
    filename='{}/gamma_{}.hdf5'.format(outputfolder,n)
    if os.path.isfile(filename) and resettable:
        return filename
    gradient = computeGradient_dask(rays, g, dobs, i0, K_ne, mTCI, mPrior.getShapedArray(), CdCt, sigma_m, Nkernel, sizeCell,covC)
    TCI = TriCubic(mTCI.xvec,mTCI.yvec,mTCI.zvec,gradient)
    
    if resettable:
        TCI.save(filename)
        return filename
    else:
        return TCI

def plot_gamma(outputfolder,n,TCI):
    foldername = '{}/gamma_{}'.format(outputfolder,n)
    animateTCISlices(TCI,foldername,numSeconds=20.)
    return foldername
    

def store_forwardEq(resettable,outputfolder,n,templateDatapack,antIdx,timeIdx,dirIdx,rays,K_ne,mTCI,i0):
    filename = "{}/g_{}.hdf5".format(outputfolder,n)
    if os.path.isfile(filename) and resettable:
        return filename
    assert not np.any(np.isnan(mTCI.m)), "nans in model"
    g = forwardEquation(rays,K_ne,mTCI,i0)
    assert not np.any(np.isnan(g)), "nans in g"
    datapack = templateDatapack.clone()
    datapack.set_dtec(g,antIdx=antIdx,timeIdx=timeIdx,dirIdx = dirIdx)
    
    dobs = templateDatapack.get_dtec(antIdx=antIdx,timeIdx=timeIdx,dirIdx = dirIdx)
    vmin = np.min(dobs)
    vmax = np.max(dobs)
    plotDataPack(datapack,antIdx=antIdx,timeIdx=timeIdx,dirIdx = dirIdx,
                figname=filename.split('.')[0], vmin = vmin, vmax = vmax)#replace('hdf5','png'))
    if resettable:
        datapack.save(filename)
        return filename
    else:
        return datapack

def pull_forwardEq(resettable,filename,antIdx,timeIdx,dirIdx):
    if resettable:
        datapack = DataPack(filename=filename)
        g = datapack.get_dtec(antIdx=antIdx,timeIdx=timeIdx,dirIdx = dirIdx)
        return g
    else:
        g = filename.get_dtec(antIdx=antIdx,timeIdx=timeIdx,dirIdx = dirIdx)
        return g

def calcEpsilon(outputfolder,n,phi,mTCI,rays,K_ne,i0,g,dobs,CdCt):
    bins = max(10,int(np.ceil(np.sqrt(g.size))))
    drad = 8.44797256e-7/120e6
    dtau = 1.34453659e-7/120e6**2
    r = dtau/drad*1e9#mu sec factor
    plt.figure()
    plt.hist(g.flatten(),alpha=0.2,label='g',bins=bins)
    plt.hist(dobs.flatten(),alpha=0.2,label='dobs',bins=bins)
    plt.legend(frameon=False)
    plt.savefig("{}/data-hist-{}.png".format(outputfolder,n))
    plt.clf()
    plt.hist((g-dobs).flatten()*drad*1e16,bins=bins)
    plt.xlabel(r"$d\phi$ [rad] | {:.2f} delay [ns]".format(r))
    plt.savefig("{}/datadiff-hist-{}.png".format(outputfolder,n))
    plt.close('all')
    ep,S,reduction = lineSearch(rays,K_ne,mTCI,i0,phi.getShapedArray(),g,dobs,CdCt,figname="{}/lineSearch{}".format(outputfolder,n))
    return ep,S,reduction

def store_m(resettable,outputfolder,n,mTCI0,phi,rays,K_ne,i0,g,dobs,CdCt,stateFile):
    filename = "{}/m_{}.hdf5".format(outputfolder,n)
    with h5py.File(stateFile,'w') as state:
        if '/{}/epsilon_n'.format(n) not in state:
            epsilon_n,S,reduction = calcEpsilon(outputfolder,n,phi,mTCI0,rays,K_ne,i0,g,dobs,CdCt)
            state['/{}/epsilon_n'.format(n)] = epsilon_n
            state['/{}/S'.format(n)] = S
            state['/{}/reduction'.format(n)] = reduction
            state.flush()
        else:
            epsilon_n,S,reduction = state['/{}/epsilon_n'.format(n)], state['/{}/S'.format(n)], state['/{}/reduction'.format(n)]
            
    if os.path.isfile(filename) and resettable:
        return filename
    mTCI = mTCI0.copy()
    mTCI.m -= epsilon_n*phi.m
    if resettable:
        mTCI.save(filename)
        return filename
    else:
        return mTCI

def pull_m(resettable,filename):
    if resettable:
        return TriCubic(filename=filename)
    else:
        return filename #object not filename


def scalarProduct(a,b,sigma_m,L_m,xvec,yvec,zvec):
    out = a*b
    out = simps(simps(simps(out,zvec,axis=2),yvec,axis=1),xvec,axis=0)
    #out /= (np.pi*8.*sigma_m**2 * L_m**3)
    return out

def calcBeta(dgamma, v, dm,sigma_m,L_m):
    xvec = dgamma.xvec
    yvec = dgamma.yvec
    zvec = dgamma.zvec
    beta = 1. + scalarProduct(dgamma.getShapedArray(),v.getShapedArray(),sigma_m,L_m,xvec,yvec,zvec)/(scalarProduct(dgamma.getShapedArray(),dm.getShapedArray(),sigma_m,L_m,xvec,yvec,zvec) + 1e-15)
    print("E[|dm|] = {} | E[|dgamma|] = {}".format(np.mean(np.abs(dm.m)),np.mean(np.abs(dgamma.m))))
    return beta

def diffTCI(TCI1,TCI2):
    TCI = TCI1.copy()
    TCI.m -= TCI2.m
    return TCI

def store_F0dot(resettable,outputfolder,n,F0,gamma):
    filename="{}/F0gamma{}.hdf5".format(outputfolder,n)
    if os.path.isfile(filename) and resettable:
        return filename
    out = gamma.copy()
    out.m *= F0.m
    if resettable:
        out.save(filename)
        return filename
    else:
        return out

def plot_model(outputfolder,n,mModel,mPrior,K_ne):
    tmp = mModel.m.copy()
    np.exp(mModel.m,out=mModel.m)
    mModel.m *= K_ne
    mModel.m -= K_ne*np.exp(mPrior.m)
    foldername = '{}/m_{}'.format(outputfolder,n)
    animateTCISlices(mModel,foldername,numSeconds=20.)
    mModel.m = tmp
    print("Animation of model - prior in {}".format(foldername))
    return foldername

def createBFGSDask(resettable,outputfolder,N,datapack,L_ne,sizeCell,i0, antIdx=-1, dirIdx=-1, timeIdx = [0]):
    try:
        os.makedirs(outputfolder)
    except:
        pass
    print("Using output folder: {}".format(outputfolder))
    stateFile = "{}/state".format(outputfolder)
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
        var = (0.5*np.percentile(dobs[dobs>0],25) + 0.5*np.percentile(-dobs[dobs<0],25))**2
        Cd = np.ones([Na,1,Nd],dtype=np.double)*var
        Ct = (np.abs(dobs)*0.05)**2
        CdCt = Cd + Ct        
    else:
        dt = times[1].gps - times[0].gps
        print("Averaging down window of length {} seconds [{} timestamps]".format(dt*Nt, Nt))
        Cd = np.stack([np.var(dobs,axis=1)],axis=1)
        dobs = np.stack([np.mean(dobs,axis=1)],axis=1)
        Ct = (np.abs(dobs)*0.05)**2
        CdCt = Cd + Ct
        timeIdx = [Nt>>1]
        times,timestamps = datapack.get_times(timeIdx=timeIdx)
        Nt = len(times)
    print("E[S/N]: {} +/- {}".format(np.mean(np.abs(dobs)/np.sqrt(CdCt+1e-15)),np.std(np.abs(dobs)/np.sqrt(CdCt+1e-15))))
    vmin = np.min(datapack.get_dtec(antIdx = antIdx, timeIdx = timeIdx, dirIdx = dirIdx))
    vmax = np.max(datapack.get_dtec(antIdx = antIdx, timeIdx = timeIdx, dirIdx = dirIdx))
    plotDataPack(datapack,antIdx=antIdx,timeIdx=timeIdx,dirIdx = dirIdx,
            figname='{}/dobs'.format(outputfolder), vmin = vmin, vmax = vmax)#replace('hdf5','png'))
    neTCI = createInitialModel(datapack,antIdx = antIdx, timeIdx = timeIdx, dirIdx = dirIdx, zmax = tmax,spacing=sizeCell)
    #make uniform
    neTCI.m[:] = np.mean(neTCI.m)
    neTCI.save("{}/nePriori.hdf5".format(outputfolder))
    rays = calcRays(antennas,patches,times, arrayCenter, fixtime, phase, neTCI, datapack.radioArray.frequency, 
                    straightLineApprox, tmax, neTCI.nz)
    mTCI = neTCI.copy()
    K_ne = np.mean(mTCI.m)
    mTCI.m /= K_ne
    np.log(mTCI.m,out=mTCI.m)
    
    Nkernel = max(1,int(float(L_ne)/sizeCell))
    sigma_m = np.log(10.)#ne = K*exp(m+dm) = K*exp(m)*exp(dm), exp(dm) in (0.1,10) -> dm = (log(10) - log(0.1))/2.
    covC = CovarianceClass(mTCI,sigma_m,L_ne,7./2.)
    #uvw = UVW(location = datapack.radioArray.getCenter().earth_location,obstime = fixtime,phase = phase)
    #ants_uvw = antennas.transform_to(uvw).cartesian.xyz.to(au.km).value.transpose()
    #dirs_uvw = patches.transform_to(uvw).cartesian.xyz.value.transpose()
    F0 = precondition(neTCI, datapack,antIdx=antIdx, dirIdx=dirIdx, timeIdx = timeIdx)
    #
    dsk = {}
    for n in range(int(N)):
        #g_n
        dsk['store_forwardEq{}'.format(n)] = (store_forwardEq,resettable,'outputfolder',n,'templateDatapack','antIdx','timeIdx','dirIdx','rays',
                                              'K_ne','pull_m{}'.format(n),'i0')
        dsk['pull_forwardEq{}'.format(n)] = (pull_forwardEq,resettable,'store_forwardEq{}'.format(n),'antIdx','timeIdx','dirIdx')
        #gradient
        dsk['store_gamma{}'.format(n)] = (store_gamma,resettable,'outputfolder',n,'rays', 'pull_forwardEq{}'.format(n), 'dobs', 'i0', 'K_ne', 
                                        'pull_m{}'.format(n),'mprior', 'CdCt', 'sigma_m', 'Nkernel', 'sizeCell', 'covC')
        dsk['pull_gamma{}'.format(n)] = (pull_gamma,resettable,'store_gamma{}'.format(n))
        #m update
        dsk['store_m{}'.format(n+1)] = (store_m,resettable,'outputfolder',n+1,'pull_m{}'.format(n),'pull_phi{}'.format(n),'rays',
                                    'K_ne','i0','pull_forwardEq{}'.format(n),'dobs','CdCt','stateFile')
        dsk['pull_m{}'.format(n+1)] = (pull_m,resettable,'store_m{}'.format(n+1))
        dsk['plot_m{}'.format(n+1)] = (plot_model,'outputfolder',n+1,'pull_m{}'.format(n+1),'mprior','K_ne')
        dsk['plot_gamma{}'.format(n)] = (plot_gamma,'outputfolder',n,'pull_gamma{}'.format(n))
        #phi
        dsk['pull_phi{}'.format(n)] = (pull_Fdot,resettable,'store_F{}(gamma{})'.format(n,n))
        dsk['store_F{}(gamma{})'.format(n+1,n+1)] = (store_Fdot,resettable,'outputfolder', n+1, n+1 ,
                                                     'pull_F{}(gamma{})'.format(n,n+1),
                                                     'pull_gamma{}'.format(n+1),
                                                     'beta{}'.format(n),
                                                     'dm{}'.format(n),
                                                     'dgamma{}'.format(n),
                                                     'v{}'.format(n),
                                                     'sigma_m','L_m'
                                                    )
        for i in range(1,n+1):
            dsk['store_F{}(gamma{})'.format(i,n+1)] = (store_Fdot, resettable,'outputfolder',i, n+1 ,
                                                     'pull_F{}(gamma{})'.format(i-1,n+1),
                                                     'pull_gamma{}'.format(n+1),
                                                     'beta{}'.format(i-1),
                                                     'dm{}'.format(i-1),
                                                     'dgamma{}'.format(i-1),
                                                     'v{}'.format(i-1),
                                                       'sigma_m','L_m'
                                                    )
            dsk['pull_F{}(gamma{})'.format(i,n+1)] = (pull_Fdot,resettable,'store_F{}(gamma{})'.format(i,n+1))
        #should replace for n=0
        dsk['store_F0(gamma{})'.format(n)] = (store_F0dot, resettable,'outputfolder',n, 'pull_F0','pull_gamma{}'.format(n))
        dsk['pull_F0(gamma{})'.format(n)] = (pull_Fdot,resettable,'store_F0(gamma{})'.format(n))
#         #epsilon_n       
#         dsk['ep{}'.format(n)] = (calcEpsilon,n,'pull_phi{}'.format(n),'pull_m{}'.format(n),'rays',
#                                     'K_ne','i0','pull_forwardEq{}'.format(n),'dobs','CdCt')
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
    dsk['covC'] = covC
    #calc rays
    #dsk['rays'] = (calcRays_dask,'antennas','patches','times', 'arrayCenter', 'fixtime', 'phase', 'neTCI', 'frequency',  'straightLineApprox','tmax')
    dsk['rays'] = rays
    dsk['outputfolder'] = outputfolder
    dsk['resettable'] = resettable
    dsk['stateFile'] = stateFile
    
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
    from RealData import DataPack
    from AntennaFacetSelection import selectAntennaFacets
    from dask.diagnostics import Profiler, ResourceProfiler, CacheProfiler
    from dask.diagnostics import visualize
    #from InitialModel import createTurbulentlModel
    i0 = 0
    datapack = DataPack(filename="output/test/simulate/simulate_3/datapackSim.hdf5")
    #datapack = DataPack(filename="output/test/datapackObs.hdf5")
    #flags = datapack.findFlaggedAntennas()
    #datapack.flagAntennas(flags)
    datapackSel = selectAntennaFacets(20, datapack, antIdx=-1, dirIdx=-1, timeIdx = np.arange(1))
    #pertTCI = createTurbulentlModel(datapackSel,antIdx = -1, timeIdx = -1, dirIdx = -1, zmax = 1000.)
    L_ne = 25.
    sizeCell = 5.
    dsk = createBFGSDask(False, "output/test/bfgs_dask_3/", 5,datapackSel,L_ne,sizeCell,i0, antIdx=-1, dirIdx=-1, timeIdx = np.arange(1))
    #dot_graph(dsk,filename="{}/BFGS_graph".format(outputfolder),format='png')
    #dot_graph(dsk,filename="{}/BFGS_graph".format(outputfolder),format='svg')
    #client = Client()
    #with TrackingCallbacks():
    with Profiler() as prof, ResourceProfiler(dt=0.25) as rprof, CacheProfiler() as cprof:
        get(dsk,['plot_m5'])
    visualize([prof,rprof,cprof])
    

