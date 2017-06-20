
# coding: utf-8

# In[ ]:

import dask.array as da
import numpy as np
from dask.dot import dot_graph
#from dask.multiprocessing import get
from dask import delayed
from dask.distributed import Client
from dask import get
from functools import partial
from time import sleep, clock
from scipy.integrate import simps
from scipy.special import gamma
from functools import partial

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
from PlotTools import animateTCISlices


outputfolder = 'output/test/MH_7'



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
    import pylab as plt
    datapack = DataPack(filename=filename)
    g = datapack.get_dtec(antIdx=antIdx,timeIdx=timeIdx,dirIdx = dirIdx)
    return g

def negloglike(g,dobs,CdCt):
    L1 = g-dobs
    L1 *= L1
    #np.abs(L2,out=L2)
    #L2 = np.power(L2,2.,out=L2)
    L1 /= CdCt
    np.sqrt(L1,out=L1)
    return np.sum(L1)
    #return np.sum(L2)/2.
def likelihood(g,dobs,CdCt):
    #print(g,dobs,g-dobs)
    neglog = negloglike(g,dobs,CdCt)
    return np.exp(-neglog)

def MHWalker(acceptanceGoal,walkerId,Nmax,mTCI,L_ne,sizeCell,i0, antIdx, dirIdx, timeIdx,
             templateDatapack,rays,K_ne,dobs,CdCt,vmin,vmax,neTCI):
    i = 1
    accepted = 0
    mAccepted = np.zeros([mTCI.m.size,Nmax],dtype=np.double)
    m_i = mTCI.m.copy()
    mAccepted[:,0] = m_i
    mML = mTCI.m.copy()
    g = forwardEquation(rays,K_ne,mTCI,i0)
    Si = negloglike(g,dobs,CdCt)
    Li = np.exp(-Si)
    maxL = Li
    #sampling
    lvec = (np.fft.fftfreq(mTCI.nx,d=mTCI.xvec[1]-mTCI.xvec[0]))
    mvec = (np.fft.fftfreq(mTCI.ny,d=mTCI.yvec[1]-mTCI.yvec[0]))
    nvec = (np.fft.fftfreq(mTCI.nz,d=mTCI.zvec[1]-mTCI.zvec[0]))
    L_,M_,N_ = np.meshgrid(lvec,mvec,nvec,indexing='ij')
    R2 = L_**2 + M_**2 + N_**2
    theta1 = np.log(1.5)
    theta2 = L_ne
    theta3 = 11./2.
    omega = theta1*theta2**3/gamma(theta3) /np.pi**(3./2.) * (1. + theta2**2 *R2)**(-(theta3 + 3./2.))
    np.sqrt(omega,out=omega)
    V = (mTCI.xvec[-1]-mTCI.xvec[0])*(mTCI.yvec[-1]-mTCI.yvec[0])*(mTCI.zvec[-1]-mTCI.zvec[0])
    omega /= V
    while accepted < acceptanceGoal and i < Nmax:
        #cycle = 1.05 + 1.2*np.cos((i%Ncycle)*np.pi/(Ncycle-1)/2.)
        #theta1 = np.log(cycle)
        dM = np.fft.fftn(np.random.normal(size=L_.shape))
        dM *= omega
        dM = np.fft.ifftn(dM).real.ravel('C')
        dM *= theta1/np.max(dM)
        mTCI.m = m_i + dM
        g = forwardEquation(rays,K_ne,mTCI,i0)
        Sj = negloglike(g,dobs,CdCt)
        Lj = np.exp(-Sj)
        if Sj < Si or np.log(np.random.uniform()) < Si - Sj:
            mAccepted[:,i] += mTCI.m
            templateDatapack.set_dtec(g,antIdx=antIdx,timeIdx=timeIdx,dirIdx = dirIdx)
            templateDatapack.save("{}/g{}-{}.hdf5".format(outputfolder,walkerId,accepted))
            plotDataPack(templateDatapack,antIdx=antIdx,timeIdx=timeIdx,dirIdx = dirIdx,
            figname='{}/g{}-{}'.format(outputfolder,walkerId,accepted), vmin = vmin, vmax = vmax)#replace('hdf5','png'))
            np.exp(mTCI.m, out=neTCI.m)
            neTCI.m *= K_ne
            neTCI.save("{}/m{}-{}.hdf5".format(outputfolder,walkerId,accepted))
            Si = Sj
            Li = Lj
            accepted += 1
        else:
            mAccepted[:,i] += mAccepted[:,i-1]
            mTCI.m -= dM
        if Lj > maxL:
            print("New max L = {}".format(Lj))
            maxL = Lj
            mML = mTCI.m.copy()
        i += 1
    mAccepted = mAccepted[:,:i]
    if accepted == acceptanceGoal:
        print("Converged in {} steps".format(i))
    print("Acceptance: {}, rate : {}".format(accepted,float(accepted)/i))
    return mAccepted
    

def metropolisHastings(binning,datapack,L_ne,sizeCell,i0, antIdx=-1, dirIdx=-1, timeIdx = [0]):
    print("Using output folder: {}".format(outputfolder))
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
        CdCt = np.ones([Na,1,Nd],dtype=np.double)*var
    else:
        dt = times[1].gps - times[0].gps
        print("Averaging down window of length {} seconds [{} timestamps]".format(dt*Nt, Nt))
        CdCt = np.stack([np.var(dobs,axis=1)],axis=1) 
        dobs = np.stack([np.mean(dobs,axis=1)],axis=1)
        timeIdx = [Nt>>1]
        times,timestamps = datapack.get_times(timeIdx=timeIdx)
        Nt = len(times)
    #CdCt = np.ones([Na,1,Nd],dtype=np.double)*0.01**2
    CdCt += 1e-15
    print("CdCt: {} +- {}".format(np.mean(CdCt), np.std(CdCt)))
    vmin = np.min(datapack.get_dtec(antIdx = antIdx, timeIdx = timeIdx, dirIdx = dirIdx))
    vmax = np.max(datapack.get_dtec(antIdx = antIdx, timeIdx = timeIdx, dirIdx = dirIdx))
    plotDataPack(datapack,antIdx=antIdx,timeIdx=timeIdx,dirIdx = dirIdx,
            figname='{}/dobs'.format(outputfolder), vmin = vmin, vmax = vmax)#replace('hdf5','png'))
    #createPrior
    neTCI = createInitialModel(datapack,antIdx = antIdx, timeIdx = timeIdx, dirIdx = dirIdx, zmax = tmax,spacing=sizeCell)
    neTCI.save("{}/nePriori.hdf5".format(outputfolder))
    rays = calcRays(antennas,patches,times, arrayCenter, fixtime, phase, neTCI, datapack.radioArray.frequency, 
                    straightLineApprox, tmax, neTCI.nz)
    K_ne = np.mean(neTCI.m)
    mTCI = neTCI.copy()
    mTCI.m /= K_ne
    np.log(mTCI.m,out=mTCI.m)
    templateDatapack = datapack.clone()
    numWalkers = 6
    dsk = {"MHWalker-{}".format(id): (MHWalker,binning**2,id,int(1000),mTCI,
                                     L_ne,sizeCell,i0, antIdx, dirIdx, timeIdx,templateDatapack,rays,
                                     K_ne,dobs,CdCt,vmin,vmax,neTCI) for id in range(numWalkers)}
    
    client = Client()
    modelledAccepted = np.concatenate(client.get(dsk,["MHWalker-{}".format(id) for id in range(numWalkers)]),axis=1)
    meanModel = np.mean(mAccepted,axis=1)
    mTCI.m = meanModel
    np.exp(mTCI.m, out=neTCI.m)
    neTCI.m *= K_ne
    neTCI.save("{}/meanModel.hdf5".format(outputfolder))
    g = forwardEquation_dask(rays,K_ne,mTCI,i0)
    templateDatapack.set_dtec(g,antIdx=antIdx,timeIdx=timeIdx,dirIdx = dirIdx)
    templateDatapack.save("{}/post_g.hdf5".format(outputfolder))
    plotDataPack(templateDatapack,antIdx=antIdx,timeIdx=timeIdx,dirIdx = dirIdx,
    figname='{}/post_g'.format(outputfolder), vmin = vmin, vmax = vmax)#replace('hdf5','png'))
    #animateTCISlices(mTCI,"{}/meanModelPlot".format(outputfolder),numSeconds=20.)
    
    #covariance
    varModel = np.var(mAccepted,axis=1)
    mTCI.m = varModel
    mTCI.save("{}/varModel.hdf5".format(outputfolder))
    #animateTCISlices(mTCI,"{}/varModelPlot".format(outputfolder),numSeconds=20.)
    
    
if __name__ == '__main__':
    import os
    try:
        os.makedirs(outputfolder)
    except:
        pass
    from RealData import DataPack
    from AntennaFacetSelection import selectAntennaFacets
    #from InitialModel import createTurbulentlModel
    i0 = 0
    datapack = DataPack(filename="output/test/simulate/simulate_0/datapackSim.hdf5")
    #flags = datapack.findFlaggedAntennas()
    #datapack.flagAntennas(flags)
    datapackSel = selectAntennaFacets(15, datapack, antIdx=-1, dirIdx=-1, timeIdx = np.arange(1))
    #pertTCI = createTurbulentlModel(datapackSel,antIdx = -1, timeIdx = -1, dirIdx = -1, zmax = 1000.)
    L_ne = 100.
    sizeCell = 5.
    #metropolisHastings(25,datapackSel,L_ne,sizeCell,i0, antIdx=-1, dirIdx=-1, timeIdx = np.arange(1))  
    m = TriCubic(filename='output/test/MH_6/m0-3.hdf5')
    neTCI = createInitialModel(datapack,antIdx = antIdx, timeIdx = timeIdx, dirIdx = dirIdx, zmax = tmax,spacing=sizeCell)
    m.m -=
    animateTCISlices(m,"output/test/MH_6/m0-3-fig",numSeconds=20.)


# In[ ]:



