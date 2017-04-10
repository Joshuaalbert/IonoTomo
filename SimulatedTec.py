
# coding: utf-8

# In[1]:

import astropy.units as au
import astropy.time as at
import astropy.coordinates as ac
import numpy as np
import pp

from RealData import DataPack,plotDataPack
from FermatClass import Fermat
from PointingFrame import Pointing
from UVWFrame import UVW
from IRI import aPrioriModel
from TricubicInterpolation import TriCubic
from ProgressBarClass import ProgressBar
from InversionUpgraded import determineInversionDomain


def simulateDtec(dataPackObs,timeIdx,numThreads,straightLineApprox=True):
    '''Fill out the dtec values in an initialized ``DataPack`` object `dataPack`.
    ionosphere model is a ``TriCubicInterpolator`` object with electron density values of ionosphere
    '''
    dataPack = dataPackObs.clone()
    zmax = 1000.
    antIdx = -1
    refAntIdx = 0
    dirIdx = -1
    L_ne,sigma_ne_factor = 15.,0.1
    def ppCastRay(origins, directions, neTCI, frequency, tmax, N, straightLineApprox):
        rays,cache = ParallelInversionProducts.castRay(origins, directions, neTCI, frequency, tmax, N, straightLineApprox)
        return rays, cache
    def ppCalculateTEC(rays, muTCI,K_e):
        tec,cache = ParallelInversionProducts.calculateTEC(rays, muTCI,K_e)
        return tec,cache
    def ppCalculateTEC_modelingError(rays, muTCI,K_e,sigma,frequency):
        tec,sigma_tec, cache = ParallelInversionProducts.calculateTEC_modelingError(rays, muTCI,K_e,sigma,frequency)
        return tec,sigma_tec, cache
    def ppInnovationPrimaryCalculation_exponential(rayPairs,muTCI,K_e,L_ne,sigma_ne_factor):
        outS_primary, cache = ParallelInversionProducts.innovationPrimaryCalculation_exponential(rayPairs,muTCI,K_e,L_ne,sigma_ne_factor)
        return outS_primary, cache
    def ppInnovationAdjointPrimaryCalculation_exponential(rays,muTCI,K_e,L_ne,sigma_ne_factor):
        outCmGt_primary, cache = ParallelInversionProducts.innovationAdjointPrimaryCalculation_exponential(rays,muTCI,K_e,L_ne,sigma_ne_factor)
        return outCmGt_primary, cache
    # get setup from dataPack
    antennas,antennaLabels = dataPack.get_antennas(antIdx = antIdx)
    times,timestamps = dataPack.get_times(timeIdx=timeIdx)
    patches, patchNames = dataPack.get_directions(dirIdx=dirIdx)
    dataPackObs.setReferenceAntenna(antennaLabels[refAntIdx])
    dataPack.setReferenceAntenna(antennaLabels[refAntIdx])
    Na = len(antennas)
    Nt = len(times)
    Nd = len(patches)  
    print("Using radio array {}".format(dataPack.radioArray))
    phase = dataPack.getCenterDirection()
    print("Using phase center as {} {}".format(phase.ra,phase.dec))
    fixtime = times[0]
    print("Fixing frame at {}".format(fixtime.isot))
    uvw = UVW(location = dataPack.radioArray.getCenter().earth_location,obstime=fixtime,phase = phase)
    print("Elevation is {}".format(uvw.elevation))
    zenith = dataPack.radioArray.getSunZenithAngle(fixtime)
    print("Sun at zenith angle {}".format(zenith))
    print("Creating ionosphere model...")
    xvec,yvec,zvec = determineInversionDomain(10.,antennas, patches,
                                              UVW(location = dataPack.radioArray.getCenter().earth_location,
                                                  obstime = fixtime, phase = phase), 
                                              zmax, padding = 5)
    X,Y,Z = np.meshgrid(xvec,yvec,zvec,indexing='ij')
    print("Nx={} Ny={} Nz={} number of cells: {}".format(len(xvec),len(yvec),len(zvec),np.size(X)))
    coords = ac.SkyCoord(X.flatten()*au.km,Y.flatten()*au.km,Z.flatten()*au.km,frame=uvw).transform_to('itrs').earth_location.to_geodetic('WGS84')
    heights = coords[2].to(au.km).value
    nePrior = aPrioriModel(heights,zenith).reshape(X.shape)
    neTCI = TriCubic(xvec,yvec,zvec,nePrior,useCache = True,default=None)
    K_e = np.mean(neTCI.m)
    neMean = neTCI.m.copy()
    
    X,Y,Z = neTCI.getModelCoordinates()
    a = 1e12
    scale=20.
    x0 = 0
    y0 = 0
    z0 = 300
    vx = 10.
    vy=0
    vz=0
    nePrior = neTCI.m.copy()
    for iter in range(len(timeIdx)):
        dne = a*np.exp(-np.sqrt((X-x0-vx*iter)**2 + (Y-y0-vy*iter)**2 + (Z-z0-vz*iter)**2)/2./scale**2)
        dne += a*np.exp(-np.sqrt((X-x0)**2 + (Y-y0)**2 + (Z-100)**2)/2./scale**2)
        neTCI.m = nePrior + dne
        neTCI.clearCache()
        times,timestamps = dataPack.get_times(timeIdx=[iter])
        Nt = len(times)
        #divide by direction
        print("Spliting up jobs into directions")
        progress = ProgressBar(Nd, fmt=ProgressBar.FULL)
        batches = {}
        k = 0
        while k < Nd:
            origins = []
            directions = []
            patchDir = patches[k]
            j = 0
            while j < Nt:
                time = times[j]
                pointing = Pointing(location = dataPack.radioArray.getCenter().earth_location,
                                    obstime = time, fixtime = fixtime, phase = phase)
                direction = patchDir.transform_to(pointing).cartesian.xyz.value
                ants = antennas.transform_to(pointing).cartesian.xyz.to(au.km).value.transpose()
                origins.append(ants)
                i = 0
                while i < Na:
                    directions.append(direction)
                    i += 1
                j += 1
            batches[k] = {'origins':np.vstack(origins),
                          'directions':np.vstack(directions)}
            progress(k)
            k += 1
        progress.done()
        jobs = {}
        print("Creating ray cast job server")
        job_server_raycast = pp.Server(numThreads, ppservers=())
        print("Submitting {} ray cast jobs".format(len(batches)))
        if straightLineApprox:
            print("Using straight line approximation")
        else:
            print("Using Fermats Principle")
        #get rays
        jobs = {}
        k = 0
        while k < Nd:
            job = job_server_raycast.submit(ppCastRay,
                           args=(batches[k]['origins'], batches[k]['directions'], neTCI, dataPack.radioArray.frequency, 
                                 zmax, 100, straightLineApprox),
                           depfuncs=(),
                           modules=('ParallelInversionProducts',))
            jobs[k] = job
            k += 1
        print("Waiting for ray cast jobs to finish.")
        rays = {}
        k = 0
        while k < Nd:
            rays[k],cache = jobs[k]() 
            neTCI.cache.update(cache)
            k += 1   
        job_server_raycast.print_stats()
        #Calculate TEC
        muTCI = neTCI.copy()
        mu = np.log(neTCI.m/K_e)
        muTCI.m = mu
        muTCI.clearCache()
        print("Creating tec/Ct integration job server")
        job_server_tec = pp.Server(numThreads, ppservers=())
        #plot rays
        #plotWavefront(neTCI,rays[0]+rays[1],save=False,saveFile=None,animate=False)
        print("Submitting {} tec calculation jobs".format(len(batches)))
        #get rays
        jobs = {}
        k = 0
        while k < Nd:
            job = job_server_tec.submit(ppCalculateTEC_modelingError,
                           args=(rays[k], muTCI, K_e,sigma_ne_factor,dataPack.radioArray.frequency),
                           depfuncs=(),
                           modules=('ParallelInversionProducts',))
            jobs[k] = job
            k += 1
        print("Waiting for jobs to finish.")
        dtec_threads = {}
        Ct_threads = {}
        k = 0
        while k < Nd:
            dtec_threads[k],Ct_threads[k],muCache = jobs[k]()  
            muTCI.cache.update(muCache)
            k += 1 
        job_server_tec.print_stats()
        print("Size of muTCI cache: {}".format(len(muTCI.cache)))
        print("Computing dtec from tec products")
        progress = ProgressBar(Nd, fmt=ProgressBar.FULL)
        dtec = np.zeros([Na,Nt,Nd],dtype=np.double)
        Ct = np.zeros([Na,Nt,Nd],dtype=np.double)
        k = 0
        while k < Nd:
            c = 0
            j = 0
            while j < Nt:
                i = 0
                while i < Na:
                    dtec[i,j,k] = dtec_threads[k][c]
                    Ct[i,j,k] = Ct_threads[k][c]
                    c += 1
                    i += 1
                j += 1
            progress(k)
            k += 1
        progress.done()
        dtec += np.random.normal(size=dtec.shape)*0.1
        dataPack.set_dtec(dtec,antIdx=antIdx,timeIdx=[iter], dirIdx=dirIdx,refAnt=None)
        job_server_tec.destroy()
        job_server_raycast.destroy()
    dataPack.save("simulatedObs.dill")
    

if __name__ == '__main__':
    from RealData import PrepareData, DataPack
    dataPackObs = PrepareData(infoFile='SB120-129/WendysBootes.npz',
                           dataFolder='SB120-129/',
                           timeStart = 0, timeEnd = 10,
                           arrayFile='arrays/lofar.hba.antenna.cfg',load=True,numThreads=4)
    dataPackObs.setReferenceAntenna('CS501HBA1')
    flags = dataPackObs.findFlaggedAntennas()
    dataPackObs.flagAntennas(flags)
    
    dataPack = simulateDtec(dataPackObs,np.arange(10),6,straightLineApprox=True)
    dataPack = DataPack(filename="simulatedObs.dill")
    plotDataPack(dataPack)
    #plotDataPack(dataPack)
    #dataPack.save("simulated.dill")
                
                
                
    


# In[ ]:



