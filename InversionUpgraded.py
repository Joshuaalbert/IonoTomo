
# coding: utf-8

# In[1]:

import astropy.units as au
import astropy.time as at
import astropy.coordinates as ac
import numpy as np
import pp
from time import clock
import pylab as plt
import h5py

from RealData import DataPack,plotDataPack
from FermatClass import Fermat
from PointingFrame import Pointing
from UVWFrame import UVW
from IRI import aPrioriModel
from TricubicInterpolation import TriCubic
from ProgressBarClass import ProgressBar

def getDatumIdx(antIdx,timeIdx,dirIdx,numAnt,numTimes):
    '''standarizes indexing'''
    idx = antIdx + numAnt*(timeIdx + numTimes*dirIdx)
    return int(idx)

def getDatum(datumIdx,numAnt,numTimes):
    antIdx = datumIdx % numAnt
    timeIdx = (datumIdx - antIdx)/numAnt % numTimes
    dirIdx = (datumIdx - antIdx - numAnt*timeIdx)/numAnt/numTimes
    return int(antIdx),int(timeIdx),int(dirIdx)


def circ_conv(signal,kernel):
    return np.abs(np.real(np.fft.fft( np.fft.ifft(signal) * np.fft.ifft(kernel) )))
    

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

def invertSingleTime(dataPackObs,numThreads,datafolder,straightLineApprox=True,
                     antIdx=np.arange(10),timeIdx=[0],dirIdx=np.arange(10)):
    '''Invert the dtec in dataPack'''
    #Set up datafolder
    import os
    try:
        os.makedirs(datafolder)
    except:
        pass
    #all products including external links
    fall = h5py.File("{}/AllProducts.hdf5".format(datafolder),"w")   
    #hyperparameters
    refAntIdx = 0
    zmax = 1000.
    L_ne,sigma_ne_factor = 20.,0.1
    def ppCastRay(origins, directions, neTCI, frequency, tmax, N, straightLineApprox):
        rays = ParallelInversionProducts.castRay(origins, directions, neTCI, frequency, tmax, N, straightLineApprox)
        return rays
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
    dataPack = dataPackObs.clone()
    antennas,antennaLabels = dataPack.get_antennas(antIdx = antIdx)
    patches, patchNames = dataPack.get_directions(dirIdx=dirIdx)
    times,timestamps = dataPack.get_times(timeIdx=timeIdx)
    dataPackObs.setReferenceAntenna(antennaLabels[refAntIdx])
    dataPack.setReferenceAntenna(antennaLabels[refAntIdx])
    dataPackObs.save("{}/dataobs.hdf5".format(datafolder))
    Na = len(antennas)
    Nt = len(times)
    Nd = len(patches)  
    #Setting up ionosphere to use
    print("Using radio array {}".format(dataPack.radioArray))
    phase = dataPack.getCenterDirection()
    print("Using phase center {} {}".format(phase.ra,phase.dec))
    fixtime = times[Nt>>1]
    print("Fixing frame at {}".format(fixtime.isot))
    uvw = UVW(location = dataPack.radioArray.getCenter().earth_location,obstime = fixtime,phase = phase)
    print("Elevation is {}".format(uvw.elevation))
    zenith = dataPack.radioArray.getSunZenithAngle(fixtime)
    print("Sun at zenith angle {}".format(zenith))
    print("Creating ionosphere model...")
    xvec,yvec,zvec = determineInversionDomain(5.,antennas, patches,
                                              UVW(location = dataPack.radioArray.getCenter().earth_location,
                                                  obstime = fixtime, phase = phase), 
                                              zmax, padding = 10)
    X,Y,Z = np.meshgrid(xvec,yvec,zvec,indexing='ij')
    print("Nx={} Ny={} Nz={} number of cells: {}".format(len(xvec),len(yvec),len(zvec),np.size(X)))
    coords = ac.SkyCoord(X.flatten()*au.km,Y.flatten()*au.km,Z.flatten()*au.km,frame=uvw).transform_to('itrs').earth_location.to_geodetic('WGS84')
    heights = coords[2].to(au.km).value
    nePrior = aPrioriModel(heights,zenith).reshape(X.shape)
    neTCI = TriCubic(xvec,yvec,zvec,nePrior,useCache = True,default=None)
    neTCI.save("{}/apriori_neModel.hdf5".format(datafolder))
    K_e = np.mean(neTCI.m)
    fall["/"].attrs['K_e'] = K_e
    neMean = neTCI.m.copy()
    kSize = min(4,Nt)
    print("Computing Cd over a window of {} seconds".format(times[kSize-1].gps - times[0].gps))
    dobs = dataPackObs.get_dtec(antIdx = antIdx,dirIdx=dirIdx,timeIdx=timeIdx)
    kernel = np.zeros(Nt)
    kernel[0:kSize] = 1./kSize#flat
    Cd = np.zeros([Na,Nt,Nd])
    i = 0 
    while i < Na:
        k = 0
        while k < Nd:
            #Cd[i,:,k] = np.convolve(dobs[i,:,k]**2,kernel,mode='same') - np.convolve(dobs[i,:,k],kernel,mode='same')**2
            Cd[i,:,k] = circ_conv(dobs[i,:,k]**2,kernel)-(circ_conv(dobs[i,:,k],kernel))**2
            Cd[i,:,k] *= np.var(dobs[i,:,k])/(np.mean(Cd[i,:,k])+1e-15)
            k += 1
        #print("{}: dtec={} C_D={} C_T={} S/N={}".format(antennaLabels[i],dobs[i,:,0],Cd[i,:,0],Ct[i,:,0],dobs[i,:,0]/np.sqrt(Cd[i,:,0]+Ct[i,:,0])))
        i += 1
    Cd[np.isnan(Cd)] = 0.
    fall["Cd"] = Cd
    fall.flush()
    #np.save("{}/Cd.npy".format(datafolder),Cd)
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
            direction = patchDir.transform_to(pointing).cartesian.xyz.value.flatten()
            ants = antennas.transform_to(pointing).cartesian.xyz.to(au.km).value.transpose()
            origins.append(ants)
            i = 0
            while i < Na:
                directions.append(direction)
                i += 1
            j += 1
        batches[k] = {'origins':np.vstack(origins),
                      'directions':np.vstack(directions)}
        fall['batches/{}/origins'.format(k)] = batches[k]['origins']
        fall['batches/{}/directions'.format(k)] = batches[k]['directions']
        progress(k)
        k += 1
    fall.flush()
    #np.save("{}/origins_directions.npy".format(datafolder),batches)
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
        rays[k] = jobs[k]()
        for rayIdx in range(len(rays[k])):
            fall['rays/{}/{}/x'.format(k,rayIdx)] = rays[k][rayIdx]['x']
            fall['rays/{}/{}/y'.format(k,rayIdx)] = rays[k][rayIdx]['y']
            fall['rays/{}/{}/z'.format(k,rayIdx)] = rays[k][rayIdx]['z']
            fall['rays/{}/{}/s'.format(k,rayIdx)] = rays[k][rayIdx]['s']
        k += 1
    fall.flush()
    job_server_raycast.print_stats()
    job_server_raycast.destroy()
    #print("Generating weight matrix")
    #progress = ProgressBar(Nd, fmt=ProgressBar.FULL)
    #weight = np.zeros(np.size(neTCI.m),dtype=np.double)
    #X,Y,Z = neTCI.getModelCoordinates()
    #ar = np.arange(np.size(X))
    #k = 0
    #while k < Nd:
    #    for ray in rays[k]:
    #        xmask = np.any(np.abs(np.subtract.outer(X,ray['x'])) < 5.,axis=1)
    #        ymask = np.any(np.abs(np.subtract.outer(Y[xmask],ray['y'])) < 5., axis=1)
    #        zmask = np.any(np.abs(np.subtract.outer(Z[xmask][ymask],ray['z'])) < 5., axis=1)
    #        weight[ar[xmask][ymask][zmask]] += 1
    #    progress(k)
    #    k += 1
    #progress.done()   
    #weight /= np.max(weight)
    #weight[weight==0] = np.min(weight[weight>0])
    #plt.hist(weight,bins=100)
    #plt.show()
    #fall['weight'] = weight
    #fall.flush()
    #np.save("{}/rays.npy".format(datafolder),rays)
    mu = np.log(neTCI.m/K_e)
    muTCI = neTCI.copy()
    muTCI.m = mu
    iteration = 0
    parmratios = []
    progress = ProgressBar(20, fmt=ProgressBar.FULL)
    while iteration < 20:
        #Calculate TEC
        muTCI.clearCache()
        neTCI.m = K_e*np.exp(muTCI.m)
        neTCI.clearCache()
        neTCI.save("{}/neModel-{}.hdf5".format(datafolder,iteration))
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
        job_server_tec.destroy()
        print("Size of muTCI cache: {}".format(len(muTCI.cache)))
        print("Computing dtec from tec products")
        #progress = ProgressBar(Nd, fmt=ProgressBar.FULL)
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
            #progress(k)
            k += 1
        #progress.done()
        fall["iterations/{}/Ct".format(iteration)] = Ct
        dataPack.set_dtec(dtec,antIdx=antIdx,timeIdx=timeIdx, dirIdx=dirIdx,refAnt=None)
        dataPack.save("{}/dataPack-{}.hdf5".format(datafolder,iteration))
        d = dataPack.get_dtec(antIdx = antIdx,dirIdx=dirIdx,timeIdx=timeIdx)
        #calculate innovation matrix
        print("Dividing innovation matrix S = Cd + Ct + G.Cm.G^t into ray pairs")
        numRays = Na*Nt*Nd
        numRayPairs = ((numRays - 1)*numRays)/2. + numRays
        rayPairs = {i:[] for i in range(numThreads)}
        rayPairsMap = {i:[] for i in range(numThreads)}
        count = 0
        #progress = ProgressBar(numRayPairs, fmt=ProgressBar.FULL)
        h1 = 0
        while h1 < numRays:
            i1,j1,k1 = getDatum(h1,Na,Nt)
            #rays are direction, time, antenna ordered
            ray1 = rays[k1][j1*Na + i1]
            h2 = h1
            while h2 < numRays:
                i2,j2,k2 = getDatum(h2,Na,Nt)
                ray2 = rays[k2][j2*Na + i2]
                rayPairs[count%numThreads].append([ray1,ray2])
                rayPairsMap[count%numThreads].append([h1,h2])
                count += 1
                h2 += 1
            #progress(count)
            h1 += 1
        #progress.done()    
        print("Creating innovation matrix job server")
        job_server_innovation = pp.Server(numThreads, ppservers=())
        #plot rays
        #plotWavefront(neTCI,rays[0]+rays[1],save=False,saveFile=None,animate=False)
        print("Submitting {} innovation matrix primary calculation jobs".format(numThreads))
        #get rays
        jobs = {}
        threadIdx = 0
        while threadIdx < numThreads:
            job = job_server_innovation.submit(ppInnovationPrimaryCalculation_exponential,
                           args=(rayPairs[threadIdx],muTCI,K_e,L_ne,sigma_ne_factor),
                           depfuncs=(),
                           modules=('ParallelInversionProducts',))
            jobs[threadIdx] = job
            threadIdx += 1
        print("Waiting for jobs to finish.")
        S_primary = np.zeros([numRays,numRays],dtype=np.double)
        S = np.zeros([numRays,numRays],dtype=np.double)
        threadIdx = 0
        while threadIdx < numThreads:
            outPairs,cache = jobs[threadIdx]()  
            muTCI.cache.update(muCache)
            pairIdx = 0
            while pairIdx < len(rayPairsMap[threadIdx]):
                h1,h2 = rayPairsMap[threadIdx][pairIdx]
                S_primary[h1,h2] = outPairs[pairIdx]
                S_primary[h2,h1] = S_primary[h1,h2] 
                pairIdx += 1
            threadIdx += 1 
        job_server_innovation.print_stats()
        job_server_innovation.destroy()
        print("Size of muTCI cache: {}".format(len(muTCI.cache)))
        print("Calculating innovation from primary")
        #progress = ProgressBar(numRayPairs, fmt=ProgressBar.FULL)
        #refAntIdx = int(np.arange(len(antennaLabels))[antennaLabels == dataPack.refAnt][0])
        count = 0
        h1 = 0
        while h1 < numRays:
            i1,j1,k1 = getDatum(h1,Na,Nt)
            h2 = h1
            while h2 < numRays:
                i2,j2,k2 = getDatum(h2,Na,Nt)
                if h1 == h2:
                    S[h1,h2] += Cd[i1,j1,k1] + Ct[i1,j1,k1]
                h2p = getDatumIdx(refAntIdx,j2,k2,Na,Nt)
                h1p = getDatumIdx(refAntIdx,j1,k1,Na,Nt)
                S[h1,h2] += S_primary[h1,h2]
                S[h1,h2] -= S_primary[h1,h2p]
                S[h1,h2] -= S_primary[h1p,h2]
                S[h1,h2] += S_primary[h1p,h2p]
                S[h2,h1] = S[h1,h2]
                count += 1
                h2 += 1
            #progress(count)
            h1 += 1
        #progress.done()
        fall["iterations/{}/S".format(iteration)] = S
        fall.flush()
        #np.save("S_full_alt.npy",S)
        print("Creating innovated adjoint (Cm.G^t).inv(S) job server")
        job_server_adjoint = pp.Server(numThreads, ppservers=())
        print("Submitting primary calculations CmGt_primary")
        jobs = {}
        k = 0
        while k < Nd:
            job = job_server_adjoint.submit(ppInnovationAdjointPrimaryCalculation_exponential,
                           args=(rays[k],muTCI,K_e,L_ne,sigma_ne_factor),
                           depfuncs=(),
                           modules=('ParallelInversionProducts',))
            jobs[k] = job
            k += 1
        print("Waiting for jobs to finish.")
        CmGt_primary_threads = {}
        k = 0
        while k < Nd:
            CmGt_primary_threads[k],muCache = jobs[k]()  
            muTCI.cache.update(muCache)
            k += 1 
        job_server_adjoint.print_stats()
        job_server_adjoint.destroy()
        print("Size of muTCI cache: {}".format(len(muTCI.cache)))
        print("Calculating CmGt from primary")
        CmGt = np.zeros([len(muTCI.m),numRays],dtype=np.double)
        #progress = ProgressBar(Nd, fmt=ProgressBar.FULL)
        k = 0
        while k < Nd:
            c = 0
            j = 0
            while j < Nt:
                #get all antennas from primary
                i = 0
                while i < Na:
                    h = getDatumIdx(i,j,k,Na,Nt)
                    CmGt[:,h] = CmGt_primary_threads[k][:,c]
                    c += 1
                    i += 1
                #subtract reference antenna from each antenna
                h0 = getDatumIdx(refAntIdx,j,k,Na,Nt)
                i = 0
                while i < Na:
                    h = getDatumIdx(i,j,k,Na,Nt)
                    CmGt[:,h] -= CmGt[:,h0]
                    i += 1
                j += 1
            #progress(k)
            k += 1
        #progress.done()
        print("Calculating innovation adjoint Y=Cm.Gt.inv(S)")
        Y = CmGt.dot(np.linalg.pinv(S))
        fall["iterations/{}/Y".format(iteration)] = Y
        fall.flush()
        print("Vectorizing dd and Cd+Ct")
        dd = np.zeros(numRays,dtype=np.double)
        CdCt=np.zeros([numRays,numRays],dtype=np.double)
        h = 0
        while h < numRays:
            i,j,k = getDatum(h,Na,Nt)
            dd[h] = dobs[i,j,k] - d[i,j,k]
            CdCt[h,h] = Cd[i,j,k] + Ct[i,j,k]
            h += 1    
        fall["iterations/{}/dd".format(iteration)] = dd
        fall["iterations/{}/CdCt".format(iteration)] = CdCt
        fall.flush()
        likelihood = np.exp(-dd.dot(np.linalg.pinv(CdCt).dot(dd))/2.) * np.sqrt(np.linalg.det(S))
        print("Current model likelihood rho(d)*sqrt(det(S)) = {}".format(likelihood))
        print("Calculating unscaled model development dm0=Y.(dobs-d)")
        dm0 = Y.dot(dd)
        print("Calculating epsilon_n = 2 (G.dm0)t.(Cd+Ct)^-1.(dobs - d) / (G.dm0)^t.(Cd+Ct)^-1.(G.dm0)")
        muCurr = muTCI.m.copy()
        muTCI.m = muTCI.m + 1e-5*dm0
        muTCI.clearCache()
        #neTCI.m = K_e*np.exp(muCurr + dm0)
        #plotWavefront(neTCI,rays[0]+rays[1],save=False,saveFile=None,animate=False)
        #plotWavefront(neTCI,rays[0]+rays[1],save=False,saveFile=None,animate=False)
        print("Submitting {} g(m + ep*dm0) calculation jobs".format(len(batches)))
        job_server_tec = pp.Server(numThreads, ppservers=())
        #get rays
        jobs = {}
        k = 0
        while k < Nd:
            job = job_server_tec.submit(ppCalculateTEC,
                           args=(rays[k], muTCI, K_e),
                           depfuncs=(),
                           modules=('ParallelInversionProducts',))
            jobs[k] = job
            k += 1
        print("Waiting for jobs to finish.")
        dtec_threads = {}
        k = 0
        while k < Nd:
            dtec_threads[k],muCache = jobs[k]()  
            muTCI.cache.update(muCache)
            k += 1 
        job_server_tec.print_stats()
        job_server_tec.destroy()
        muTCI.m = muCurr
        print("Size of muTCI cache: {}".format(len(muTCI.cache)))
        print("Computing G.dm0 with products")
        dtec_pert = np.zeros([Na,Nt,Nd],dtype=np.double)
        k = 0
        while k < Nd:
            c = 0
            j = 0
            while j < Nt:
                i = 0
                while i < Na:
                    #h = getDatumIdx(i,j,k,Na,Nt)
                    dtec_pert[i,j,k] = dtec_threads[k][c]
                    c += 1
                    i += 1
                j += 1
            k += 1
        dataPack.set_dtec(dtec_pert,antIdx=antIdx,timeIdx=timeIdx, dirIdx=dirIdx,refAnt=None)
        d_pert = dataPack.get_dtec(antIdx = antIdx,dirIdx=dirIdx,timeIdx=timeIdx)
        Gdm0 = np.zeros(numRays,dtype=np.double)
        k = 0
        while k < Nd:
            j = 0
            while j < Nt:
                i = 0
                while i < Na:
                    h = getDatumIdx(i,j,k,Na,Nt)
                    Gdm0[h] = (d_pert[i,j,k] - d[i,j,k])/1e-5
                    i += 1
                j += 1
            k += 1
        epsilon_n = (2*Gdm0.dot(np.linalg.pinv(CdCt).dot(dd))/Gdm0.dot(np.linalg.pinv(CdCt).dot(Gdm0)))
        
        if np.isnan(epsilon_n):
            epsilon_n = 1.
        epsilon_n = min(1,epsilon_n)
        print("Found epsilon_n = {}".format(epsilon_n))
        misfit0 = dd.dot(np.linalg.pinv(CdCt).dot(dd))
        print("Misfit0 = {}".format(misfit0))
        while np.abs(epsilon_n) > 1e-10:
            print("Testing epsilon_n = {}".format(epsilon_n))
            job_server_tec = pp.Server(numThreads, ppservers=())
            muTCI.m = muCurr + epsilon_n*dm0
            muTCI.clearCache()
            #get rays
            jobs = {}
            k = 0
            while k < Nd:
                job = job_server_tec.submit(ppCalculateTEC,
                               args=(rays[k], muTCI, K_e),
                               depfuncs=(),
                               modules=('ParallelInversionProducts',))
                jobs[k] = job
                k += 1
            #print("Waiting for jobs to finish.")
            dtec_threads = {}
            k = 0
            while k < Nd:
                dtec_threads[k],muCache = jobs[k]()  
                muTCI.cache.update(muCache)
                k += 1 
            #job_server_tec.print_stats()
            job_server_tec.destroy()
            muTCI.m = muCurr
            #print("Size of muTCI cache: {}".format(len(muTCI.cache)))
            #print("Computing G.dm0 with products")
            dtec_pert = np.zeros([Na,Nt,Nd],dtype=np.double)
            k = 0
            while k < Nd:
                c = 0
                j = 0
                while j < Nt:
                    i = 0
                    while i < Na:
                        #h = getDatumIdx(i,j,k,Na,Nt)
                        dtec_pert[i,j,k] = dtec_threads[k][c]
                        c += 1
                        i += 1
                    j += 1
                k += 1
            dataPack.set_dtec(dtec_pert,antIdx=antIdx,timeIdx=timeIdx, dirIdx=dirIdx,refAnt=None)
            d_pert = dataPack.get_dtec(antIdx = antIdx,dirIdx=dirIdx,timeIdx=timeIdx)
            dd_pert = np.zeros(numRays,dtype=np.double)
            h = 0
            while h < numRays:
                i,j,k = getDatum(h,Na,Nt)
                dd_pert[h] = dobs[i,j,k] - d_pert[i,j,k]
                h += 1   
            misfit = dd_pert.dot(np.linalg.pinv(CdCt).dot(dd_pert))
            print("Misfit = {}".format(misfit))
            if (misfit > misfit0):
                epsilon_n /= 2.
            else:
                print("Good epsilon_n = {}".format(epsilon_n))
                break;

        fall["iterations/{}".format(iteration)].attrs['epsilon_n'] = epsilon_n
        dm = epsilon_n*dm0
        muTCI.m = muCurr + dm
        neTCI.m = K_e*np.exp(muTCI.m)
        
        log10parmratio = np.log10(np.abs(dm/muCurr))
        fig=plt.figure()
        ax1 = plt.subplot(1,1,1)
        ax1.plot(log10parmratio)
        ax1.set_title("Log10 parm ratios, iter: {}".format(iteration))
        plt.savefig("{}/log10parmratios-{}.png".format(datafolder,iteration))
        parmratios.append(np.mean(log10parmratio))
        print("Mean Log10 of parameter ratio = {}".format(np.mean(log10parmratio)))
        
        #plt.hist(dm0,bins=100,alpha=0.2,color='green')
        #plt.show()
        fig,(ax1,ax2) = plt.subplots(1,2)
        ax1.hist(dm0,bins=100,alpha=0.2,color='green')
        ax1.set_yscale('log')
        ax2.hist(dm,bins=100,alpha=0.2,color='blue')
        ax2.set_yscale('log')
        plt.title("Unscale (green) scaled (blue) model developments: iter {}".format(iteration))
        plt.savefig("{}/modelDevelopments-{}.png".format(datafolder,iteration),format='png')
        plt.close()
        dataPack.set_dtec(dobs - d,antIdx=antIdx,timeIdx=timeIdx, dirIdx=dirIdx,refAnt=None)
        #dataPack.save("{}/dataPack-{}.hdf5".format(datafolder,iteration))
        plotDataPack(dataPack,antIdx=antIdx,timeIdx=timeIdx,dirIdx=dirIdx)
        data = neTCI.getShapedArray() - nePrior
        xy = np.mean(data,axis=2)
        yz = np.mean(data,axis=0)
        zx = np.mean(data,axis=1)
        vmin = np.min(xy)
        vmax = np.max(xy)
        fig,(ax1,ax2,ax3) = plt.subplots(1,3)
        im = ax1.imshow(xy,vmin=vmin,vmax=vmax,origin='lower',aspect='auto')
        ax2.imshow(yz,vmin=vmin,vmax=vmax,origin='lower',aspect='auto')
        ax3.imshow(zx,vmin=vmin,vmax=vmax,origin='lower',aspect='auto')
        plt.colorbar(im)
        plt.show()
        progress(iteration)
        fall.flush()
        iteration += 1
    progress.done()
    fall.close()
    neTCI.save("{}/neModel-final.hdf5".format(datafolder))
    plt.plot(parmratios)
    plt.title("Mean Log10 parameter ratios")
    plt.savefig("{}/logparamratios.png".format(datafolder),format='png')
    print("Time: {} seconds".format(clock() - startTime))
    
if __name__ == '__main__':
    from RealData import prepareDataPack,DataPack
    #dataPackObs = prepareDataPack('SB120-129/dtecData.hdf5',timeStart=0,timeEnd=4,
    #                       arrayFile='arrays/lofar.hba.antenna.cfg')
    #dataPackObs.setReferenceAntenna('CS501HBA1')
    dataPackObs = DataPack(filename="simulatedObs.hdf5")
    flags = dataPackObs.findFlaggedAntennas()
    dataPackObs.flagAntennas(flags)
    antennas,antennaLabels = dataPackObs.get_antennas(antIdx = -1)
    patches, patchNames = dataPackObs.get_directions(dirIdx=-1)
    phase = dataPackObs.getCenterDirection()
    center = dataPackObs.radioArray.getCenter()
    dpoint = np.sqrt((phase.ra.deg - patches.ra.deg)**2 + (phase.dec.deg - patches.dec.deg)**2)
    dant = np.sqrt(np.sum((antennas.cartesian.xyz.to(au.km).value.transpose() - center.cartesian.xyz.to(au.km).value.flatten())**2,axis=1))
    #choose
    sortPoint = np.argsort(dpoint)
    sortAnt = np.argsort(dant)
    antIdx = sortAnt[0:len(sortAnt):int(len(sortAnt)/10.)]
    dirIdx = sortPoint[-10:]
    dataPack = invertSingleTime(dataPackObs,6,"output/bootesInversion0-4",antIdx=antIdx,timeIdx=np.arange(1),dirIdx=dirIdx)
    #plotDataPack(dataPack)
    #dataPack.save("simulated.dill")


# In[ ]:



