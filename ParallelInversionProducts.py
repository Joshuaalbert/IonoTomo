
# coding: utf-8

# In[1]:

### contains tasks to be run in parallel with pp


def fetchPatch(patchFile,timeStart,timeEnd,radioArray):
    import numpy as np
    import astropy.units as au
    import astropy.time as at
    import astropy.coordinates as ac
    from ProgressBarClass import ProgressBar
    
    outAntennas = None
    outAntennaLabels = None
    outTimes = None
    outTimeStamps = None
    try:
        d = np.load(patchFile)
        print("Loading data file: {0}".format(patchFile))
    except:
        print("Failed loading data file: {0}".format(patchFile))
        return
    #internal data of each patch file (directions set by infoFile)
    antennas = d['antennas']
    times = d['times'][timeStart:timeEnd]#gps tai
    tecData = d['data'][timeStart:timeEnd,:]#times x antennas
    outTimes_ = []
    outTimeStamps_ = []
    outDtec_ = []
    numTimes = len(times)
    timeIdx = 0
    #progress = ProgressBar(numTimes, fmt=ProgressBar.FULL)
    while timeIdx < numTimes:
        time = at.Time(times[timeIdx],format='gps',scale='tai')
        #print("Processing time: {0}".format(time.isot))
        outTimes_.append(time)
        outTimeStamps_.append(time.isot)
        # get direction of patch at time wrt fixed frame
        outAntennas_ = []
        outAntennaLabels_ = []
        antIdx = 0#index in solution table
        numAnt = len(antennas)
        while antIdx < numAnt:
            ant = antennas[antIdx]
            labelIdx = radioArray.getAntennaIdx(ant)  
            if labelIdx is None:
                print("failed to find {}".format(ant))
            #ITRS WGS84
            stationLoc = radioArray.locs[labelIdx]
            outAntennaLabels_.append(ant)
            outAntennas_.append(stationLoc)
            outDtec_.append(tecData[timeIdx,antIdx])
            antIdx += 1
        #progress(timeIdx)
        timeIdx += 1
    #progress.done()
            
    if outTimes is None:
        timeArray = np.zeros(len(outTimes_))
        j = 0
        while j < len(outTimes_):
            timeArray[j] = outTimes_[j].gps
            j += 1
        outTimes = at.Time(timeArray,format='gps',scale='tai')
        outTimeStamps = np.array(outTimeStamps_)
    if outAntennas is None:
        antennasArray = np.zeros([len(outAntennas_),3])
        i = 0
        while i < len(outAntennas_):
            antennasArray[i,:] = outAntennas_[i].transform_to('itrs').cartesian.xyz.to(au.km).value.flatten()
            i += 1
        outAntennas = ac.SkyCoord(antennasArray[:,0]*au.km,antennasArray[:,1]*au.km,antennasArray[:,2]*au.km,
                                 frame = 'itrs')
        outAntennaLabels = np.array(outAntennaLabels_)
    return outAntennas, outAntennaLabels, outTimes, outTimeStamps, outDtec_
    
def castRay(origins, directions, neTCI, frequency, tmax, N, straightLineApprox):
    '''Calculates TEC for all given rays.
    ``origins`` is an array with coordinates in prefered frame
    ``diretions`` is an array with coordinates in prefered frame
    ``tmax`` is the length of rays to use.
    ``neTCI`` is the tri cubic interpolator
    return list of ray trajectories'''
    from FermatClass import Fermat
    #neTCI.clearCache()
    fermat = Fermat(neTCI=neTCI,frequency = frequency,type='z',straightLineApprox=straightLineApprox)
    Nr = origins.shape[0]
    rays = []
    r = 0
    while r < Nr:
        origin = origins[r,:]
        direction = directions[r,:]
        x,y,z,s = fermat.integrateRay(origin,direction,tmax,N=N)
        rays.append({'x':x,'y':y,'z':z,'s':s})
        r += 1
    return rays, neTCI.cache

#def forwardEquations(rays,TCI,mu,Kmu,rho,Krho,numTimes,numDirections):

def calculateTEC(rays, muTCI,K_e):
    '''Calculates TEC for all given rays in ``rays``.
    ``muTCI`` is the tri cubic interpolator
    ``K_e`` the log constant
    return ordered array of tec and updated cache of muTCI'''
    import numpy as np
    from scipy.integrate import simps
    #K_e = np.mean(neTCI.m)
    #mu = np.log(neTCI.m/K_e)
    #neTCI.m = mu
    #do all 
    #neTCI.clearCache()
    Nr = len(rays)
    Ns = len(rays[0]['s'])
    #muint = np.zeros([Nr,Ns])
    muint = np.zeros(Ns,dtype=np.double)
    tec = np.zeros(Nr)
    i = 0
    while i < Nr:
        ray = rays[i]
        j = 0
        while j < Ns:
            x,y,z = ray['x'][j],ray['y'][j],ray['z'][j]
            #muint[j] = neTCI.interp(x,y,z)
            muint[j] = muTCI.interp(x,y,z)
            j += 1
        tec[i] = simps(K_e*np.exp(muint),rays[i]['s'])/1e13
        i += 1
    #tec = simps(K_e*np.exp(muint),rays[0]['s'],axis = 1)/1e13
    return tec,muTCI.cache

def calculateModelingError(rays,muTCI,K_e,sigma,frequency):
    '''Calculates model error of TEC for all given rays.
    ``rays`` used to calculate along
    ``neTCI`` is the tri cubic interpolator
    ``sigma`` 
    ``frequency`` in Hz
    return ordered array of tec and updated cache'''
    import numpy as np
    from scipy.integrate import simps
    n_p = 1.240e-2 * frequency**2
    #K_e = np.mean(neTCI.m)
    #mu = np.log(neTCI.m/K_e)
    #neTCI.m = mu
    #do all 
    #neTCI.clearCache()
    Nr = len(rays)
    Ns = len(rays[0]['s'])
    #muint = np.zeros([Nr,Ns])
    muint = np.zeros(Ns,dtype=np.double)
    sigma_tec = np.zeros(Nr)
    i = 0
    while i < Nr:
        ray = rays[i]
        j = 0
        while j < Ns:
            x,y,z = ray['x'][j],ray['y'][j],ray['z'][j]
            #muint[j] = neTCI.interp(x,y,z)
            muint[j] = muTCI.interp(x,y,z)
            j += 1
        alphaUpper = (K_e*np.exp(muint)*(1. + sigma))/n_p
        sigma_tec[i] = (n_p/8.)*simps(alphaUpper**3/(1-alphaUpper)**(5./2.),rays[i]['s'])/1e13
        i += 1
    #tec = simps(K_e*np.exp(muint),rays[0]['s'],axis = 1)/1e13
    return sigma_tec,muTCI.cache

def calculateTEC_modelingError(rays, muTCI,K_e,sigma,frequency):
    '''Calculates TEC for all given rays. and modelling error
    ``length`` is the length of rays to use.
    ``muTCI`` is the tri cubic interpolator
    ``K_e`` log constant
    return ordered array of tec and updated cache'''
    import numpy as np
    from scipy.integrate import simps
    n_p = 1.240e-2 * frequency**2
    #K_e = np.mean(neTCI.m)
    #mu = np.log(neTCI.m/K_e)
    #neTCI.m = mu
    #do all 
    #neTCI.clearCache()
    Nr = len(rays)
    Ns = len(rays[0]['s'])
    #muint = np.zeros([Nr,Ns])
    muint = np.zeros(Ns,dtype=np.double)
    tec = np.zeros(Nr)
    sigma_tec = np.zeros(Nr)
    i = 0
    while i < Nr:
        ray = rays[i]
        j = 0
        while j < Ns:
            x,y,z = ray['x'][j],ray['y'][j],ray['z'][j]
            muint[j] = muTCI.interp(x,y,z)
            j += 1
        alphaUpper = (K_e*np.exp(muint)*(1. + sigma))/n_p
        sigma_tec[i] = (n_p/8.)*simps(alphaUpper**3/(1-alphaUpper)**(5./2.),rays[i]['s'])/1e13
        tec[i] = simps(K_e*np.exp(muint),rays[i]['s'])/1e13
        i += 1
    #tec = simps(K_e*np.exp(muint),rays[0]['s'],axis = 1)/1e13
    return tec,sigma_tec,muTCI.cache

def innovationPrimaryCalculation_exponential(rayPairs,muTCI,K_e,L_ne,sigma_ne_factor):
    '''Calculate the first part of S, i.e. 
    Int_R^ijk exp(m(x)) [ Int_R^nmp Cm(x,y) exp(x(y)) ]'''
    import numpy as np
    from scipy.integrate import simps
    #from time import clock
    fp = (7./3. - 4./3. - 1.)
    Ns = len(rayPairs[0][0]['s'])
    dy = np.zeros([Ns,Ns],dtype=np.double)
    dz = np.zeros([Ns,Ns],dtype=np.double)
    Cm_pair = np.zeros([Ns,Ns],dtype=np.double)
    outer = np.zeros(Ns,dtype=np.double)   
    outPairs = np.zeros(len(rayPairs),dtype=np.double)
    rayPairIdx = 0
    while rayPairIdx < len(rayPairs):
        ray1 = rayPairs[rayPairIdx][0]
        ray2 = rayPairs[rayPairIdx][1]
        np.subtract.outer(ray1['x'],ray2['x'],out=Cm_pair)
        np.subtract.outer(ray1['y'],ray2['y'],out=dy)
        np.subtract.outer(ray1['z'],ray2['z'],out=dz)
        #dx**2
        Cm_pair *= Cm_pair
        #dy**2
        dy *= dy
        #dz**2
        dz *= dz
        Cm_pair += dy
        Cm_pair += dz
        np.sqrt(Cm_pair,out=Cm_pair)
        Cm_pair /= -L_ne
        np.exp(Cm_pair,out=Cm_pair)
        Cm_pair *= sigma_ne_factor**2
        #transform to Cm = log(1+Cne)
        Cm_pair += 1.
        np.log(Cm_pair,out=Cm_pair)
        #Get the model at points
        
        
        j = 0           
        while j < Ns:
            x2,y2,z2 = ray2['x'][j],ray2['y'][j],ray2['z'][j]
            Cm_pair[:,j] *= np.exp(muTCI.interp(x2,y2,z2))
            j += 1
        outer[:] = simps(Cm_pair,ray2['s'],axis=1)
        i = 0
        while i < Ns:
            x1,y1,z1 = ray1['x'][i],ray1['y'][i],ray1['z'][i]
            outer[i] *= np.exp(muTCI.interp(x1,y1,z1))
            i += 1
        outPairs[rayPairIdx] = simps(outer,ray1['s'])
        if False:
            #import pylab as plt
            #Q = np.arange(101)
            #perc = []
            #for q in Q:
            #    perc.append(np.percentile(Cm_pair[Cm_pair>0].flatten(),q))
            #plt.plot(Q,perc)
            #plt.savefig('perc.pdf',format='pdf')
            #plt.show
            threshold = np.percentile(Cm_pair[Cm_pair>64*fp].flatten(),80)
            #print(threshold)
            #return
            mask = Cm_pair > threshold
            i = 0
            while i < Ns:
                x1,y1,z1 = ray1['x'][i],ray1['y'][i],ray1['z'][i]
                Cm_pair[i,:] *= np.exp(muTCI.interp(x1,y1,z1))
                j = 0           
                while j < Ns:
                    if mask[i,j]:
                        x2,y2,z2 = ray2['x'][j],ray2['y'][j],ray2['z'][j]
                        Cm_pair[i,j] *= np.exp(muTCI.interp(x2,y2,z2))
                    j += 1
                outer[i] = simps(Cm_pair[i,:],ray2['s'])
                i += 1
            outPairs[rayPairIdx] = simps(outer,ray1['s'])
        rayPairIdx += 1
    outPairs *= (K_e/1e13)**2
    return outPairs,muTCI.cache   

def innovationAdjointPrimaryCalculation_exponential(rays,muTCI,K_e,L_ne,sigma_ne_factor):
    '''Calculate the first part of Y, i.e. 
    Int_R^ijk Cm(x,y) exp(m(y)) ]'''
    import numpy as np
    from scipy.integrate import simps
    fp = (7./3. - 4./3. - 1.)
    X,Y,Z = muTCI.getModelCoordinates()
    Nm = len(X)
    Ns = len(rays[0]['s'])
    dy = np.zeros([Nm,Ns],dtype=np.double)
    dz = np.zeros([Nm,Ns],dtype=np.double)
    Cm_ray = np.zeros([Nm,Ns],dtype=np.double)  
    outCmGt_primary = np.zeros([Nm,len(rays)],dtype=np.double)
    rayIdx = 0
    while rayIdx < len(rays):
        ray = rays[rayIdx]
        np.subtract.outer(X,ray['x'],out=Cm_ray)
        np.subtract.outer(Y,ray['y'],out=dy)
        np.subtract.outer(Z,ray['z'],out=dz)
        #dx**2
        Cm_ray *= Cm_ray
        #dy**2
        dy *= dy
        #dz**2
        dz *= dz
        Cm_ray += dy
        Cm_ray += dz
        np.sqrt(Cm_ray,out=Cm_ray)
        Cm_ray /= -L_ne
        np.exp(Cm_ray,out=Cm_ray)
        Cm_ray *= sigma_ne_factor**2
        #transform to Cm = log(1+Cne)
        Cm_ray += 1.
        np.log(Cm_ray,out=Cm_ray)
        #Get the model at points
        #import pylab as plt
        #Q = np.arange(101)
        #perc = []
        #for q in Q:
        #    perc.append(np.percentile(Cm_ray[Cm_ray>64*fp].flatten(),q))
        #plt.plot(Q,perc)
        #plt.yscale('log')
        #plt.savefig('adjointPerc.pdf',format='pdf')
        #return
        #plt.show
        #threshold = np.percentile(Cm_ray[Cm_ray>64*fp].flatten(),80)
        #print(threshold)
        #return
        #mask = Cm_ray > threshold
        j = 0
        while j < Ns:
            x,y,z = ray['x'][j],ray['y'][j],ray['z'][j]
            Cm_ray[:,j] *= np.exp(muTCI.interp(x,y,z))
            j += 1
        outCmGt_primary[:,rayIdx] = simps(Cm_ray,ray['s'],axis=1)
        rayIdx += 1
    outCmGt_primary *= (K_e/1e13)
    return outCmGt_primary,muTCI.cache   
        
def primaryInversionSteps(dtec,rays,TCI,mu,Kmu,rho,Krho,muprior,rhoprior,sigma_ne,L_ne,sigma_rho,numTimes,numDirections,priorFlag=True):
    '''Performs forward integration of kernel, as well as derivative kernels. Time scales linearly with number of antennas.
    ``dtec`` - dict, datumIdx: dtec
    ``rays`` - dict, datumIdx: x,y,z,s arrays
    ``TCI`` - TriCubic object
    ``mu`` - current log(ne/K_mu) model
    ``Kmu`` - K_mu
    ``rho`` - current baseline log(tec0/K_rho/S)
    ``Krho`` - K_rho
    ``muprior`` - a priori log(ne/K_mu)
    ``rhoprior`` - a priori baseline log(tec0/K_rho/S)
    ``sigma_ne`` - expected deviation from mlogprior that mlog_true will be
    ``L_ne`` - coherence scale of ne in km
    ``sigma_rho - expected deviate from rhoprior that rho_true will be
    ``priorFlag`` - Wether or not to computer the G.(mp - m) term (not if m=mp)'''
    import numpy as np
    from scipy.integrate import simps
    #print("Serial primary thread")
    #forward equation
    #print('Forward equation...')
    #calculate data residuals dd = d - g
    #print('dd = d - g')
    #calculate G^i = (K*exp(mu)*delta(R^i), 1)
    #print('G kernels')
    #calculate int_R^i Cm(x,x').G^i(x')
    #print('int_R^i Cm(x,y).G^i(y)')
        
    TCI.m = mu
    TCI.clearCache()
    dtecModel = {}
    dd = {}
    G = {} #datumIdx: (Gmu,Grho)
    xmod,ymod,zmod = TCI.getModelCoordinates()
    CmGt = {}#each ray gives a column vector of int_R^i Cmu(x,x').Gmu^i(x'), Crho.Grho evaluated at all x of model
    keys = rays.keys()
    for datumIdx in keys:
        antIdx, dirIdx, timeIdx = reverseDatumIdx(datumIdx,numTimes,numDirections)
        ray = rays[datumIdx]
        Ns = len(ray['s'])
        Gmu = np.zeros(Ns) #K_mu * exp(mu(x)) along ray-ijk
        Cmu = np.zeros([np.size(xmod),Ns])
        i = 0
        while i < Ns:
            x,y,z = ray['x'][i],ray['y'][i],ray['z'][i]
            mu = TCI.interp(x,y,z)
            Gmu[i] += mu
            diff = np.sqrt((xmod - x)**2 + (ymod - y)**2 + (zmod - z)**2)
            Cmu[:,i] = np.log(1. + (sigma_ne/Kmu)**2 * np.exp(-diff/L_ne))
            i += 1
        Gmu = Kmu*np.exp(Gmu)/1e13#same as ne
        Grho = np.zeros(numDirections)
        Grho[dirIdx] = -Krho*np.exp(rho[dirIdx])/1e13
        #tec = simps(Gi,ray['s'])
        #dtecModel[datumIdx] = (tec - tec0)/1e13
        #dd[datumIdx] = dtec[datumIdx] - dtecModel[datumIdx]
        #G[datumIdx] = Gi
        #B = A*G[datumIdx]#should broadcast properly last indcies the same
        
        #CmGt[datumIdx] = simps(B,ray['s'],axis=1) + sigma_rho**2
        D = simps(np.vstack((Gmu + Grho[dirIdx],Cmu*Gmu)),ray['s'],axis=1)
        G[datumIdx] = [Gmu,Grho]
        dtecModel[datumIdx] = D[0]
        dd[datumIdx] = dtec[datumIdx] - dtecModel[datumIdx]
        CmGt[datumIdx] = [D[1:],sigma_rho**2 * Grho * (ray['s'][-1] - ray['s'][0])]#
        #batch all simps together
        
              
    if priorFlag:
        ##calculate G.(mp - m)
        TCI.m = muprior - mu
        drhopriorrho = rhoprior - rho
        TCI.clearCache()
        Gdmpm = {}
        for datumIdx in keys:
            antIdx, dirIdx, timeIdx = reverseDatumIdx(datumIdx,numTimes,numDirections)
            ray = rays[datumIdx]
            Ns = len(ray['s'])
            mupmu = np.zeros(Ns)
            i = 0
            while i < Ns:
                x,y,z = ray['x'][i],ray['y'][i],ray['z'][i]
                mupmu[i] += TCI.interp(x,y,z)#K*exp(mu)*delta(R^i)
                i += 1
            Gdmpm[datumIdx] = simps(G[datumIdx][0]*mupmu + G[datumIdx][1][dirIdx]*drhopriorrho[dirIdx],ray['s'])
        ##calculate the difference dd - Gdmpm
        #print('dd - D.(mp - m)')
        ddGdmpm = {}
        for datumIdx in keys:
            ddGdmpm[datumIdx] = dd[datumIdx] - Gdmpm[datumIdx]
    else:
        ddGdmpm = {}
        for datumIdx in keys:
            ddGdmpm[datumIdx] = dd[datumIdx]
        
    return G, CmGt, ddGdmpm, dd

def secondaryInversionSteps(rays, G, CmGt, TCI, sigma_rho, Cd,numTimes,numDirections):
    '''Compute S = Cd + G.Cm.G^t using parameters:
    ``rays`` - the dict {datumIdx:x,y,z,s arrays}
    ``G`` - the derivative along rays, a map {datumIdx: array of G^i(x) along ray^i} (product of primary inversion steps)
    ``CmGt`` - a map of int R^j Cm(x,x') G^j(x') evaluated at all model points (product of primary inversion steps)
    ``TCI`` - a tricubic interpolator
    ``sigma_rho`` - the deviation of rho (TEC baseline) because Cm only contains C_mu'''
    #G.Cm.G^t = int R^i G^i(x) int R^j Cmu(x,x') G_mu^j(x') + sigma_rho**2*G_rho
    import numpy as np
    from scipy.integrate import simps
    Nr = len(G)
    Ns = len(rays[rays.keys()[0]]['s'])
    GCmGt = np.zeros(Ns)#will be overwritten many times
    S = np.zeros([Nr,Nr],dtype=np.double)
    datumIdxj = 0
    while datumIdxj < Nr:#contain swapping this out to first loop (TCI operations are slowest I think)
        antIdxj, dirIdxj, timeIdxj = reverseDatumIdx(datumIdxj,numTimes,numDirections)
        TCI.m = CmGt[datumIdxj][0] #interpolate already done int_R^i Cmu(x,x').Gmu(x')
        TCI.clearCache()
        for datumIdxi in rays:
            if datumIdxj >= datumIdxi:#only do j>=i indicies, swap after
                antIdxi, dirIdxi, timeIdxi = reverseDatumIdx(datumIdxi,numTimes,numDirections)
                ray = rays[datumIdxi]
                #Ns = len(ray['s'])
                #GCmGt = np.zeros(Ns)#will be overwritten many times
                i = 0
                while i < Ns:
                    x,y,z = ray['x'][i],ray['y'][i],ray['z'][i]
                    if dirIdxi == dirIdxj:#correlation between rho directions
                        GCmGt[i] += G[datumIdxi][0][i] * TCI.interp(x,y,z) + G[datumIdxi][1][dirIdxi] * CmGt[datumIdxj][1][dirIdxj]
                    else:
                        GCmGt[i] += G[datumIdxi][0][i] * TCI.interp(x,y,z)
                    i += 1
                S[datumIdxi,datumIdxj] += simps(GCmGt,ray['s']) + Cd[datumIdxi,datumIdxj]
                if datumIdxi != datumIdxj:
                    S[datumIdxj,datumIdxi] = S[datumIdxi,datumIdxj]
        datumIdxj += 1
    return S    
    

