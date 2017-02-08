
# coding: utf-8

# In[3]:

### contains tasks to be run in parallel with pp

def getDatumIdx(antIdx,dirIdx,timeIdx,numDirections,numTimes):
    '''standarizes indexing'''
    idx = antIdx*numDirections*numTimes + dirIdx*numTimes + timeIdx
    return idx
    
def reverseDatumIdx(datumIdx,numTimes,numDirections):
    '''Reverse standardized indexing'''
    timeIdx = datumIdx % numTimes
    dirIdx = (datumIdx - timeIdx)/numTimes % numDirections
    antIdx = (datumIdx - timeIdx - dirIdx*numTimes)/numTimes/numDirections
    return antIdx, dirIdx, timeIdx
    
def forwardEquations(rays,TCI,mu,Kmu,rho,Krho,numTimes,numDirections):
    '''Perform the forward equation of this model for each ray in rays.
    ``rays`` is a dict with datumIdx keys
    ``TCI`` is the tricubic interpolator.
    return a dictionary of dtec with datumIdx keys
    if ``rho`` is None then take first antenna tec for each direction for rho'''
    import numpy as np
    from scipy.integrate import simps
    TCI.m = mu
    TCI.clearCache()
    if rho is None:
        rho = np.zeros(numDirections)
    dtec = {}
    setList = []
    for datumIdx in rays.keys():
        antIdx, dirIdx, timeIdx = reverseDatumIdx(datumIdx,numTimes,numDirections)
        ray = rays[datumIdx]
        Ns = len(ray['s'])
        neint = np.zeros(Ns)
        i = 0
        while i < Ns:
            x,y,z = ray['x'][i],ray['y'][i],ray['z'][i]
            neint[i] += TCI.interp(x,y,z)
            i += 1
        tec = simps(Kmu*np.exp(neint),ray['s'])
        if Krho is None:
            Krho = tec/(ray['s'][-1] - ray['s'][0])
        if dirIdx not in setList:
            rho[dirIdx] = np.log(tec / (Krho * (ray['s'][-1] - ray['s'][0])))
            setList.append(dirIdx)
        dtec[datumIdx] = ( tec - Krho * np.exp(rho[dirIdx]) * (ray['s'][-1] - ray['s'][0]) ) / 1e13
    return dtec,rho,Krho

def primaryInversionStepsSerial(dtec,rays,TCI,mu,Kmu,rho,Krho,muprior,rhoprior,sigma_ne,L_ne,sigma_rho,priorFlag=True):
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
    print("Serial primary thread")
    #forward equation
    print('Forward equation...')
    TCI.m = mu
    TCI.clearCache()
    dtecModel = {}
    keys = rays.keys()
    for datumIdx in rays.keys():
        ray = rays[datumIdx]
        Ns = len(ray['s'])
        neint = np.zeros(Ns)
        i = 0
        while i < Ns:
            x,y,z = ray['x'][i],ray['y'][i],ray['z'][i]
            ne = TCI.interp(x,y,z)
            neint[i] += ne
            i += 1
        tec = simps(neint,ray['s'])
        if tec0 is None:
            tec0 = tec
        dtecModel[datumIdx] = (tec - tec0)/1e13
    #calculate data residuals dd = d - g
    print('dd = d - g')
    dd = {}
    for datumIdx in keys:
        dd[datumIdx] = dtec[datumIdx] - dtecModel[datumIdx]
    #calculate G^i = (K*exp(mu)*delta(R^i), 1)
    print('G kernels')
    G = {}
    for datumIdx in keys:
        ray = rays[datumIdx]
        Ns = len(ray['s'])
        Gi = np.zeros(Ns)
        i = 0
        while i < Ns:
            x,y,z = ray['x'][i],ray['y'][i],ray['z'][i]
            ne = TCI.interp(x,y,z)#K*exp(mu)*delta(R^i)
            Gi[i] += ne
            i += 1
        G[datumIdx] = Gi
    if priorFlag:
        ##calculate G.(mp - m)
        print("G.(mp - m)")
        TCI.m = mlogprior - mlog
        TCI.clearCache()
        Gdmpm = {}
        for datumIdx in keys:
            ray = rays[datumIdx]
            Ns = len(ray['s'])
            mpm = np.zeros(Ns)
            i = 0
            while i < Ns:
                x,y,z = ray['x'][i],ray['y'][i],ray['z'][i]
                mpm[i] += TCI.interp(x,y,z)#K*exp(mu)*delta(R^i)
                i += 1
            Gdmpm[datumIdx] = simps(G[datumIdx]*mpm,ray['s'])
            #G_rho(rho_p - rho) term
            Gdmpm[datumIdx] += (rhoprior - rho)
        ##calculate the difference dd - Gdmpm
        print('dd - D.(mp - m)')
        ddGdmpm = {}
        for datumIdx in keys:
            ddGdmpm[datumIdx] = dd[datumIdx] - Gdmpm[datumIdx]
    else:
        ddGdmpm = {}
        for datumIdx in keys:
            ddGdmpm[datumIdx] = dd[datumIdx]
        
    #calculate int_R^i Cm(x,x').G^i(x')
    print('int_R^i Cm(x,y).G^i(y)')
    xmod,ymod,zmod = TCI.getModelCoordinates()
    CmGt = {}#each ray gives a column vector of int_R^i Cm(x,x').G^i(x') evaluated at all x of model
    for datumIdx in keys:
        #print(datumIdx)
        ray = rays[datumIdx]
        
        Ns = len(ray['s'])

        A = np.zeros([np.size(xmod),Ns])
        i = 0
        while i < Ns:
            x,y,z = ray['x'][i],ray['y'][i],ray['z'][i]
            diff = np.sqrt((xmod - x)**2 + (ymod - y)**2 + (zmod - z)**2)
            A[:,i] = np.log(1. + (sigma_ne/K)**2 * np.exp(-diff/L_ne))
            i += 1
        B = A*G[datumIdx]#should broadcast properly last indcies the same
        CmGt[datumIdx] = simps(B,ray['s'],axis=1)# + sigma_rho**2
    return G, CmGt, ddGdmpm

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
    CmuGmut = {}#each ray gives a column vector of int_R^i Cm(x,x').G^i(x') evaluated at all x of model
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
        Grho = -Krho*np.exp(rho[dirIdx])/1e13
        #tec = simps(Gi,ray['s'])
        #dtecModel[datumIdx] = (tec - tec0)/1e13
        #dd[datumIdx] = dtec[datumIdx] - dtecModel[datumIdx]
        #G[datumIdx] = Gi
        #B = A*G[datumIdx]#should broadcast properly last indcies the same
        
        #CmGt[datumIdx] = simps(B,ray['s'],axis=1) + sigma_rho**2
        D = simps(np.vstack((Gmu + Grho,Cmu*Gmu)),ray['s'],axis=1)
        G[datumIdx] = [Gmu,Grho]
        dtecModel[datumIdx] = D[0]
        dd[datumIdx] = dtec[datumIdx] - dtecModel[datumIdx]
        CmuGmut[datumIdx] = D[1:]# + sigma_rho**2
        #batch all simps together
        
              
    if priorFlag:
        ##calculate G.(mp - m)
        #print("G.(mp - m)")
        TCI.m = muprior - mu
        drhopriorrho = rhoprior - rho
        TCI.clearCache()
        Gdmpm = {}
        for datumIdx in keys:
            antIdx, dirIdx, timeIdx = reverseDatumIdx(datumIdx,numTimes,numDirections)
            ray = rays[datumIdx]
            Ns = len(ray['s'])
            mpm = np.zeros(Ns)
            i = 0
            while i < Ns:
                x,y,z = ray['x'][i],ray['y'][i],ray['z'][i]
                mpm[i] += TCI.interp(x,y,z)#K*exp(mu)*delta(R^i)
                i += 1
            Gdmpm[datumIdx] = simps(G[datumIdx][0]*mpm + G[datumIdx][1]*drhopriorrho[dirIdx],ray['s'])
        ##calculate the difference dd - Gdmpm
        #print('dd - D.(mp - m)')
        ddGdmpm = {}
        for datumIdx in keys:
            ddGdmpm[datumIdx] = dd[datumIdx] - Gdmpm[datumIdx]
    else:
        ddGdmpm = {}
        for datumIdx in keys:
            ddGdmpm[datumIdx] = dd[datumIdx]
        
    return G, CmuGt, ddGdmpm
    
def secondaryInversionStepsTCIInside(rays, G, CmGt, TCI, sigma_rho, Cd):
    '''Compute S = Cd + G.Cm.G^t using parameters:
    ``rays`` - the dict {datumIdx:x,y,z,s arrays}
    ``G`` - the derivative along rays, a map {datumIdx: array of G^i(x) along ray^i} (product of primary inversion steps)
    ``CmGt`` - a map of int R^j Cm(x,x') G^j(x') evaluated at all model points (product of primary inversion steps)
    ``TCI`` - a tricubic interpolator
    ``sigma_rho`` - the deviation of rho (TEC baseline) because Cm only contains C_mu'''
    #G.Cm.G^t = int R^i G^i(x) int R^j Cmu(x,x') G^j(x') + sigma_rho**2
        
    Nr = len(G)
    S = np.zeros([Nr,Nr],dtype=np.double)
    for datumIdxi in rays:
        ray = rays[datumIdxi]
        Ns = len(ray['s'])
        GCmGt = np.zeros(Ns)#will be overwritten many times
        datumIdxj = datumIdxi
        while datumIdxj < Nr:#contain swapping this out to first loop (TCI operations are slowest I think)
            TCI.m = CmGt[datumIdxj]
            TCI.clearCache()
            i = 0
            while i < Ns:
                x,y,z = ray['x'][i],ray['y'][i],ray['z'][i]
                GCmGt[i] += G[datumIdxi][i] * TCI.interp(x,y,z)
                i += 1
            S[datumIdxi,datumIdxj] += simps(GCmGt,ray['s']) + sigma_rho**2 + Cd[datumIdxi,datumIdxj]
            if datumIdxi != datumIdxj:
                S[datumIdxj,datumIdxi] = S[datumIdxi,datumIdxj]
            datumIdxj += 1
    return S

def secondaryInversionSteps(rays, G, CmGt, TCI, sigma_rho, Cd):
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
        TCI.m = CmGt[datumIdxj]
        TCI.clearCache()
        for datumIdxi in rays:
            if datumIdxj >= datumIdxi:
                ray = rays[datumIdxi]
                #Ns = len(ray['s'])
                #GCmGt = np.zeros(Ns)#will be overwritten many times
                i = 0
                while i < Ns:
                    x,y,z = ray['x'][i],ray['y'][i],ray['z'][i]
                    GCmGt[i] += G[datumIdxi][i] * TCI.interp(x,y,z)
                    i += 1
                S[datumIdxi,datumIdxj] += simps(GCmGt,ray['s']) + sigma_rho**2/1e26 + Cd[datumIdxi,datumIdxj]
                if datumIdxi != datumIdxj:
                    S[datumIdxj,datumIdxi] = S[datumIdxi,datumIdxj]
        datumIdxj += 1
    return S

            
    
    

