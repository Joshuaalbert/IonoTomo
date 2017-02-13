
# coding: utf-8

# In[1]:

from time import time as tictoc
import numpy as np
from scipy.integrate import odeint
from scipy.integrate import simps
import math
import pp
import astropy.coordinates as ac
import astropy.units as au
import astropy.time as at

#User defined
from IRI import *
from TricubicInterpolation import TriCubic
from RadioArray import RadioArray
from ENUFrame import ENU
try:
    from MayaviPlotting import *
except:
    print("Unable to import mayavi")

class Fermat(object):
    def __init__(self,neTCI=None,frequency = 120e6,type='s'):
        self.type = type
        self.frequency = frequency#Hz
        if neTCI is not None:
            self.ne2n(neTCI)  
            return
        
    def loadFunc(self,file):
        '''Load the model given in `file`'''
        data = np.load(file)
        if 'ne' in data.keys():
            ne = data['ne']
            xvec = data['xvec']
            yvec = data['yvec']
            zvec = data['zvec']
            self.ne2n(TriCubic(xvec,yvec,zvec,ne))
            return
        if 'n' in data.keys():
            ne = data['n']
            xvec = data['xvec']
            yvec = data['yvec']
            zvec = data['zvec']
            self.n2ne(TriCubic(xvec,yvec,zvec,n))
            return
    
    def saveFunc(self,file):
        np.savez(file,xvec=self.nTCI.xvec,yvec=self.nTCI.yvec,zvec=self.nTCI.zvec,n=self.nTCI.m,ne=self.neTCI.m)
            
    def ne2n(self,neTCI):
        '''Analytically turn electron density to refractive index. Assume ne in m^-3'''
        self.neTCI = neTCI
        #copy object
        self.nTCI = neTCI.copy(default=1.)
        #inplace change to refractive index
        self.nTCI.m *= -8.980**2/self.frequency**2
        self.nTCI.m += 1.
        self.nTCI.m = np.sqrt(self.nTCI.m)
        #wp = 5.63e4*np.sqrt(ne/1e6)/2pi#Hz^2 m^3 lightman p 226
        return self.nTCI
    
    def n2ne(self,nTCI):
        """Get electron density in m^-3 from refractive index"""
        self.nTCI = nTCI
        #convert to 
        self.neTCI = nTCI.copy()
        self.neTCI.m *= -self.neTCI.m
        self.neTCI.m += 1.
        self.neTCI.m *= self.frequency**2/8.980**2
        #wp = 5.63e4*np.sqrt(ne/1e6)/2pi#Hz^2 m^3 lightman p 226
        return self.neTCI
    
    def eulerODE(self,y,t,*args):
        '''return pxdot,pydot,pzdot,xdot,ydot,zdot,sdot'''
        #print(y)
        px,py,pz,x,y,z,s = y
        #n,nx,ny,nz,nxy,nxz,nyz,nxyz = self.nTCI.interp(x,y,z,doDiff=True)
        #ne,nex,ney,nez,nexy,nexz,neyz,nexyz = self.neTCI.interp(x,y,z,doDiff=True)
        #A = - 8.98**2/self.frequency**2
        #n = math.sqrt(1. + A*ne)
        #ndot = A/(2.*n)
        #nx = ndot * nex
        #ny = ndot * ney
        #nz = ndot * nez
        
        #print(n)
        n,nx,ny,nz = 1.,0,0,0
        #if (n>1):
        #    print(x,y,z,n)
        if self.type == 'z':
            sdot = n / pz
            pxdot = nx*n/pz
            pydot = ny*n/pz
            pzdot = nz*n/pz

            xdot = px / pz
            ydot = py / pz
            zdot = 1.
        
        if self.type == 's':
            sdot = 1.
            pxdot = nx
            pydot = ny
            pzdot = nz

            xdot = px / n
            ydot = py / n
            zdot = pz / n
        
        return [pxdot,pydot,pzdot,xdot,ydot,zdot,sdot]
    
    def jacODE(self,y,t,*args):
        '''return d ydot / d y, with derivatives down column for speed'''
        px,py,pz,x,y,z,s = y
        #n,nx,ny,nz,nxy,nxz,nyz,nxyz = self.nTCI.interp(x,y,z,doDiff=True)
        nxx,nyy,nzz = 0.,0.,0.
        n,nx,ny,nz,nxy,nxz,nyz = 1.,0,0,0,0,0,0
        
        #ne,nex,ney,nez,nexy,nexz,neyz,nexyz = self.neTCI.interp(x,y,z,doDiff=True)
        #A = - 8.98**2/self.frequency**2
        #n = math.sqrt(1. + A*ne)
        #ndot = A/(2.*n)
        #nx = ndot * nex
        #ny = ndot * ney
        #nz = ndot * nez
        
        #ndotdot = -(A * ndot)/(2. * n**2)
        
        #nxy = ndotdot * nex*ney + ndot * nexy
        #nxz = ndotdot * nex * nez + ndot * nexz
        #nyz = ndotdot * ney * nez + ndot * neyz
        
        #if (n>1):
        #    print(x,y,z,n)
            
        if self.type == 'z':
            x0 = n
            x1 = nx
            x2 = pz**(-2)
            x3 = x0*x2
            x4 = 1./pz
            x5 = ny
            x6 = x4*(x0*nxy + x1*x5)
            x7 = nz
            x8 = x4*(x0*nxz + x1*x7)
            x9 = x4*(x0*nyz + x5*x7)
            jac = np.array([[ 0,  0, -x1*x3, x4*(x0*nxx + x1**2),x6, x8, 0.],
                            [ 0,  0, -x3*x5,x6, x4*(x0*nyy + x5**2), x9, 0.],
                            [ 0,  0, -x3*x7,x8, x9, x4*(x0*nzz + x7**2), 0.],
                            [x4,  0, -px*x2, 0, 0,  0, 0.],
                            [ 0, x4, -py*x2, 0, 0, 0, 0.],
                            [ 0,  0, 0, 0, 0, 0, 0.],
                            [ 0,  0,-x3,x1*x4, x4*x5, x4*x7, 0.]])
        
        if self.type == 's':
            x0 = n
            x1 = nxy
            x2 = nxz
            x3 = nyz
            x4 = 1./x0
            x5 = nx
            x6 = x0**(-2)
            x7 = px*x6
            x8 = ny
            x9 = nz
            x10 = py*x6
            x11 = pz*x6
            jac = np.array([[ 0,  0,  0, nxx, x1, x2, 0.],
                            [ 0,  0,  0, x1, nyy, x3, 0.],
                            [ 0,  0,  0, x2, x3, nzz, 0.],
                            [x4,  0,  0, -x5*x7, -x7*x8, -x7*x9, 0.],
                            [ 0, x4,  0, -x10*x5, -x10*x8, -x10*x9, 0.],
                            [ 0,  0, x4, -x11*x5, -x11*x8, -x11*x9, 0.],
                            [ 0,  0,  0, 0, 0, 0, 0.]])
        return jac
        
    def integrateRay(self,X0,direction,tmax,time = 0,N=100):
        '''Integrate rays from x0 in initial direction'''
        direction /= np.linalg.norm(direction)
        x0,y0,z0 = X0
        xdot0,ydot0,zdot0 = direction
        sdot = np.sqrt(xdot0**2 + ydot0**2 + zdot0**2)
        px0 = xdot0/sdot
        py0 = ydot0/sdot
        pz0 = zdot0/sdot
        init = [px0,py0,pz0,x0,y0,z0,0]
        if self.type == 'z':
            tarray = np.linspace(z0,tmax,N)
        if self.type == 's':
            tarray = np.linspace(0,tmax,N)
        #print("Integrating at {0} from {1} in direction {2} until {3}".format(time,X0,direction,tmax))
        #print(init)
        #print("Integrating from {0} in direction {1} until {2}".format(x0,directions,tmax))
        Y,info =  odeint(self.eulerODE, init, tarray, args=(time,),Dfun = self.jacODE, col_deriv = True, full_output=1)
        #print(info['hu'].shape,np.sum(info['hu']),info['hu'])
        #print(Y)
        x = Y[:,3]
        y = Y[:,4]
        z = Y[:,5]
        s = Y[:,6]
        return x,y,z,s   


def createPrioriModel(iri = None,L_ne=15.):
    if iri is None:
        iri = IriModel()
    xmin = -200.
    xmax = 200.
    ymin = -200.
    ymax = 200.
    zmin = -10.
    zmax = 3000.
    
    eastVec = np.linspace(xmin,xmax,int(np.ceil((xmax-xmin)/L_ne)))
    northVec = np.linspace(ymin,ymax,int(np.ceil((ymax-ymin)/L_ne)))
    upVec = np.linspace(zmin,zmax,int(np.ceil((zmax-zmin)/L_ne)))

    E,N,U = np.meshgrid(eastVec,northVec,upVec,indexing='ij')
    #get the points in ITRS frame
    points = ac.SkyCoord(E.flatten()*au.km,N.flatten()*au.km,U.flatten()*au.km,frame=iri.enu).transform_to('itrs').cartesian.xyz.to(au.km).value
    X = points[0,:].reshape(E.shape)
    Y = points[1,:].reshape(N.shape)
    Z = points[2,:].reshape(U.shape)

    ne = iri.evaluate(X,Y,Z)
    print("created an a priori cube of shape: {0}".format(ne.shape))
    return eastVec,northVec,upVec,ne

def perturbModel(eastVec,northVec,upVec,ne,loc,width,amp):
    nePert = ne.copy()
    E,N,U = np.meshgrid(eastVec,northVec,upVec,indexing='ij')
    for l,w,a in zip(loc,width,amp):
        print("Adding amp:{0:1.2e} at: {1} scale:{2:0.2f}".format(a,l,w))
        nePert += a*np.exp(-((E-l[0])**2 + (N-l[1])**2 + (U-l[2])**2)/w**2)
    return nePert

    
def plot_dtec(Nant,directions,dtec,title='',subAnt=None,labels=None):
    def getDatumIdx(antIdx,dirIdx,timeIdx,numDirections,numTimes):
        '''standarizes indexing'''
        idx = antIdx*numDirections*numTimes + dirIdx*numTimes + timeIdx
        return idx
    vmin = np.min(dtec)
    vmax = np.max(dtec)
    #data -= np.min(dtec)
    #data /= np.max(dtec)
    Nperaxis = int(np.ceil(np.sqrt(Nant)))
    import pylab as plt
    cm = plt.cm.get_cmap('RdYlBu')
    f = plt.figure(figsize=(22,17))
    #f,ax = plt.subplots(int(np.ceil(np.sqrt(numAntennas))),int(np.ceil(np.sqrt(numAntennas))))
    for antIdx in range(Nant):
        ax = plt.subplot(Nperaxis,Nperaxis,antIdx+1)
        if labels is not None:

            ax.text(0.05, 0.95,"{}".format(labels[antIdx]),transform=ax.transAxes,fontsize=12,weight='bold')
        else:
            ax.text(0.05, 0.95,"Antenna {}".format(antIdx),transform=ax.transAxes,fontsize=12,weight='bold')
        for dirIdx in range(len(directions)):
            datumIdx = getDatumIdx(antIdx,dirIdx,0,len(directions),1)
            if subAnt is not None:
                datumIdx0 = getDatumIdx(subAnt,dirIdx,0,len(directions),1)
                sc=ax.scatter(directions[dirIdx,0],directions[dirIdx,1],c=dtec[datumIdx]-dtec[datumIdx0],s=20**2,vmin=vmin,vmax=vmax,cmap=cm)
            else:
                sc=ax.scatter(directions[dirIdx,0],directions[dirIdx,1],c=dtec[datumIdx],s=20**2,vmin=vmin,vmax=vmax,cmap=cm)
        plt.colorbar(sc)
    if title is not "":
        f.savefig("figs/dtec/{}.png".format(title),format='png')
    #plt.show()

def SimulatedDataInversion(numThreads = 1,noise=None,eta=1.):
    '''Test the full system.'''
    
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
    
    def datumDicts2array(datumDicts):
        '''Given a tupel of dicts where each dict is of datumIdx:value
        convert into single array with index giving order'''
        N = 0
        for datumDict in datumDicts:
            N += len(datumDict)
        array = np.zeros(N,dtype=np.double)
        for datumDict in datumDicts:
            for datumIdx in datumDict.keys():#ordering set by datumIdx function 1-to-1
                array[datumIdx] = datumDict[datumIdx]
        return array

    raylength = 2000.
    print("Using lofar array")
    radioArray = RadioArray(arrayFile='arrays/lofar.hba.antenna.cfg')
    timestamp = '2017-02-7T15:37:00.000'
    timeIdx = 0#one time stamp for now
    numTimes = 1
    time = at.Time(timestamp,format='isot',scale='tai')
    enu = ENU(obstime=time,location=radioArray.getCenter().earth_location)
    phase = ac.SkyCoord(east=0,north=0,up=1,frame=enu).transform_to(ac.ITRS(obstime=time)).transform_to('icrs')#straight up for now
    dec = phase.dec.rad
    ra = phase.ra.rad
    print("Simulating observation on {0}: {1}".format(time.isot,phase))
    
    stations = radioArray.locs.transform_to(enu).cartesian.xyz.to(au.km).value.transpose()
    stations = stations[46:53,:]
    labels = radioArray.labels[46:53]
    Nant = stations.shape[0]
    print("Using {0} stations".format(Nant))
    #print(stations,labels)
    #stations = np.random.multivariate_normal([0,0,0],[[20**2,0,0],[0,20**2,0],[0,0,0.01**2]],Nant)
    #stations = np.array([[0,0,0],[20,0,0]])
    
    Ndir = 10
    fov = radioArray.getFov()#radians
    print("Creating {0} directions in FOV of {1}".format(Ndir,fov))
    directions = np.random.multivariate_normal([ra,dec],[[(fov/2.)**2,0],[0,(fov/2.)**2]],Ndir)
    #print(directions)
    
    directions = ac.SkyCoord(directions[:,0]*au.radian,directions[:,1]*au.radian,frame='icrs').transform_to(enu).cartesian.xyz.value.transpose()
    
    #print(directions)
    
    print("Setting up tri cubic interpolator")
    L_ne = 15.
    
    # The priori ionosphere 
    iri = IriModel()
    print("Creating priori model")
    eastVec,northVec,upVec,nePriori = createPrioriModel(iri,L_ne)
    print("Creating perturbed model")
    nePert = perturbModel(eastVec,northVec,upVec,nePriori,([0,0,200.],[20,20,450.],[-100,-50,600]),(40.,40,50),(1e10,1e10,1e10))
    print("Creating TCI object")
    neTCI = TriCubic(eastVec,northVec,upVec,nePert)
    neTCIModel = TriCubic(eastVec,northVec,upVec,nePriori)
    TCI = TriCubic(eastVec,northVec,upVec,np.zeros_like(nePert))
    
    print("Creating fermat object - based on a priori (second order corrections require iterating this)")
    f =  Fermat(neTCI = neTCIModel,type = 's')
    
    print("Integrating rays with fermats principle")
    t1 = tictoc()
    rays = {}
    for antIdx in range(Nant):
        for dirIdx in range(Ndir):
            datumIdx = getDatumIdx(antIdx,dirIdx,timeIdx,Ndir,numTimes)
            #print(antIdx,dirIdx,timeIdx,datumIdx)
            origin = stations[antIdx,:]#ENU frame, later use UVW frame
            direction = directions[dirIdx,:]
            x,y,z,s = f.integrateRay(origin,direction,raylength,time=0.)
            rays[datumIdx] = {'x':x,'y':y,'z':z,'s':s}   
    Nd = len(rays)
    print("Time (total/per ray): {0:0.2f} / {1:0.2e} s".format(tictoc()-t1,(tictoc()-t1)/Nd))
    
    print("Setting up ray chunks for {0} threads".format(numThreads))
    #split up rays
    raypack = {i:{} for i in range(numThreads)}
    c = 0
    for datumIdx in rays.keys():
        raypack[c%numThreads][datumIdx] = rays[datumIdx]
        c += 1
     
    def ppForwardEquation(rays,TCI,mu,Kmu,rho,Krho,numTimes,numDirections):
        dtec, rho, Krho = ParallelInversionProducts.forwardEquations(rays,TCI,mu,Kmu,rho,Krho,numTimes,numDirections)
        return dtec, rho, Krho
    def ppPrimaryInversionSteps(dtec,rays,TCI,mu,Kmu,rho,Krho,muprior,rhoprior,sigma_ne,L_ne,sigma_rho,numTimes,numDirections,priorFlag=True):
        G, CmGt, ddGdmpm, dd = ParallelInversionProducts.primaryInversionSteps(dtec,rays,TCI,mu,Kmu,rho,Krho,muprior,rhoprior,sigma_ne,L_ne,sigma_rho,numTimes,numDirections,priorFlag=True)
        return G, CmGt, ddGdmpm, dd
    def ppSecondaryInversionSteps(rays, G, CmGt, TCI, sigma_rho, Cd,numTimes,numDirections):
        S = ParallelInversionProducts.secondaryInversionSteps(rays, G, CmGt, TCI, sigma_rho, Cd,numTimes,numDirections)
        return S
        
    jobs = {}
    job_server = pp.Server(numThreads, ppservers=())
    print("Creating dTec simulated data")
    job = job_server.submit(ppForwardEquation,
                   args=(rays,TCI,np.log(neTCI.m/np.mean(neTCI.m)),np.mean(neTCI.m),None,None,numTimes,Ndir),
                   depfuncs=(),
                   modules=('ParallelInversionProducts',))
    jobs['dtecSim'] = job
    
    job = job_server.submit(ppForwardEquation,
                   args=(rays,TCI,np.log(neTCIModel.m/np.mean(neTCIModel.m)),np.mean(neTCIModel.m),None,None,numTimes,Ndir),
                   depfuncs=(),
                   modules=('ParallelInversionProducts',))
    jobs['dtecModel'] = job
        
    dtecSim,rhoSim0, KrhoSim0 = jobs['dtecSim']()
    dobs = datumDicts2array((dtecSim,))
    #print("dobs: {0}".format(dobs))
    if noise is not None:
        print("Adding {0:0.2f}-sigma noise to simulated dtec".format(noise))
        dtecStd = np.std(dobs)
        dobs += np.random.normal(loc=0,scale=dtecStd*noise,size=np.size(dobs))
    #print("dobs: {0}".format(dobs))
    dtecModel,rhoModel0,KrhoModel0 = jobs['dtecModel']()
    g = datumDicts2array((dtecModel,))
    #print("g: {0}".format(g))
    job_server.print_stats()
    job_server.destroy()
    
    subAnt = None
    plot_dtec(Nant,directions,dobs,title='sim_dtec',subAnt=subAnt,labels=labels)
    plot_dtec(Nant,directions,g,title='model_dtec',subAnt=subAnt,labels=labels)
    plot_dtec(Nant,directions,dobs-g,title='sim-mod_dtec',subAnt=subAnt,labels=labels)
    
    print("Setting up inversion with parameters:")
    print("Number of rays: {0}".format(Nd))
    print("Forward equation: g(m) = int_R^i (K_mu * EXP[mu(x)] - K_rho * EXP[rho])/TECU ds")
    
    #gaussian process assumption, d = g + G.dm -> Cd = Gt.Cm.G (not sure)
    Cd = np.eye(Nd)*np.std(dobs)
    print("<Diag(Cd)> = {0:0.2e}".format(np.mean(np.diag(Cd))))
    
    print("a priori model is IRI")
    print("Define: mu(x) = LOG[ne(x) / K_mu]")
    Kmu = np.mean(neTCIModel.m)
    mu = np.log(neTCIModel.m/Kmu)
    muPrior = mu.copy()
    print("K_mu = {0:0.2e}".format(Kmu))
    
    #spatial-ergodic assumption
    sigma_ne = np.std(neTCIModel.m)
    print("Coherence scale: L_ne = {0:0.2e}".format(L_ne))
    print("C_ne = ({0:0.2e})**2 EXP[-|x1 - x2| / {1:0.1f}]".format(sigma_ne,L_ne))
    print("Define: rho = LOG[TEC_0 / K_rho / S]")
    Krho = KrhoModel0
    rho = rhoModel0
    rhoPrior = rho.copy()
    sigma_TEC = np.std(g*1e13)
    sigma_rho = np.sqrt(np.log(1+(sigma_TEC/Krho/raylength)**2))
    print("K_rho = {0:0.2e}".format(Krho))
    print("a priori rho (reference TEC): {0}".format(rho))
    print("sigma_rho = {0:0.2e}".format(sigma_rho))
    
    #TCI.m = np.log(neTCI.m/Kmu) - np.log(neTCIModel.m/Kmu)
    #TCI.clearCache()
    #plotWavefront(TCI,rays,save=False,animate=False)
    ddArray = dobs - g
    #inversion steps
    iter = 0
    residuals = np.inf
    modelFile = "results/model-{}.npz".format(0)
    np.savez(modelFile,mu=mu,rho=rho,Kmu=Kmu,Krho=Krho)
    print("Storing model")
    while residuals > 1e-10:
        print("Performing iteration: {0}".format(iter))
        likelihood = np.exp(-ddArray.dot(np.linalg.pinv(Cd)).dot(ddArray)/2.)
        print("Likelihood  = {0}".format(postLikelihood ))
        print("Performing primary inversion steps on {0}".format(numThreads))
        job_server = pp.Server(numThreads, ppservers=())
        
        for i in range(numThreads):
            job = job_server.submit(ppPrimaryInversionSteps,
                       args=(dtecSim,raypack[i],TCI,mu,Kmu,rho,Krho,muPrior,rhoPrior,sigma_ne,L_ne,sigma_rho,numTimes,Ndir,True),
                       depfuncs=(),
                       modules=('ParallelInversionProducts',))
            jobs['ppPrimaryInversionSteps_{0}'.format(i)] = job
        G,CmGt,ddGdmpm, dd = {},{},{},{}
        for i in range(numThreads):
            G_, CmGt_, ddGdmpm_, dd_ = jobs['ppPrimaryInversionSteps_{0}'.format(i)]()
            #print(G_, CmGt_, ddGdmpm_)
            G.update(G_)
            CmGt.update(CmGt_)
            ddGdmpm.update(ddGdmpm_)
            dd.update(dd_)
        job_server.print_stats()
        job_server.destroy()
        print("Performing secondary inversion steps")
        job_server = pp.Server(numThreads, ppservers=())
        for i in range(numThreads):
            job = job_server.submit(ppSecondaryInversionSteps,
                       args=(raypack[i], G, CmGt, TCI, sigma_rho, Cd*eta,numTimes,Ndir),
                       depfuncs=(),
                       modules=('ParallelInversionProducts',))
            jobs['ppSecondaryInversionSteps_{0}'.format(i)] = job
        S = np.zeros([Nd,Nd],dtype=np.double)
        for i in range(numThreads):
            S_ = jobs['ppSecondaryInversionSteps_{0}'.format(i)]()
            S += S_
        print("Inverting S")
        T = np.linalg.pinv(S)
        if False:
            import pylab as plt
            ax = plt.subplot(121)
            p1 = ax.imshow(S)
            plt.colorbar(p1)
            ax = plt.subplot(122)
            p2 = ax.imshow(T)
            plt.colorbar(p2)
            print("S:",S)
            print("T:",T)
        #plt.show()
        job_server.print_stats()
        job_server.destroy()
        # dm = (mp-m) + CmGt.T.ddGdmpm
        ddGdmpmArray = datumDicts2array([ddGdmpm])
        TddGdmpmArray = T.dot(ddGdmpmArray)
        CmGtArray = np.zeros([np.size(mu)+np.size(rho),Nd])
        for i in range(Nd):
            CmGtArray[:np.size(mu),i] = CmGt[i][0]
            CmGtArray[np.size(mu):,i] = CmGt[i][1]
        dm = CmGtArray.dot(TddGdmpmArray)
        dmu = (muPrior - mu) + dm[:np.size(mu)]
        drho = (rhoPrior - rho) + dm[np.size(mu):]
        
        residuals = np.sum(dmu**2) / np.sum(mu**2) + np.sum(drho**2) / np.sum(rho**2)
        ddArray = datumDicts2array([dd])
        print("Residual:",residuals)
        print("Incrementing mu and rho")
        print("dmu:",0.1*dmu)
        print("drho:",0.1*drho)
        mu += 0.1*dmu
        rho += 0.1*drho
        print("Storing model")
        modelFile = "results/model-{}.npz".format(iter)
        np.savez(modelFile,mu=mu,rho=rho,Kmu=Kmu,Krho=Krho)
        #muPrior = mu.copy()
        #rhoPrior = rho.copy()
        
        #TCI.m = mu - np.log(neTCIModel.m/Kmu)
        #TCI.clearCache()
        #plotWavefront(TCI,rays,save=False)
        iter += 1
    print('Finished inversion with {0} iterations'.format(iter))
    #print(rays)
    #TCI.m = Kmu*np.exp(mu) - neTCIModel.m
    #TCI.clearCache()
    #plotWavefront(TCI,rays,save=False)
    #plotWavefront(f.nFunc.subs({'t':0}),rays,*getSolitonCube(sol),save = False)
    #plotFuncCube(f.nFunc.subs({'t':0}), *getSolitonCube(sol),rays=rays)


if __name__=='__main__':
    np.random.seed(1234)
    #testSquare()
    #testSweep()
    SimulatedDataInversion(4,noise=None,eta=1.)
    #SimulatedDataInversionMCMC(4,noise=None,eta=1.)
    #testThreadedFermat()
    #testSmoothify()
    #testcseLam()


# In[ ]:



