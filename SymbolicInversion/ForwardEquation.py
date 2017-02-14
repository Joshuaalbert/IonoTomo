
# coding: utf-8

# In[1]:

from IRI import *
import numpy as np
from FermatPrincipleThreaded import *
from RealData import *
from Symbolic import *
from sympy import symbols,lambdify,Matrix
from time import time as tictoc
import threading
from scipy.integrate import simps
import pp
import sympy
import numpy
import sympy.utilities.decorator
import inspect
import pp

###
# Forward model defines the symbolic kernel and it's derivative wrt to an ordered parameter list (index + map)
# It then performs tracing to derivate the ray trajectories.
# The forward model is then functionals along the ray tracjectories.
# Cached ray trajectories speeds the process.
# Parallel computation of ray trajectories is possible (numThreads = 1).
###

class ForwardModel(Model):
    def __init__(self,numAntennas,numDirections,numTimes,
                 pathlength=2000,filename=None,numThreads=1,
                 numSolitons = 1,radioArray = None,frequency = 120e6,**kwargs):
        '''The ionosphere is modelled as the IRI plus a set of solitons
        with time coherence imposed via linear coherence over short time intervals'''
        super(ForwardModel,self).__init__(**kwargs)
        self.filename=filename
        self.numDirections = numDirections
        self.numTimes = numTimes
        self.numAntennas = numAntennas
        self.pathlength = pathlength
        self.numThreads = numThreads
        self.frequency = frequency#for now use single freq
        #neFunc from iri and solitons
        if radioArray is None:
            radioArray = RadioArray(arrayFile='arrays/lofar.hba.antenna.cfg')
        self.radioArray = radioArray
        self.iriModel = IriModel(radioArray = self.radioArray)
        self.solitonsModel = SolitonModel(numSolitons=numSolitons,radioArray = self.radioArray)
        self.enu = self.solitonsModel.enu
        self.neFunc = self.iriModel.iriFunc + self.solitonsModel.solitonsFunc
        #kernel and param dict
        self.forwardKernelFunc,self.forwardJKernelFunc,self.forwardKernelMap = self.makeSymbolicForwardKernel()
        self.forwardKernelParamDict = self.initForwardKernelParams.copy()
        self.numForwardKernelParams = len(self.forwardKernelOrder)
        print("Generated forward kernel symbolic function with {0} params".format(self.numForwardKernelParams))
        
    def getDatumIdx(self,antIdx,dirIdx,timeIdx):
        idx = antIdx*self.numDirections*self.numTimes + dirIdx*self.numTimes + timeIdx
        return idx
    
    def getIndicies(self,datumIdx):
        timeIdx = datumIdx % self.numTimes
        dirIdx = ((datumIdx - timeIdx)/self.numTimes) % self.numDirections
        antIdx = (datumIdx - timeIdx - dirIdx*self.numTimes)/self.numDirections/self.numTimes
        return antIdx,dirIdx,antIdx
    
    def setForwardKernelParams(self,paramVec):
        '''Set the paramDict for kernel from vector'''
        self.forwardKernelParamDict = self.makeParamDict(paramVec,self.forwardKernelOrder)
        
    def getForwardKernelParams(self):
        return self.makeParamVec(self.forwardKernelParamDict,self.forwardKernelOrder)
    
    def makeSymbolicForwardKernel(self):
        '''Create the kernel G^ijk (for each direction, time, antenna) 
        
        such that int_(R^i) G^i ds = data^i
        
        d^i(t) = int ne(beta;x,y,z,t) R^i(x,y,z,t) - rho^antIdx(t) R^i(x,y,z,t) - psi(t) R^i(x,y,z,t) dV
        kernel is three terms:
        
        ne(beta;x,y,z,t) R^i(x,y,z,t)
            path integral of ne along ray^i
            
        rho^antIdx(t) R^i(x,y,z,t) **not incorperated yet**
            antenna based gain for direction i (systematics that are not spatially correlated)
        
        psi(t) R^i(x,y,z,t)
            offset which should equal int ne(beta;x,y,z,t) R^j(x,y,z,t) - rho^j(t) R^i(x,y,z,t) for some j
        '''
        self.initForwardKernelParams = self.solitonsModel.initSolitonsParams.copy()
        self.initForwardKernelParams.update(self.iriModel.initIriParams.copy())
        count = 0
        self.forwardKernelFunc = []
        
        self.forwardKernelMap = {}
        dirIdx = 0
        while dirIdx < self.numDirections:
            timeIdx = 0
            while timeIdx < self.numTimes:
                rho = symbols("rho_{0}_{1}".format(dirIdx,timeIdx))#global offset
                self.initForwardKernelParams[rho.name] = 28.5
                self.forwardKernelFunc.append(self.neFunc/Rational(1e13) - rho/self.pathlength)
                
                
                antIdx = 0
                while antIdx < self.numAntennas:
                    datumIdx = self.getDatumIdx(antIdx,dirIdx,timeIdx)
                    self.forwardKernelMap[datumIdx] = count#the slot that datumIdx corresponds to
                    antIdx += 1
                count += 1
                timeIdx += 1
            dirIdx += 1
        self.forwardKernelOrder = self.makeOrderList(self.initForwardKernelParams)  
        self.forwardJKernelFunc = []
        for kernel in self.forwardKernelFunc:
            J = []
            for param in self.forwardKernelOrder:
                J.append(kernel.diff(param))
            self.forwardJKernelFunc.append(J)
        return self.forwardKernelFunc,self.forwardJKernelFunc,self.forwardKernelMap
    
    def generateKernel(self,paramVec=None,load=False):
        '''Create the kernel G^i such that int_(R^i) G^i ds = data^i
        
        d^i(t) = int ne(beta;x,y,z,t) R^i(x,y,z,t) - rho^antIdx(t) R^i(x,y,z,t) - psi(t) R^i(x,y,z,t) dV
        kernel is three terms:
        
        ne(beta;x,y,z,t) R^i(x,y,z,t)
            path integral of ne along ray^i
            
        rho^antIdx(t) R^i(x,y,z,t) **not incorperated yet**
            antenna based gain for direction i (systematics that are not spatially correlated)
        
        psi(t) R^i(x,y,z,t)
            offset which should equal int ne(beta;x,y,z,t) R^j(x,y,z,t) - rho^j(t) R^i(x,y,z,t) for some j
        '''
        print("Generating symbolic kernel for {0} rays on {1} threads".format(self.numAntennas*self.numDirections*self.numTimes,self.numThreads))
        if self.filename is not None and load:
            try:
                file = np.load('output/{0}-forwardKernel.npz'.format(self.filename))
                self.forwardKernel = file['forwardKernel']
                self.forwardJKernel = file['forwardJKernel']
                self.forwardKernelMap = file['forwardKernelMap'].item(0)
            except:
                print("Failed to load {0}".format(self.filename))
        else:   
            if paramVec is not None:
                self.setForwardKernelParams(paramVec)
            def ppGenerateKernels(gFile,forwardKernelParamDict):
                return FermatPrincipleCartesian.generateKernel(gFile,forwardKernelParamDict)
            t1 = tictoc()

            jobs = []
            job_server = pp.Server(self.numThreads, ppservers=())
            idx = 0
            while idx < len(self.forwardKernelFunc):
                file = 'kernels/forwardKernel-{0}.npz'.format(idx)
                np.savez(file,Gk=self.forwardKernelFunc[idx],
                        Jk = self.forwardJKernelFunc[idx])
                #file = sol.saveNeFunc(neFunc)
                job = job_server.submit(ppGenerateKernels,
                           args=(file,
                                 self.forwardKernelParamDict),
                           depfuncs=(),
                           modules=('FermatPrincipleCartesian',),
                           globals={})
                jobs.append(job)
                idx += 1
            #collapse results
            self.forwardKernel = []
            self.forwardJKernel = []
            for job in jobs:
                result = job()
                self.forwardKernel.append(result['G'])
                self.forwardJKernel.append(result['J'])
            print("Made {0} symbolic kernels in {1} seconds".format(len(self.forwardKernel) + len(self.forwardKernel)*len(self.forwardKernelParamDict),tictoc() - t1))
            job_server.print_stats()
            job_server.destroy()
            if self.filename is not None:
                    np.savez('output/{0}-forwardKernel.npz'.format(self.filename),
                             forwardKernel = self.forwardKernel,
                             forwardJKernel = self.forwardJKernel,
                             forwardKernelMap = self.forwardKernelMap)
     
        return self.forwardKernel,self.forwardJKernel
            
    def calcRays(self,inRays,N=100,load=False):
        '''Calculate the ray trajectories in parallel.'''
        if self.filename is not None and load:
            try:
                rays = np.load('output/{0}-rays.npz'.format(self.filename))['rays'].item(0)
            except:
                print("Failed to load {0}".format(self.filename))
        else: 

            def ppRayProp(file,inRays,N,pathlength,frequency):
                '''
                file contains a symbolic function
                inRay contains a list of ray objects
                '''
                fermat =  FermatPrincipleCartesian.Fermat(neFunc = None,type = 's',frequency = frequency)
                fermat.loadFunc(file)
                rays = {}
                for ray in inRays:
                    datumIdx = ray.id
                    origin = ray.origin
                    direction = ray.dir
                    time = ray.time
                    x,y,z,s = fermat.integrateRay(origin,direction,pathlength,time=time,N=N)
                    rays[datumIdx] = {'x':x,'y':y,'z':z,'s':s}
                return rays
            
            t1 = tictoc()
            chunkedRays = {i:[] for i in range(self.numThreads)}
            count = 0
            for ray in inRays:
                origin = ray.origin#ac.SkyCoord(*(ray.origin*au.km),frame=self.enu).transform_to('itrs').cartesian.xyz.to(au.km).value
                direction = ray.dir#ac.SkyCoord(*ray.dir,frame=self.enu).transform_to('itrs').cartesian.xyz.value
                chunkedRays[count%self.numThreads].append(Ray(origin,direction,time = ray.time,id=ray.id))
                count += 1
            print("Calculating rays on {0} threads".format(self.numThreads))
            neFunc = self.neFunc.subs(self.forwardKernelParamDict)
            file = 'kernels/neFunc.npz'
            np.savez(file,neFunc=neFunc)
            jobs = []
            job_server = pp.Server(self.numThreads, ppservers=())
            for i in range(self.numThreads):
                #file = sol.saveNeFunc(neFunc)
                job = job_server.submit(ppRayProp,
                           args=(file,
                                 chunkedRays[i],
                                 N,
                                 self.pathlength,
                                 self.frequency),
                           depfuncs=(),
                           modules=('FermatPrincipleCartesian',),
                           globals={})
                jobs.append(job)
            #collapse results
            rays = {}
            for job in jobs:
                result = job()
                rays.update(result)
            print("Calculated {0} rays in {1} seconds".format(len(rays),tictoc() - t1))
            job_server.print_stats()
            job_server.destroy()
            if self.filename is not None:
                np.savez('output/{0}-rays.npz'.format(self.filename),rays=rays)
        return rays
    
    def doForward(self,inRays,N=100,load=False):
        '''Run forward model by calculating the ray trajectories from which kernels can be integrated.
        Could also add the kernel as an euler eqns too in the future.
        ``antennas`` is list of ENU frame origins of antennas
        ``directions`` is list of ENU frame directions
        ``times`` is list of astropy.time objects
        ``N`` is the resolution along path to split'''
        print("Doing forward")
        rays = self.calcRays(inRays,N=N,load=load)
        self.generateKernel(paramVec=None,load=load)
        def ppForward(raySet,file):
            '''ray is a dictionary {'x':ndarray,'y':ndarray,'z':ndarray,'s':ndarray}
            kernel is a symbolic function of (x,y,z,t) symbols
            time is a double'''
            result = FermatPrincipleCartesian.calcForwardKernel(raySet,file)
            return result
        #each same kernel for all antennas, different per time and dir
        t1 = tictoc()
        print("Integration forward kernel on {0} threads".format(self.numThreads))
        ncpus = self.numThreads
        # Creates jobserver with ncpus workers
        jobs = []
        
        job_server = pp.Server(self.numThreads, ppservers=())
        
        #build ray sets
        invKernelMap = {i:[] for i in range(len(self.forwardKernel))}
        rayIdx = 0
        while rayIdx < len(rays):
            datumIdx = inRays[rayIdx].id
            invKernelMap[self.forwardKernelMap[datumIdx]].append([inRays[rayIdx],rays[rayIdx]])
            rayIdx += 1
        kernelIdx = 0
        while kernelIdx < len(self.forwardKernel):
            kernelRays = invKernelMap[kernelIdx]
            kernel = self.forwardKernel[kernelIdx]
            Jkernel = self.forwardJKernel[kernelIdx]
            file = 'kernels/substitutedForwardKernel-{0}.npz'.format(kernelIdx)
            np.savez(file,kernel=kernel,Jkernel=Jkernel)
            job = job_server.submit(ppForward,
                       args=(kernelRays,file),
                       depfuncs=(),
                       modules=('FermatPrincipleCartesian',))
            jobs.append(job)
            kernelIdx += 1        

        resultCollection = {}

        kernelIdx = 0
        while kernelIdx < len(self.forwardKernel):
            job = jobs[kernelIdx]
            result = job()
            #print(result)
            resultCollection.update(result)
            kernelIdx += 1
        
        g = np.zeros(len(rays))
        J = np.zeros([len(rays),len(self.forwardKernelOrder)])
        count = 0
        for ray in inRays:
            datumIdx = ray.id
            g[count] = resultCollection[datumIdx]['g']
            J[count,:] = resultCollection[datumIdx]['J']
            count += 1
        print("Calculated {0} kernels in {1} seconds".format(len(rays),tictoc() - t1))
        job_server.print_stats()
        job_server.destroy()        
        return g,J
    
    def doJkernel(self,inRays,N=100,load=False):
        '''Run forward model by calculating the ray trajectories from which kernels can be integrated.
        Could also add the kernel as an euler eqns too in the future.
        ``antennas`` is list of ENU frame origins of antennas
        ``directions`` is list of ENU frame directions
        ``times`` is list of astropy.time objects
        ``N`` is the resolution along path to split'''
        pass
        
            
if __name__=='__main__':
    dataDict = PrepareData(infoFile='SB120-129/WendysBootes.npz',
                           dataFolder='SB120-129/',
                           timeStart = 0, timeEnd = 0,
                           arrayFile='arrays/lofar.hba.antenna.cfg',load=True)
    
    model = ForwardModel(dataDict['numAntennas'],dataDict['numDirections'],dataDict['numTimes'],
                 pathlength=2000,filename='model-test',numThreads=8,
                 numSolitons = 1,radioArray = None)
    
    paramVec = np.load('Inversion0.npz')['paramVec']
    model.setForwardKernelParams(paramVec)
    model.iriModel.plotModel()
    res = 1.
    iter = 0
    while np.abs(res) > 1e-6:
        print("Inversion iter:",iter)
        dobs = dataDict['dtec']
        g,J = model.doForward(dataDict['rays'],N=100,load=False)
        dd = dobs - g
        Cd = np.eye(len(dd))*np.mean(dd**2)
        Cdinv = np.linalg.pinv(Cd)
        jtC = J.transpose().dot(Cdinv)
        wdd = jtC.dot(dd)
        S = jtC.dot(J)
        T = np.linalg.pinv(S)
        dbeta = T.dot(wdd)
        paramVec = model.getForwardKernelParams()
        res = np.mean(dbeta/paramVec)
        paramVec += dbeta
        model.setForwardKernelParams(paramVec)
        print("progress:",np.sum(dd))
        print("res:",res)
        iter += 1
    np.savez('Inversion0.npz',paramVec = model.getForwardKernelParams())
        


# In[44]:

paramVec = model.getForwardKernelParams()
import pylab as plt
plt.plot(paramVec)
plt.show()


# In[3]:

paramVec = np.load('Inversion0.npz')['paramVec']
print(paramVec)
plotFuncCube(model.solitonsModel.solitonsFunc.subs(model.forwardKernelParamDict),*getSolitonCube(model))


# In[ ]:



