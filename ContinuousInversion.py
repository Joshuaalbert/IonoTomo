
# coding: utf-8

# In[1]:

import numpy as np

from scipy.integrate import simps
#my things
from FermatPrincipleCartesian import *
from Geometry import *
from Symbolic import *
from sympy import Matrix
from RealData import PrepareData
from ForwardEquation import *




def LMSolContinous(dataDict,mu = 0.5):
    '''
    ``rays`` origin and dir are in ENU frame.
    data is d = dtec = int_i ne ds - int_i0 ne ds.
        neFunc = f(beta)
        g(beta) = int_i f(beta) + rho_i ds - int_i0 f(beta) + rho_i0 ds
        minimize (dobs - d)Cdinv(dobs - d) + mu (log(neFunc)  - log(neprior))Cminv(log(neFunc) - log(neprior))
        
        Solve in continuous basis.
    Steps:
    1. propagate rays
    2. dd = d - g
    3. wdd = Cdinv.dd
    4. S = G^t.Cdinv.G + mu*lambda^t.Cminv.lambda
    5. T = Sinv
    6. dm = T.G^t.wdd
    '''
    #first fit just iri layers and global offsets
    Nsol = 0
    print("Constructing the model with {0} solitons".format(Nsol))
    model = ForwardModel(dataDict['numAntennas'],dataDict['numDirections'],dataDict['numTimes'],
                 pathlength=2000,filename='ww-background',numThreads=1,
                 numSolitons = Nsol,radioArray = None)
    #a priori 
    params = model.getForwardKernelParams()
    g = model.doForward(dataDict['rays'],N=100,load=False)
    dd = dataDict['dtec'] - g
    Cd = np.eye(np.size(params))*np.var(g)*1.2
    Cdinv = np.linalg.pinv(Cd)
    wdd = Cdinv.dot(dd)
    rays = model.calcRays(dataDict['rays'],load=True)
    plotWavefront(lambda x,y,z : model.generateSolitonModel()(x,y,z,0),rays,*getSolitonCube(model))
    g = model.doForward(dataDict['rays'],N=100,load=True)
    dd = dataDict['dtec'] - g
    print("Computing observation covariance.")
    Cd = np.eye(np.size(params))*np.var(g)*1.2
    Cdinv = np.linalg.pinv(Cd)
    J = self.doJkernel(inRays,N=100,load=True)
    S = J.transpose().dot(Cdinv).dot(J)
    T = np.linalg.pinv(S)
    
    wdd = J.transpose().dot(Cdinv).dot(dd)
    dbeta = T.dot(wdd)
    params += dbeta
    model.setModelParams(params)
    #monte carlo L.Cminv.L
    #neFunc = model.solitonModelSymbolic
    #paramDict = self.getModelParamDict()
    #L = []
    #for param i paramDict.keys():
    #    L.append(neFunc.diff(param))
    
    
def testForwardProblem():
    sol = SolitonModel(8)
    
    neFunc = sol.generateSolitonModel()

    theta = np.linspace(-np.pi/8.,np.pi/8.,2)
    #phi = np.linspace(0,2*np.pi,6)
    rays = []
    origin = ac.ITRS(sol.enu.location).cartesian.xyz.to(au.km).value
    for t in theta:
        for p in theta:
            direction = ac.SkyCoord(np.sin(t),
                                    np.sin(p),
                                    1.,frame=sol.enu).transform_to('itrs').cartesian.xyz.value
            rays.append(Ray(origin,direction))
    forwardProblem = ForwardProblem(sol)
    times = np.zeros(len(rays))
    d = forwardProblem.doForward(rays,times,N=1000)
    print(d)
    #plotWavefront(f.nFunc.subs({'t':0}),rays,*getSolitonCube(sol))
    #plotFuncCube(f.nFunc.subs({'t':0}), *getSolitonCube(sol),rays=rays)
    
    
if __name__ == '__main__':
    np.random.seed(1234)
    #testForwardProblem()
    dataDict = PrepareData(infoFile='SB120-129/WendysBootes.npz',
                           dataFolder='SB120-129/',
                           timeStart = 0, timeEnd = 0,
                           arrayFile='arrays/lofar.hba.antenna.cfg',load=True)
    LMSolContinous(dataDict,mu = 0.5)
    #LMSolContinous(**dataDict)


# In[ ]:

import pylab as plt
plt.hist(dataDict['dtec'])
plt.show()


# In[ ]:



