
# coding: utf-8

# In[6]:

import numpy as np
from scipy.integrate import odeint
from scipy.integrate import ode

import numpy as np
from sympy import symbols,sqrt,sech,Rational,lambdify,Matrix,exp,cosh,cse,simplify,cos,sin
from sympy.vector import CoordSysCartesian

from theano.scalar.basic_sympy import SymPyCCode
from theano import function
from theano.scalar import floats

from IRI import *
from Symbolic import *
from ENUFrame import ENU

import astropy.coordinates as ac
import astropy.units as au
import astropy.time as at

#from multiprocessing import Pool
import dill
dill.settings['recurse'] = True
#import sys
#sys.setrecursionlimit(30000)

import threading
from time import sleep,time

import cloudpickle

class Fermat(object):
    def __init__(self,nFunc=None,neFunc=None,frequency = 120e6,type='s'):
        self.type = type
        self.frequency = frequency#Hz
        if nFunc is not None and neFunc is not None:
            self.nFunc = nFunc
            self.neFunc = neFunc
            self.eulerLambda, self.jacLambda = self.generateEulerEqnsSym(self.nFunc)
            return
        if nFunc is not None:
            self.nFunc = nFunc
            self.neFunc = self.n2ne(nFunc)
            self.eulerLambda, self.jacLambda = self.generateEulerEqnsSym(self.nFunc)
            return
        if neFunc is not None:
            self.neFunc = neFunc
            self.nFunc = self.ne2n(neFunc)
            self.eulerLambda, self.jacLambda = self.generateEulerEqnsSym(self.nFunc)
            return
    def loadFunc(self,filename):
        '''Load symbolic functions'''
        file = np.load(filename)
        if 'neFunc' in file.keys():
            self.neFunc = file['neFunc']
            self.nFunc = self.ne2n(neFunc)
            self.eulerLambda, self.jacLambda = self.generateEulerEqnsSym(self.nFunc)
            return
        if 'nFunc' in file.keys():
            self.nFunc = file['nFunc']
            self.neFunc = self.n2ne(nFunc)
            self.eulerLambda, self.jacLambda = self.generateEulerEqnsSym(self.nFunc)
            return
        
    def ne2n(self,neFunc):
        '''Analytically turn electron density to refractive index. Assume ne in m^-3'''
        self.neFunc = neFunc
        #wp = 5.63e4*np.sqrt(ne/1e6)/2pi#Hz^2 m^3 lightman p 226
        fp2 = 8.980**2 * neFunc
        self.nFunc = sqrt(Rational(1) - fp2/self.frequency**2)
        return self.nFunc
    
    def n2ne(self,nFunc):
        """Get electron density in m^-3 from refractive index"""
        self.nFunc = nFunc
        self.neFunc = (Rational(1) - nFunc**2)*self.frequency**2/8.980**2
        return self.neFunc
        
    def euler(self,px,py,pz,x,y,z,s,t=0):
        N = np.size(px)
        euler = np.zeros([7,N])
        i = 0
        while i < 7:
            euler[i,:] = self.eulerLambda[i](px,py,pz,x,y,z,s,t)
            i += 1
        return euler
    
    def eulerODE(self,y,t,*args):
        '''return pxdot,pydot,pzdot,xdot,ydot,zdot,sdot'''
        #print(y)
        e = self.euler(y[0],y[1],y[2],y[3],y[4],y[5],y[6],args[0]).flatten()
        #print(e)
        return e
    
    def jac(self,px,py,pz,x,y,z,s,t=0):
        N = np.size(px)
        jac = np.zeros([7,7,N])
        i = 0
        while i < 7:
            j = 0
            while j < 7:
                jac[i,j,:] = self.jacLambda[i][j](px,py,pz,x,y,z,s,t)
                j += 1
            i += 1
        return jac
    
    def jacODE(self,y,t,*args):
        '''return d ydot / d y'''
        j = self.jac(y[0],y[1],y[2],y[3],y[4],y[5],y[6],args[0]).reshape([7,7])

        #print('J:',j)
        return j
        
    def generateEulerEqnsSym(self,nFunc=None):
        '''Generate function with call signature f(t,y,*args)
        and accompanying jacobian jac(t,y,*args), jac[i,j] = d f[i] / d y[j]'''
        if nFunc is None:
            nFunc = self.nFunc
        x,y,z,px,py,pz,s,t = symbols('x y z px py pz s t')
        if self.type == 'z':
            sdot = nFunc / pz
            pxdot = nFunc.diff('x')*nFunc/pz
            pydot = nFunc.diff('y')*nFunc/pz
            pzdot = nFunc.diff('z')*nFunc/pz

            xdot = px / pz
            ydot = py / pz
            zdot = Rational(1)
        
        if self.type == 's':
            sdot = Rational(1)
            pxdot = nFunc.diff('x')
            pydot = nFunc.diff('y')
            pzdot = nFunc.diff('z')

            xdot = px / nFunc
            ydot = py / nFunc
            zdot = pz / nFunc

        eulerEqns = (pxdot,pydot,pzdot,xdot,ydot,zdot,sdot)

        euler = [lambdify((px,py,pz,x,y,z,s,t),eqn,"numpy") for eqn in eulerEqns]
        self.eulerLambda = euler
        jac = []
        
        for eqn in eulerEqns:
            jac.append([lambdify((px,py,pz,x,y,z,s,t),eqn.diff(var),"numpy") for var in (px,py,pz,x,y,z,s)])
        self.jacLambda = jac
        
        return self.eulerLambda, self.jacLambda
    def integrateRay(self,X0,direction,tmax,time = 0,N=100):
        '''Integrate rays from x0 in initial direction where coordinates are (r,theta,phi)'''
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
        Y,info =  odeint(self.eulerODE, init, tarray, args=(time,),Dfun = self.jacODE, col_deriv = 0, full_output=1)
        #print(info['hu'].shape,np.sum(info['hu']),info['hu'])
        #print(Y)
        x = Y[:,3]
        y = Y[:,4]
        z = Y[:,5]
        s = Y[:,6]
        return x,y,z,s    
    
def synchronized(func):
    func.__lock__ = threading.Lock()
    def synced_func(*args, **kws):
        with func.__lock__:
            return func(*args, **kws)
    return synced_func
    
class FermatIntegrationThread (threading.Thread):
    '''Only works with one thread running per time.'''
    def __init__(self, fermat, threadId):
        super(FermatIntegrationThread,self).__init__()
        self.fermat = fermat
        self.stop = False
        self.threadId = threadId
        self.numJobs = 0
        self.jobIdx = 0
        self.jobs = {}
        self.resultMap = {}
        self.results = {}
        self.totalTime = 0.
        
    def addJob(self,X0,direction,tmax,time = 0,N=100,resultIdx=None):
        '''Add a ray job return the result index'''
        self.jobs[self.numJobs] = (X0,direction,tmax,time,N)
        if resultIdx is None:
            resultIdx = self.numJobs
        self.resultMap[self.numJobs] = resultIdx
        self.numJobs += 1
        return self.numJobs - 1
    
    def kill(self):
        self.stop = True
        
    def isEmpty(self):
        return self.jobIdx == self.numJobs
    def getAverageIntegrationTime(self):
        if self.numJobs > 0:
            return self.totalTime/self.numJobs
        else:
            return 0.
        
    def run(self):
        print ("Starting thread-{0}".format(self.threadId))
        while not self.stop or not self.isEmpty():
            while self.jobIdx < self.numJobs:
                tic = time()
                x,y,z,s = self.fermat.integrateRay(*self.jobs[self.jobIdx])
                self.totalTime += time() - tic
                self.results[self.resultMap[self.jobIdx]] = {'x':x,'y':y,'z':z,'s':s}
                self.jobIdx += 1
            #sleep(self.getAverageIntegrationTime())
        print ("Stopping thread-{0}".format(self.threadId))
        print("Number of jobs done: {0}".format(self.numJobs))
        print("Integration time: {0} s [total] | {1} s [average]".format(self.totalTime,self.getAverageIntegrationTime()))

        


def testThreadedFermat():
    sol = SolitonModel(5)
    neFunc = sol.generateSolitonsModel()
    f =  Fermat(neFunc = neFunc,type = 's')
    n = 1
    threads = []
    for i in range(n):
        threads.append(FermatIntegrationThread(f,i))
        threads[i].start()
    
    count = 0
    
    theta = np.linspace(-np.pi/8.,np.pi/8.,5)
    #phi = np.linspace(0,2*np.pi,6)
    rays = []
    origin = ac.ITRS(sol.enu.location).cartesian.xyz.to(au.km).value
    for t in theta:
        for p in theta:
            direction = ac.SkyCoord(np.sin(t),
                                    np.sin(p),
                                    1.,frame=sol.enu).transform_to('itrs').cartesian.xyz.value
            threads[count % n].addJob(origin,direction,1000,0.,100,resultIdx = count)
            count += 1
    for i in range(n):
        threads[i].kill()
    #print('waiting for completion')
    for i in range(n):
        threads[i].join()
        #pass
        
    #plotWavefront(f.nFunc.subs({'t':0}),rays,*getSolitonCube(sol))
    
if __name__=='__main__':
    np.random.seed(1234)
    #testSquare()
    #testSweep()
    testThreadedFermat()
    #testSmoothify()
    #testcseLam()


# In[1]:

import numpy as np

import dill
dill.settings['recurse'] = True

from sympy import symbols,sqrt,Rational,lambdify,Matrix,exp,cosh,cse,simplify,cos,sin
from sympy.vector import CoordSysCartesian

def createInitSolitonParam():
    '''Create an initial random param for a soliton'''
    #initial amp
    amp = 10**np.random.uniform(low = 9.5, high = 10.5)#electron / m^3
    #initial velcoity
    maxVel = 350./3600.#100km/hour in km/s pi*(6300+350)*2/24.*0.2 (20% of solar pressure field movement)
    initc = [np.random.uniform(low=-maxVel,high=maxVel),
             np.random.uniform(low=-maxVel,high=maxVel),
             np.random.uniform(low=-maxVel,high=maxVel)]
                        
    #initial  location of blobs
    initx = [np.random.uniform(low=-100,high=100),
             np.random.uniform(low=-100,high=100),
             np.random.uniform(low=50,high=800)]
    b = np.random.uniform(low = 10.,high=100.)

    return {"A":np.sqrt(amp),
                       "cx":initc[0],
                       "cy":initc[1],
                       "cz":initc[2],
                       "x0":initx[0],
                       "y0":initx[1],
                       "z0":initx[2],
                       "b":b
                      }
    
solitonsFunc = None
initSolitonsParams = {}
i = 0
while i < 100:
    A = symbols ('A_{0}'.format(i))
    cx = symbols ('cx_{0}'.format(i))
    cy = symbols ('cy_{0}'.format(i))
    cz = symbols ('cz_{0}'.format(i))
    x0 = symbols ('x0_{0}'.format(i))
    y0 = symbols ('y0_{0}'.format(i))
    z0 = symbols ('z0_{0}'.format(i))
    b = symbols ('b_{0}'.format(i))
    i += 1
    init = createInitSolitonParam()

    initSolitonsParams[A.name] = init['A']
    initSolitonsParams[cx.name] = init['cx']
    initSolitonsParams[cy.name] = init['cy']
    initSolitonsParams[cz.name] = init['cz']
    initSolitonsParams[x0.name] = init['x0']
    initSolitonsParams[y0.name] = init['y0']
    initSolitonsParams[z0.name] = init['z0']
    initSolitonsParams[b.name] = init['b']

    x,y,z,t = symbols('x,y,z,t')

    N = CoordSysCartesian('N')
    c = cx*N.i + cy*N.j + cz*N.k  
    X = x*N.i + y*N.j + z*N.k
    X0 = x0*N.i + y0*N.j + z0*N.k
    xx0 = X - t*c - X0
    func = A*A* exp(-xx0.dot(xx0)/b**Rational(2))
    if solitonsFunc is None:
        solitonsFunc = func
    else:
        solitonsFunc += func

print("Function: {0}".format(solitonsFunc))
solitonsFunc = solitonsFunc.subs(initSolitonsParams)
print("Lambdifying")
lam = lambdify(symbols('x,y,z,t'),solitonsFunc,"numpy")
dill.dump(lam,file("lambdified_func",'wb'))



# In[4]:

help(dill.dump)


# In[ ]:



