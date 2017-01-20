
# coding: utf-8

# In[4]:

import numpy as np
from sympy import Rational,symbols,exp,lambdify,sqrt,tanh,log,pi
from RadioArray import RadioArray
from ENUFrame import ENU
import astropy.coordinates as ac
import astropy.units as au
import astropy.time as at
from sympy.vector import CoordSysCartesian
from Symbolic import *

from TricubicInterpolation import TriCubic
import tempfile

def neQuick(h,No=2.2e11,hmax=368.,Ho=50.):
    '''Ne quick model for one layer'''
    res = np.zeros_like(h)
    g = 1./7.
    rfac = 100.
    dh = h - hmax
    g1 = g * dh
    z = dh / (Ho * (1. + rfac * g1 / (rfac * Ho + g1)))
    ee = np.exp(z)
    dial = ee / 1e7 - 1
    sig = 1./(1. + np.exp(-dial))
    res = No*4.0*ee/(1.0 + ee)**2
    #res[ee>1e7] = No*4.0/ee[ee>1e7]
    #res[ee<=1e7] = No*4.0*ee[ee<=1e7]/(1.0 + ee[ee<=1e7])**2
    #res[z > 40] = 0
    return res

def xef1f2(h):
    f2 = neQuick(h,No=2.2e11,hmax=359.,Ho=46.)
    f1 = neQuick(h,No=3e9,hmax=195.,Ho=20.)
    e = neQuick(h,No=3e9,hmax=90.,Ho=10.)
    return f2 + f1 + e


def symbolicChapman(h, nepeak,hmax,width):
    y = (h-hmax)/width
    return nepeak*exp(Rational(1,2)*(Rational(1) - y - exp(-y)))


def symbolicChapman_def1f2(h = None):
    if h is None:
        h = symbols('h')
    #f = symbolicChapman(h,Rational(1e10),Rational(450),Rational(250))
    f2 = symbolicChapman(h,Rational(2.2e11),Rational(359),Rational(46))
    f1 = symbolicChapman(h,Rational(3e9),Rational(195),Rational(20))
    e = symbolicChapman(h,Rational(3e9),Rational(90),Rational(10))
    #d = symbolicChapman(h,Rational(3e9),Rational(80),Rational(50))
    return f2 + f1 + e  + Rational(4.5e8)
    
def ExampleIRI():
    d = np.genfromtxt('exampleIRI.txt',names=True)
    profile = d['ne']
    return d['height'],d['ne']
    
def plotModels():
    lam = lambdify(symbols('h'),symbolicChapman_def1f2(),'numpy')
    
    import pylab as plt
    h1,ne1 = ExampleIRI()
    plt.plot(h1,ne1,c='black',label='Ex. IRI')
    ne3 = xef1f2(h1)
    plt.plot(h1,ne3,c='blue',label='neQuick_ef1f2')
    ne4 = lam(np.linspace(0,3000,10000))
    plt.plot(np.linspace(0,3000,10000),ne4,c='green',label='symbolicChapman_def1f2')
    plt.legend(frameon=False)
    plt.xlabel('Height above surface (km)')
    plt.ylabel(r'Electron density $n_e$ (${\rm m}^{-3}$)')
    #plt.yscale('log')
    plt.grid()
    plt.title('Ionosphere Models')
    plt.show()
    
class Model(object):
    '''Symbolic model object. '''
    def makeOrderList(self,paramDict):
        orderList = []
        for key in paramDict.keys():
            orderList.append(key)
        return orderList
    def makeParamDict(self,paramVec,orderList):
        paramDict = {}
        N = np.size(paramVec)
        i = 0
        while i < N:
            paramDict[orderList[i]] = paramVec[i]
            i += 1
        return paramDict
    def makeParamVec(self,paramDict,orderList):
        N = len(orderList)
        paramVec = np.zeros(N)
        i = 0
        while i < N:
            paramVec[i] = paramDict[orderList[i]]
            i += 1
        return paramVec        

class ElectronContentModel(Model):
    def __init__(self,radioArray = None,**kwargs):
        super(ElectronContentModel,self).__init__(**kwargs)
        if radioArray is None:
            radioArray = RadioArray(arrayFile='arrays/lofar.hba.antenna.cfg')
        self.radioArray = radioArray
        self.enu = ENU(location=radioArray.getCenter().earth_location)
    def saveNeFunc(self,neFunc):
        f = tempfile.SpooledTemporaryFile()
        np.savez(f,neFunc=neFunc)
        f.flush()
        return f       
    
class IriModel(ElectronContentModel):
    def __init__(self,**kwargs):
        '''The ionosphere is modelled as the IRI plus a set of solitons
        with time coherence imposed via linear coherence over short time intervals'''
        super(IriModel,self).__init__(**kwargs)
        self.initIriParams = self.createInitIriParams()
        self.iriParamDict = self.initIriParams.copy()
        self.iriOrder = self.makeOrderList(self.initIriParams)
        self.numIriParams = len(self.iriOrder)
        self.iriFunc = self.makeSymbolicIri()
        print("Generated IRI symbolic function with {0} params".format(self.numIriParams))
        
    def setIriParams(self,paramVec):
        '''Set the paramDict for iri from vector'''
        self.iriParamDict = self.makeParamDict(paramVec,self.iriOrder)
        
    def getIriParams(self):
        return self.makeParamVec(self.iriParamDict,self.iriOrder)
        
    def createInitIriParams(self):
        '''Create an initial random param for a soliton'''
        init = {'fhm' : np.random.uniform(low = 400, high = 600),
                'fw' : np.random.uniform(low = 200,high = 300),
                'nefm' : 10**np.random.uniform(low = 9,high=11),
                'f2hm' : np.random.uniform(low = 300, high = 400),
                'f2w' : np.random.uniform(low = 150,high = 200),
                'nef2m' : 10**np.random.uniform(low = 11,high=12),
                'f1hm' : np.random.uniform(low = 100, high = 140),
                'f1w' : np.random.uniform(low = 60,high = 100),
                'nef1m' : 10**np.random.uniform(low = 9,high=11),
                'ehm' : np.random.uniform(low = 80, high = 120),
                'ew' : np.random.uniform(low = 50,high = 80),
                'neem' : 10**np.random.uniform(low = 9,high=10),
                'dhm' : np.random.uniform(low = 50, high = 80),
                'dw' : np.random.uniform(low = 50,high = 60),
                'nedm' : 10**np.random.uniform(low = 8,high=10)}
        init = {'f2hm' : 360,
                'f2w' : 46,
                'nef2m' : 2.2e11,
                'f1hm' : 195,
                'f1w' : 20,
                'nef1m' : 3e9,
                'ehm' : 90,
                'ew' : 10,
                'neem' : 3e9}
        return init
    
    def makeSymbolicIri(self):
        x,y,z = symbols('x y z')
        #R = Rational(6371)
        R = Rational(int(self.radioArray.getCenter().spherical.distance.to(au.km).value))
        r = sqrt(x**(Rational(2))+y**(Rational(2))+z**(Rational(2)))
        h = r - R
        #f = symbolicChapman(h,*symbols('nefm fhm fw'))
        f2 = symbolicChapman(h,*symbols('nef2m f2hm f2w'))
        f1 = symbolicChapman(h,*symbols('nef1m f1hm f1w'))
        e = symbolicChapman(h,*symbols('neem ehm ew'))
        #d = symbolicChapman(h,*symbols('nedm dhm dw'))
        self.iriFunc = f2 + f1 + e + Rational(4.5e8)
        return self.iriFunc
        
    def generateIri(self,paramVec=None):
        '''Sustitute paramDict into symbolic function'''
        if paramVec is not None:
            self.iriParamDict = self.setIriParams(paramVec)
        self.iriModel = self.iriFunc.subs(self.iriParamDict)
        return self.iriModel
    
    def evaluate(self,X,Y,Z):
        iri = lambdify(symbols('x y z'),self.generateIri(),'numpy')
        ne = iri(X.flatten(),Y.flatten(),Z.flatten()).reshape(X.shape)
        return ne
    
    def plotModel(self):
        import pylab as plt
        iri = lambdify(symbols('x y z'),self.generateIri(),'numpy')
        h = np.linspace(6300,9000,1000)
        plt.plot(h,iri(h,0,0))
        plt.xlabel('geocentric radius (km)')
        plt.ylabel('ne m^-3')
        plt.yscale('log')
        plt.grid()
        plt.show()
        

        
    
class SolitonModel(ElectronContentModel):
    def __init__(self,numSolitons=0,**kwargs):
        '''The ionosphere is modelled as the IRI plus a set of solitons
        with time coherence imposed via linear coherence over short time intervals.
        This class does solitons'''
        super(SolitonModel,self).__init__(**kwargs)
        self.numSolitons = 0#soliton index
        self.solitonsFunc = None
        self.initSolitonsParams = {}
        i = 0
        while i < numSolitons:
            self.addSoliton()
            i += 1
        self.solitonsParamDict = self.initSolitonsParams.copy()
        self.solitonsOrder = self.makeOrderList(self.initSolitonsParams)
        self.numSolitonsParams = len(self.solitonsOrder)
        print("Generated solitons symbolic function with {0} params".format(self.numSolitonsParams))

    def createInitSolitonParam(self):
        '''Create an initial random param for a soliton'''
        
        #initial amp
        amp = 10**np.random.uniform(low = 9.5, high = 10.5)#electron / m^3
        #initial velcoity
        maxVel = 350./3600.#100km/hour in km/s pi*(6300+350)*2/24.*0.2 (20% of solar pressure field movement)
        initc = ac.SkyCoord(np.random.uniform(low=-maxVel,high=maxVel)*au.km,
                           np.random.uniform(low=-maxVel,high=maxVel)*au.km,
                           np.random.uniform(low=-maxVel,high=maxVel)*au.km,
                           frame=self.enu).transform_to('itrs').cartesian.xyz.to(au.km).value
        #initial  location of blobs
        initx = ac.SkyCoord(np.random.uniform(low=-100,high=100)*au.km,
                           np.random.uniform(low=-100,high=100)*au.km,
                           np.random.uniform(low=50,high=800)*au.km,
                           frame=self.enu).transform_to('itrs').cartesian.xyz.to(au.km).value
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
            
    def setSolitonsParams(self,paramVec):
        '''Set the paramDict for iri from vector'''
        self.solitonsParamDict = self.makeParamDict(paramVec,self.solitonsOrder)
        
    def getSolitonsParams(self):
        return self.makeParamVec(self.solitonsParamDict,self.solitonsOrder)
    
    def reinitialzeSolitons(self):
        '''Sample new initial params for all solitons'''
        pass
        
    def addSoliton(self):
        '''Soliton consists of a function of the form
        A * exp(-(x-c * t - x0)**2/b**2)
        
        In 3D this gives A: 1, c: 3, x0: 3 ,b: 1-> 8 params per soliton
        '''
        init = self.createInitSolitonParam()
        
        A = symbols ('A_{0}'.format(self.numSolitons))
        cx = symbols ('cx_{0}'.format(self.numSolitons))
        cy = symbols ('cy_{0}'.format(self.numSolitons))
        cz = symbols ('cz_{0}'.format(self.numSolitons))
        x0 = symbols ('x0_{0}'.format(self.numSolitons))
        y0 = symbols ('y0_{0}'.format(self.numSolitons))
        z0 = symbols ('z0_{0}'.format(self.numSolitons))
        b = symbols ('b_{0}'.format(self.numSolitons))
        self.numSolitons += 1
        #soliton = self.createInitParam()
        init = self.createInitSolitonParam()
        
        self.initSolitonsParams[A.name] = init['A']
        self.initSolitonsParams[cx.name] = init['cx']
        self.initSolitonsParams[cy.name] = init['cy']
        self.initSolitonsParams[cz.name] = init['cz']
        self.initSolitonsParams[x0.name] = init['x0']
        self.initSolitonsParams[y0.name] = init['y0']
        self.initSolitonsParams[z0.name] = init['z0']
        self.initSolitonsParams[b.name] = init['b']

        
        x,y,z,t = symbols('x,y,z,t')
        
        N = CoordSysCartesian('N')
        c = cx*N.i + cy*N.j + cz*N.k  
        X = x*N.i + y*N.j + z*N.k
        X0 = x0*N.i + y0*N.j + z0*N.k
        xx0 = X - t*c - X0
        func = A*A* exp(-xx0.dot(xx0)/b**Rational(2))
        if self.solitonsFunc is None:
            self.solitonsFunc = func
        else:
            self.solitonsFunc += func
        return self.solitonsFunc

    def generateSolitonsModel(self,paramVec=None):
        '''Sustitute paramDict into symbolic function'''
        if paramVec is not None:
            self.solitonsParamDict = self.setSolitonsParams(paramVec)
        self.solitonsModel = self.solitonsFunc.subs(self.solitonsParamDict)
        return self.solitonsModel
        
class DiscreteModel(ElectronContentModel):
    def __init__(self,xvec,yvec,zvec,**kwargs):
        '''discrete model with tricubic interpolation'''
        super(DiscreteModel,self).__init__(**kwargs)
        self.xvec = xvec
        self.yvec = yvec
        self.zvec = zvec
        
        self.nx = np.size(xvec)
        self.ny = np.size(yvec)
        self.nz = np.size(zvec)
        
        self.ne = self.initialize()
        
        self.numParams = self.nx*self.ny*self.nz
        
        print("Generated discrete model with {0} params".format(self.numParams))
   
    def initialize(self,func=None):
        pass
            
        
    
    
def getSolitonCube(sol):
    #1000x1000x1000 km^3 cube centered around 500km above array
    c = ac.SkyCoord(0*au.km,0*au.km,500*au.km,frame=sol.enu).transform_to('itrs').cartesian.xyz.to(au.km).value
    xmin = c[0] - 500
    xmax = c[0] + 500
    ymin = c[1] - 500
    ymax = c[1] + 500
    zmin = c[2] - 500
    zmax = c[2] + 500
    return xmin,xmax,ymin,ymax,zmin,zmax
        
def plotSolitonModel(solitonsModel):
    func = solitonsModel.generateSolitonsModel()
    plotFuncCube(func,*getSolitonCube(solitonsModel),N=128,dx=None,dy=None,dz=None,rays=None)
    
if __name__=='__main__':
    plotModels()
    from ENUFrame import *
    iri = IriModel()
    #print(iri.iriFunc)
    print (iri.generateIri())
    xvec = [0]
    yvec = [0]
    zvec = np.linspace(0,3000,10000)
    X,Y,Z = np.meshgrid(xvec,yvec,zvec,indexing='ij')
    points = ac.SkyCoord(X.flatten()*au.km,Y.flatten()*au.km,Z.flatten()*au.km,frame=iri.enu).transform_to('itrs').cartesian.xyz.to(au.km).value
    
    #for x,y,z in zip(X.flatten(),Y.flatten(),Z.flatten()):
    #    points.append(ac.SkyCoord(x*au.km,y*au.km,y*au.km,frame=iri.enu).transform_to('itrs').cartesian.xyz.to(au.km).value)
    #points = np.array(points)
    xvec = np.linspace(np.min(points[0,:]),np.max(points[0,:]),len(xvec))
    yvec = np.linspace(np.min(points[1,:]),np.max(points[1,:]),len(yvec))
    zvec = np.linspace(np.min(points[2,:]),np.max(points[2,:]),len(zvec))
    X,Y,Z = np.meshgrid(xvec,yvec,zvec)
    #X = points[0,:].reshape(X.shape)
    #Y = points[1,:].reshape(Y.shape)
    #Z = points[2,:].reshape(Z.shape)
    #X,Y,Z=np.meshgrid(xvec,yvec,zvec,indexing='ij')
    ne = iri.evaluate(points[0,:],points[1,:],points[2,:])
    print (ne)
    import pylab as plt
    plt.plot(np.linspace(0,3000,10000),ne)
    plt.show()
    
    

