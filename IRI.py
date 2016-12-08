
# coding: utf-8

# In[1]:

import numpy as np
from sympy import Rational,symbols,exp,lambdify,sqrt
from RadioArray import RadioArray
from ENUFrame import ENU

def neQuick(h,No=2.2e11,hmax=368.,Ho=50.):
    '''Ne quick model'''
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
    f2 = neQuick(h1,No=2.2e11,hmax=359.,Ho=46.)
    f1 = neQuick(h1,No=3e9,hmax=195.,Ho=20.)
    e = neQuick(h1,No=3e9,hmax=90.,Ho=10.)
    return f2 + f1 + e

def symbolicNeQuick(h, No,hmax,Ho):
    g = Rational(1,7)
    rfac = Rational(100)
    #h = symbols('h')
    dh = h - hmax
    g1 = g * dh
    z = dh / (Ho * (Rational(1) + rfac * g1 / (rfac * Ho + g1)))
    ee = exp(z)
    return No*Rational(4)*ee/(Rational(1) + ee)**Rational(2)
 
def symbolicXef1f2(h = None):
    if h is None:
        h = symbols('h')
    f2 = symbolicNeQuick(h,No=Rational(2.2e11),hmax=Rational(359),Ho=Rational(46))
    f1 = symbolicNeQuick(h,No=Rational(3e9),hmax=Rational(195),Ho=Rational(20))
    e = symbolicNeQuick(h,No=Rational(3e9),hmax=Rational(90),Ho=Rational(10))
    return f2 + f1 + e

def ionosphereModel(h):
    Nf1 = 10*4*np.exp((h-350)/50.)/(1 + np.exp((h-350)/50.))**2
    res = Nf1    
    Ne = 0.3*4*np.exp((h-85.)/50.)/(1 + np.exp((h-85.)/50.))**2
    res += Ne
    return res*5e10
    
def ExampleIRI():
    d = np.genfromtxt('exampleIRI.txt',names=True)
    profile = d['ne']
    return d['height'],d['ne']
    
def plotModels():
    lam = lambdify(symbols('h'),symbolicXef1f2(),'numpy')
    
    import pylab as plt
    h1,ne1 = ExampleIRI()
    plt.plot(h1,ne1,c='black',label='IRI')
    ne2 = ionosphereModel(h1)
    plt.plot(h1,ne2,c='red',label='simple')
    ne3 = neQuick(h1,No=2.2e11,hmax=359.,Ho=46.) + neQuick(h1,No=3e9,hmax=195.,Ho=20.)+ neQuick(h1,No=3e9,hmax=90.,Ho=10.)
    plt.plot(h1,ne3,c='blue',label='neQuick')
    ne4 = lam(h1)
    plt.plot(h1,ne4,c='green',label='neQuick_symbol')
    plt.legend(frameon=False)
    plt.xlabel('Height (km)')
    plt.ylabel(r'Electron density $n_e$ (${\rm m}^{-3}$)')
    plt.yscale('log')
    plt.grid()
    plt.title('Ionosphere Models')
    plt.show()

class ElectronContentModel(object):
    def __init__(self,radioArray = None,**kwargs):
        if radioArray is None:
            radioArray = RadioArray(arrayFile='arrays/lofar.hba.antenna.cfg')
        self.radioArray = radioArray
        self.enu = ENU(location=radioArray.getCenter().earth_location)
    def getCartesianModel(self):
        '''Cartesian model, suitable for use with ITRF frame'''
        pass
    def getSphericalModel(self):
        '''Spherical model, lon,lat,r_geocentric '''
        pass
    
class IRIModel(ElectronContentModel):
    def __init__(self,**kwargs):
        '''The ionosphere is modelled as the IRI plus a set of solitons
        with time coherence imposed via linear coherence over short time intervals'''
        super(IRIModel,self).__init__(**kwargs)
        self.generateIRI()
    def generateIRI(self):
        '''x,y,z are ITRF coords'''
        x,y,z = symbols('x y z')
        R = Rational(6371)
        R = self.radioArray.getCenter().height
        r = sqrt(x**(Rational(2))+y**(Rational(2))+z**(Rational(2)))
        h = r - R
        self.iri = symbolicXef1f2(h)
        return self.iri
        
    
class SolitonModel(IRIModel):
    def __init__(self,numSolitons=0,**kwargs):
        '''The ionosphere is modelled as the IRI plus a set of solitons
        with time coherence imposed via linear coherence over short time intervals'''
        super(SolitonModel,self).__init__(**kwargs)
        self.paramIdx = 0
        self.initParams = []
        self.solitons = []
        i = 0
        while i < numSolitons:
            self.addSoliton()
            i += 1
    def getParamName(self):
        #assert type(idx)==type(1),"{0} is not an integer".format(idx)
        param = "p{0}".format(self.paramIdx)
        self.paramIdx += 1
        return param
    def addSolitonvdK(self):
        '''Soliton consists of a function of the form
        A * f(x-ct) where f(x) is
        1/2 * |c| sech^2 (1/2 * sqrt(|c|) |x - x0|^2)
        sech(x) is 2 / exp(x) + exp(-x)
        
        In 3D this gives A: 1, c: 3, x0: 3 -> 7 params per soliton
        '''
        A = symbols (self.getParamName())
        self.initParams.append(0.)
        cx = symbols (self.getParamName())
        self.initParams.append(0.)
        cy = symbols (self.getParamName())
        self.initParams.append(0.)
        cz = symbols (self.getParamName())
        self.initParams.append(0.)
        x0 = symbols (self.getParamName())
        self.initParams.append(0.)
        y0 = symbols (self.getParamName())
        self.initParams.append(0.)
        z0 = symbols (self.getParamName())
        self.initParams.append(0.)
        
        soliton = {"A":A,"c":[cx,cy,cz],"x":[x0,y0,z0],
                   "init":{A.name:1e11,
                           cx.name:1./10.,
                           cy.name:1./10.,
                           cz.name:1./100.,
                           x0.name:0.,
                           y0.name:0.,
                           z0.name:6500.}}


        x,y,z,t = symbols('x,y,z,t')
        
        N = CoordSysCartesian('N')
        c = cx*N.i + cy*N.j + cz*N.k
        cabs = sqrt(c.dot(c))
        
        X = x*N.i + y*N.j + z*N.k
        X0 = x0*N.i + y0*N.j + z0*N.k
        xx0 = X - t*c - X0
        soliton['func'] = A * sech(Rational(1,2) * sqrt(cabs) * xx0.dot(xx0))**Rational(2)
        self.solitons.append(soliton)
        return soliton['func']
    def addSoliton(self):
        '''Soliton consists of a function of the form
        A * exp(-(x-c * t))
        
        In 3D this gives A: 1, c: 3, x0: 3 -> 7 params per soliton
        '''
        A = symbols (self.getParamName())
        self.initParams.append(0.)
        cx = symbols (self.getParamName())
        self.initParams.append(0.)
        cy = symbols (self.getParamName())
        self.initParams.append(0.)
        cz = symbols (self.getParamName())
        self.initParams.append(0.)
        x0 = symbols (self.getParamName())
        self.initParams.append(0.)
        y0 = symbols (self.getParamName())
        self.initParams.append(0.)
        z0 = symbols (self.getParamName())
        self.initParams.append(0.)
        b = symbols (self.getParamName())
        self.initParams.append(0.)
        
        init = self.radioArray.getCenter().transform_to(self.enu)
        init.data[2] += 350000.
        init = init.transform_to('itrs')
        
        soliton = {"A":A,"c":[cx,cy,cz],"x":[x0,y0,z0],
                   "init":{A.name:1e11,
                           cx.name:0.,
                           cy.name:0.,
                           cz.name:0.,
                           x0.name:0.,
                           y0.name:0.,
                           z0.name:init.cartesian.xyz.value[2]/1000.,
                           b.name:50.
                          }}

        x,y,z,t = symbols('x,y,z,t')
        
        N = CoordSysCartesian('N')
        c = cx*N.i + cy*N.j + cz*N.k
        cabs = sqrt(c.dot(c))
        
        X = x*N.i + y*N.j + z*N.k
        X0 = x0*N.i + y0*N.j + z0*N.k
        xx0 = X - t*c - X0
        soliton['func'] = A * exp(-xx0.dot(xx0)/b**Rational(2))
        self.solitons.append(soliton)
        return soliton['func']
    def generateSolitonModel(self):
        x,y,z,t = symbols('x,y,z,t')
        res = self.iri
        for soliton in self.solitons:
            sol = soliton['func']
            sol = sol.subs(soliton['init'])
            res = res + sol
        self.solitonModel = res
        return self.solitonModel
        
def plotSoliton():
    def sech(x):
        return 2./(np.exp(x) + np.exp(-x))
    def sol(x,t,c,a):
        cabs = np.abs(c)
        return a*sech(np.sqrt(cabs)/2. * (x-c*t))**2
    import pylab as plt
    x = np.linspace(-10,10,1000)
    t = np.linspace(0,5,4)
    for ti in t:
        plt.plot(x,sol(x,ti,1/10.,1))
    plt.show()
        
    
if __name__=='__main__':
    #plotSoliton()
    from Symbolic import *
    iri = IRIModel()
    #print(cseLambdify(symbols('x y z'),iri.generateIRI()))
    sol = SolitonModel(1)
    sol.generateSolitonModel()
    plotFuncCube(sol.solitonModel.subs({'t':0}), xmin = -50, xmax = 50, ymin = -50, ymax = 50, zmin = 6371, zmax = 7000)
    #plotModels()
    


# In[ ]:



