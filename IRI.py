
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

def symbolicChapman_def1f2(h = None,cond=0):
    if h is None:
        h = symbols('h')
    if cond == 0:
        p = 1
        w = 1
    if cond == 1:
        p = 5
        w = 1.5
    if cond == 2:
        p = 10
        w = 2.
    #f = symbolicChapman(h,Rational(1e10),Rational(450),Rational(250))
    f2 = symbolicChapman(h,Rational(p*2.2e11),Rational(359),Rational(w*46))
    f1 = symbolicChapman(h,Rational(3e9),Rational(195),Rational(20))
    e = symbolicChapman(h,Rational(3e9),Rational(90),Rational(10))
    #d = symbolicChapman(h,Rational(3e9),Rational(80),Rational(50))
    return f2 + f1 + e  + Rational(4.5e8)
    

def ExampleIRI():
    d = np.genfromtxt('exampleIRI.txt',names=True)
    profile = d['ne']
    return d['height'],d['ne']
    
def plotModels():
    '''Plot various models'''
    import pylab as plt
    h = np.linspace(0,3000,10000)
    ### from IRI
    h1,ne1 = ExampleIRI()
    plt.plot(h1,ne1,c='black',label='Ex. IRI')
    #neQuick layers
    ne3 = xef1f2(h1)
    plt.plot(h1,ne3,c='blue',label='neQuick_ef1f2')
    lam = lambdify(symbols('h'),symbolicChapman_def1f2(),'numpy')
    ne4 = lam(h)
    plt.plot(h,ne4,c='green',label='Chapman_def1f2')
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
        

        
def IntegrateChapman(Npeak,hmax,width):
    y0 = -hmax/width
    ymax = 10#(h+3*w-h)/w
    from scipy.special import erf
    return -Npeak*np.sqrt(2*np.exp(1)*np.pi)*(erf(np.exp(-ymax/2)/np.sqrt(2)) - erf(np.exp(-y0/2)/np.sqrt(2)))
    
def IntegrateChapmanNumerical(Npeak,hmax,width,var=1e10):
    h = np.linspace(0,3000,10000)
    lam = lambdify(symbols('h'),symbolicChapman(symbols('h'), Npeak,hmax,width),'numpy')
    ne = lam(h)
    #import pylab as plt
    #plt.plot(h,ne)
    #plt.show()
    from scipy.integrate import simps
    res = []
    for i in range(100):
        res.append(simps(ne+np.random.normal(loc=0,scale=var,size=h.size),h))
    #import pylab as plt
    #plt.hist(res)
    #plt.show()
    return np.mean(res)

def R2upperbound(frequency=100e6,cond=0):
    n_p = 1.24e-2 * frequency**2
    h = np.linspace(0,3000,10000)
    lam = lambdify(symbols('h'),symbolicChapman_def1f2(h = symbols('h'),cond=cond),'numpy')
    if cond == 0:
        var = 1e10
    if cond == 1:
        var = 1e11
    if cond == 2:
        var = 1e12
    ne = lam(h)
    #import pylab as plt
    #plt.plot(h,ne)
    #plt.show()
    from scipy.integrate import simps
    res = []
    alpha = (ne + var)/n_p
    num = alpha**3
    integrand = num/((1-alpha)**(5./2.))
    print("freq:",frequency,"cond:",cond,"alphahat:{0:1.1e}".format(np.mean(alpha)),"sigma_alpha^2:{0:1.1e}".format(np.mean(alpha**2)-np.mean(alpha)**2))
    return "{0:1.1e}".format(simps(integrand,h)*n_p/8./1e16)

def fitExampleIonospheres(folder):
    import glob
    files = glob.glob("{0}/*".format(folder))
    
    # 0 - DOY
    # 1 - Hour
    # 2 - Solar_zenith_angle, degree
    # 3 - Height, km
    # 4 - Electron_density_Ne, m-3
    # 5 - hmF2, km
    # 6 - hmF1, km
    # 7 - hmE, km
    # 8 - hmD, km
    # 9 - NmF2, m-3
    # 10 - NmF1, m-3
    # 11 - NmE, m-3
    # 12 - NmD, m-3
    # 13 - E_valley_width, km
    d = []
    for f in files:
        d.append(np.genfromtxt(f))
    d = np.vstack(d)
    
    mask = d[:,10] == d[:,10]
    doy = d[mask,0]
    hr = d[mask,1]
    chi = d[mask,2]
    nf2 = d[mask,9]
    nf1 = d[mask,10]
    ne = d[mask,11]
    nd = d[mask,12]
    hf2 = d[mask,5]
    hf1 = d[mask,6]
    he = d[mask,7]
    hd = d[mask,8]
    Z = d[mask,3]
    N = d[mask,4]
    
    
    
    from scipy.optimize import curve_fit
    import pylab as plt
    

    def gauss(x, *p):
        hr = x[0,:]
        doy = x[1,:]
        chi = x[2,:]
        A, mu, sigma, C = p
        return A*np.exp(-(hr-mu)**2/(2.*sigma**2)) + C
    
    def peakDensity(chi,A,C,tau,b):
        return np.abs(C) + (np.abs(A)) *np.exp(-(chi/np.abs(tau))**2)/(1 +  (chi/tau)**(2*np.abs(b)))
    
    def peakDensityFit(chi,*p):
        A,C,tau,b = p
        return peakDensity(chi,A,C,tau,b)
    
    def layerDensity(z,nm,zm,H):
        y = (z-zm)/np.abs(H)
        return np.abs(nm) * np.exp(0.5*(1-y-np.exp(-y)))
    

    def peakHeightFit(pred,*p):
        #doy = pred[0,:]
        #hr = pred[1,:]
        chi = pred[:]
        C,D,ACHI,BCHI = p
        return C + (D)/(1 + np.exp(-(chi-ACHI)/BCHI)) #np.exp((doy - ADOY)**2/(2*BDOY**2)) * np.arctan(BCHI*(chi-ACHI)) * (AHR + BHR*hr)
    
    
    
    p0=(1e8,1e8,45,4)
    coeffnd, var_matrix = curve_fit(peakDensityFit, chi, nd, p0=p0)
    print("nd:",coeffnd)
    plt.scatter(chi,nd,c='blue')
    plt.scatter(chi,peakDensityFit(chi,*coeffnd),c='red')
    plt.show()
    p0=(1e9,1e9,45,4)
    coeffne, var_matrix = curve_fit(peakDensityFit, chi, ne, p0=p0)
    print("ne:",coeffne)
    plt.scatter(chi,ne,c='blue')
    plt.scatter(chi,peakDensityFit(chi,*coeffne),c='red')
    plt.show()
    p0=(1e12,1e11,45,4)
    coeffnf2, var_matrix = curve_fit(peakDensityFit, chi, nf2, p0=p0)
    print("nf2:",coeffnf2)
    plt.scatter(chi,nf2,c='blue')
    plt.scatter(chi,peakDensityFit(chi,*coeffnf2),c='red')
    plt.show()
    p0=(1e12,1e11,45,4)
    mask = hf1 != -1
    coeffnf1, var_matrix = curve_fit(peakDensityFit, chi[mask], nf1[mask], p0=p0)
    print("nf1:",coeffnf1)
    plt.scatter(chi[mask],nf1[mask],c='blue')
    plt.scatter(chi,peakDensityFit(chi,*coeffnf1),c='red')
    plt.show()

    pred = chi
    p0 = (np.min(hd),np.max(hd),100,1)
    plt.scatter(chi,hd)
    plt.scatter(chi,peakHeightFit(pred,*p0),color='red')
    plt.show()
    
    
    coeffhd, var_matrix = curve_fit(peakHeightFit, pred, hd, p0=p0)
    print(coeffhd)
    plt.scatter(doy,hd)
    plt.scatter(doy,peakHeightFit(pred,*coeffhd),color='red')
    plt.show()
    
    p0 = (np.min(he),np.max(he),100,1)
    coeffhe, var_matrix = curve_fit(peakHeightFit, pred, he, p0=p0)
    print(coeffhe)
    plt.scatter(doy,he)
    plt.scatter(doy,peakHeightFit(pred,*coeffhe),color='red')
    plt.show()
    
    p0 = (np.min(hf2),np.max(hf1),100,1)
    coeffhf2, var_matrix = curve_fit(peakHeightFit, pred, hf2, p0=p0)
    print(coeffhf2)
    plt.scatter(doy,hf2)
    plt.scatter(doy,peakHeightFit(pred,*coeffhf2),color='red')
    plt.show()
    
    mask = hf1 != -1
    pred = chi[mask]
    p0 = (np.min(hf1[mask]),np.max(hf1[mask]),100,1)
    coeffhf1, var_matrix = curve_fit(peakHeightFit, pred, hf1[mask], p0=p0)
    print(coeffhf1)
    plt.scatter(doy[mask],hf1[mask])
    plt.scatter(doy[mask],peakHeightFit(pred,*coeffhf1),color='red')
    plt.show()
    

    def DEFDensity(pred,*p):
        chi = pred[0,:]
        z = pred[1,:]
        Hd,He,Hf2,Hf1 = p
        nmd = peakDensityFit(chi,*coeffnd)
        nme = peakDensityFit(chi,*coeffne)
        nmf2 = peakDensityFit(chi,*coeffnf2)
        nmf1 = peakDensityFit(chi,*coeffnf1)
        zmd = peakHeightFit(chi,*coeffhd)
        zme = peakHeightFit(chi,*coeffhe)
        zmf2 = peakHeightFit(chi,*coeffhf2)
        zmf1 = peakHeightFit(chi,*coeffhf1)
        
        return layerDensity(z,nmd,zmd,Hd)+layerDensity(z,nme,zme,He)+layerDensity(z,nmf2,zmf2,Hf2)+layerDensity(z,nmf1,zmf1,Hf1)
        
    pred = np.vstack([chi,Z])
    # p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
    #Hd,He,Hf,Ad,Cd,taud,bd,Ae,Ce,taue,be,Af,Cf,tauf,bf
    p0 = [  8,11,55,40]
    
    coeffH, var_matrix = curve_fit(DEFDensity, pred, N, p0=p0)
    print("H:",coeffH)
    plt.scatter(chi,N)
    plt.scatter(chi,DEFDensity(pred,*coeffH),color='red')
    plt.show()
    plt.scatter(Z,N)
    plt.scatter(Z,DEFDensity(pred,*coeffH),color='red')
    plt.show()
    
if __name__=='__main__':
    fitExampleIonospheres("IriData")
    #print( IntegrateChapman(2.2e12,350,50))
    #print( IntegrateChapmanNumerical(2.2e12,350,50))
    print(R2upperbound(frequency=25e6,cond=0))
    print(R2upperbound(frequency=50e6,cond=0))
    print(R2upperbound(frequency=100e6,cond=0))
    print(R2upperbound(frequency=150e6,cond=0))
    print(R2upperbound(frequency=200e6,cond=0))
    
    print(R2upperbound(frequency=25e6,cond=1))
    print(R2upperbound(frequency=50e6,cond=1))
    print(R2upperbound(frequency=100e6,cond=1))
    print(R2upperbound(frequency=150e6,cond=1))
    print(R2upperbound(frequency=200e6,cond=1))
    
    print(R2upperbound(frequency=25e6,cond=2))
    print(R2upperbound(frequency=50e6,cond=2))
    print(R2upperbound(frequency=100e6,cond=2))
    print(R2upperbound(frequency=150e6,cond=2))
    print(R2upperbound(frequency=200e6,cond=2))
    #print( IntegrateChapman(2.2e12,350,100))
    
    #plotModels()
    #from ENUFrame import *
    #iri = IriModel()
    #print(iri.iriFunc)
    #print (iri.generateIri())
    #xvec = [0]
    #yvec = [0]
    #zvec = np.linspace(0,3000,10000)
    #X,Y,Z = np.meshgrid(xvec,yvec,zvec,indexing='ij')
    #points = ac.SkyCoord(X.flatten()*au.km,Y.flatten()*au.km,Z.flatten()*au.km,frame=iri.enu).transform_to('itrs').cartesian.xyz.to(au.km).value
    
    #for x,y,z in zip(X.flatten(),Y.flatten(),Z.flatten()):
    #    points.append(ac.SkyCoord(x*au.km,y*au.km,y*au.km,frame=iri.enu).transform_to('itrs').cartesian.xyz.to(au.km).value)
    #points = np.array(points)
    #xvec = np.linspace(np.min(points[0,:]),np.max(points[0,:]),len(xvec))
    #yvec = np.linspace(np.min(points[1,:]),np.max(points[1,:]),len(yvec))
    #zvec = np.linspace(np.min(points[2,:]),np.max(points[2,:]),len(zvec))
    #X,Y,Z = np.meshgrid(xvec,yvec,zvec)
    #X = points[0,:].reshape(X.shape)
    #Y = points[1,:].reshape(Y.shape)
    #Z = points[2,:].reshape(Z.shape)
    #X,Y,Z=np.meshgrid(xvec,yvec,zvec,indexing='ij')
    #ne = iri.evaluate(points[0,:],points[1,:],points[2,:])
    #print (ne)
    #import pylab as plt
    #plt.plot(np.linspace(0,3000,10000),ne)
    #plt.show()
    
    


# In[ ]:




# In[ ]:



