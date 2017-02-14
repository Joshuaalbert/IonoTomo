
# coding: utf-8

# In[1]:

import numpy as np
from scipy.integrate import odeint

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

class Fermat(object):
    def __init__(self,nFunc=None,neFunc=None,frequency = 120e6, type = 'r'):
        self.frequency = frequency#Hz
        self.type = type
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
        
    def euler(self,pr,ptheta,pphi,r,theta,phi,s):
        N = np.size(pr)
        euler = np.zeros([7,N])
        i = 0
        while i < 7:
            euler[i,:] = self.eulerLambda[i](pr,ptheta,pphi,r,theta,phi,s)
            i += 1
        return euler
    
    def eulerODE(self,y,r):
        '''return prdot,pphidot,pthetadot,rdot,phidot,thetadot,sdot'''
        e = self.euler(y[0],y[1],y[2],y[3],y[4],z,y[5]).flatten()
        return e
    
    def jac(self,pr,ptheta,pphi,r,theta,phi,s):
        N = np.size(px)
        jac = np.zeros([7,7,N])
        i = 0
        while i < 7:
            j = 0
            while j < 7:
                jac[i,j,:] = self.jacLambda[i][j](pr,pphi,ptheta,r,phi,theta,s)
                j += 1
            i += 1
        return jac
    
    def jacODE(self,y,z):
        '''return d ydot / d y'''
        j = self.jac(y[0],y[1],y[2],y[3],y[4],z,y[5]).reshape([7,7])
        #print('J:',j)
        return j
        
    def generateEulerEqnsSym(self,nFunc=None):
        '''Generate function with call signature f(t,y,*args)
        and accompanying jacobian jac(t,y,*args), jac[i,j] = d f[i] / d y[j]'''
        
        if nFunc is None:
            nFunc = self.nFunc

        r,phi,theta,pr,pphi,ptheta,s = symbols('r phi theta pr pphi ptheta s')
        if self.type == 'r':
            sdot = sqrt(Rational(1) + (ptheta / r / pr)**Rational(2) + (pphi / r / sin(theta) / pr)**Rational(2))
            prdot = nFunc.diff('r')*sdot + nFunc/sdot * ((ptheta/r/pr)**Rational(2)/r + (pphi/r /sin(theta)/pr)**Rational(2)/r)
            pthetadot = nFunc.diff('theta')*sdot + nFunc/sdot * cos(theta) /sin(theta) *(pphi/r/sin(theta))**Rational(2)
            pphidot = nFunc.diff('phi')*sdot

            rdot = Rational(1)
            thetadot = ptheta/pr/r**Rational(2)
            phidot = pphi/pr/(r*sin(theta))**Rational(2)

            
        if self.type == 's':
            sdot = Rational(1)
            prdot = nFunc.diff('r') + (ptheta**Rational(2) + (pphi/sin(theta))**Rational(2))/nFunc/r**Rational(3)
            pthetadot = nFunc.diff('theta') + cos(theta) / sin(theta)**Rational(3) * (pphi/r)**Rational(2)/nFunc
            pphidot = nFunc.diff('phi')

            rdot = pr/nFunc
            thetadot = ptheta/nFunc/r**Rational(2)
            phidot = pphi/nFunc/(r*sin(theta))**Rational(2)

        eulerEqns = (prdot,pthetadot,pphidot,rdot,thetadot,phidot,sdot)

        euler = [lambdify((pr,ptheta,pphi,r,theta,phi,s),eqn,"numpy") for eqn in eulerEqns]
        self.eulerLambda = euler
        jac = []
        
        for eqn in eulerEqns:
            #print([eqn.diff(var) for var in (px,py,pz,x,y,z,s)])
            jac.append([lambdify((pr,ptheta,pphi,r,theta,phi,s),eqn.diff(var),"numpy") for var in (pr,ptheta,pphi,r,theta,phi,s)])
        self.jacLambda = jac
        
        return self.eulerLambda, self.jacLambda
    
    def integrateRay(self,x0,direction,tmax,N=100):
        '''Integrate rays from x0 in initial direction where coordinates are (r,theta,phi)'''
        direction /= np.linalg.norm(direction)
        r0,theta0,phi0 = x0
        rdot0,thetadot0,phi0 = direction
        sdot = np.sqrt(rdot**2 + r0**2 * (thetadot0**2 + np.sin(theta0)**2 * phidot0**2))
        pr0 = rdot/sdot
        ptheta0 = r0**2 * thetadot/sdot
        pphi0 = (r0 * np.sin(theta0))**2/sdot * phidot
        init = [pr0,ptheta0,pphi0,r0,theta0,phi0,0]
        if self.type == 'r':
            tarray = np.linspace(r0,tmax,N)
        if self.type == 's':
            tarray = np.linspace(0,tmax,N)
        #print("Integrating from {0} in direction {1} until {2}".format(x0,direction,tmax))
        Y,info =  odeint(self.eulerODE, init, tarray, Dfun = self.jacODE, col_deriv = 0, full_output=1)
        r = Y[:,3]
        theta = Y[:,4]
        phi = Y[:,5]
        s = Y[:,6]
        return r,theta,phi,s
    

def testSweep():
    import pylab as plt
    x,y,z = symbols('x y z')
    sol = SolitonModel(4)
    sol.generateSolitonModel()
    neFunc = sol.solitonModel
    
    f =  Fermat(neFunc = neFunc)
    n = f.nFunc

    theta = np.linspace(-np.pi/4.,np.pi/4.,5)
    rays = []
    for t in theta:
        origin = ac.SkyCoord(0*au.km,0*au.km,0*au.km,frame=sol.enu).transform_to('itrs').cartesian.xyz.to(au.km).value
        direction = ac.SkyCoord(np.cos(t+np.pi/2.),0,np.sin(t+np.pi/2.),frame=sol.enu).transform_to('itrs').cartesian.xyz.value
        x,y,z,s = integrateRay(origin,direction,f,origin[2],7000)
        rays.append({'x':x,'y':y,'z':z})
        #plt.plot(x,z)
    plotFuncCube(n.subs({'t':0}), *getSolitonCube(sol),rays=rays)
    #plt.show()


def testSmoothify():
    octTree = OctTree([0,0,500],dx=100,dy=100,dz=1000)
    octTree = subDivide(octTree)
    octTree = subDivide(octTree)
    s = SmoothVoxel(octTree)
    model = s.smoothifyOctTree()
    plotCube(model ,-50.,50.,-50.,50.,0.,1000.,N=128,dx=None,dy=None,dz=None)
    
if __name__=='__main__':
    #testSquare()
    testSweep()
    #testSmoothify()
    #testcseLam()


# In[ ]:



