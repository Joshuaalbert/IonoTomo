
# coding: utf-8

# In[1]:

import numpy as np
from scipy.integrate import odeint

import numpy as np
from sympy import symbols,sqrt,sech,Rational,lambdify,Matrix,exp,cosh,cse,simplify,cos,sin
from sympy.vector import CoordSysCartesian

#from theano.scalar.basic_sympy import SymPyCCode
#from theano import function
#from theano.scalar import floats

from IRI import *
from Symbolic import *
from scipy.integrate import simps
from ENUFrame import ENU

import astropy.coordinates as ac
import astropy.units as au
import astropy.time as at

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
        
    def loadFunc(self,file):
        '''Load symbolic functions'''
        data = np.load(file)
        if 'neFunc' in data.keys():
            self.neFunc = data['neFunc']
            self.nFunc = self.ne2n(self.neFunc)
            self.eulerLambda, self.jacLambda = self.generateEulerEqnsSym(self.nFunc)
            return
        if 'nFunc' in data.keys():
            self.nFunc = data['nFunc']
            self.neFunc = self.n2ne(self.nFunc)
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
    
def calcForwardKernel(kernelRays,file):
    '''ray is a dictionary {datumIdx:{'x':ndarray,'y':ndarray,'z':ndarray,'s':ndarray}}
    kernel is a symbolic function of (x,y,z,t) symbols
    time is a double'''
    results = {}
    
    data = np.load(file)
    kernel = data['kernel'].item(0)
    Jkernel = data['Jkernel']
    #print(kernel)
    GLamb = lambdify(symbols('x y z t'),kernel,'numpy')
    #for each thing do kernel forward problem
    for rayPair in kernelRays:
        datumIdx = rayPair[0].id
        time = rayPair[0].time
        ray = rayPair[1]
        gres = GLamb(ray['x'],ray['y'],ray['z'],time)
        if np.size(gres) == 1:
            g = gres*(ray['s'][-1] - ray['s'][0])
        else:
            g = simps(gres,ray['s'])
        results[datumIdx] = {'g':g,'J':np.zeros(len(Jkernel))}
    paramIdx = 0
    while paramIdx < len(Jkernel):
        JLamb = lambdify(symbols('x y z t'),Jkernel[paramIdx],'numpy')
        for rayPair in kernelRays:
            datumIdx = rayPair[0].id
            time = rayPair[0].time
            ray = rayPair[1]
            jres = JLamb(ray['x'],ray['y'],ray['z'],time)
            if np.size(jres) == 1:
                results[datumIdx]['J'][paramIdx] = jres*(ray['s'][-1] - ray['s'][0])
            else:
                results[datumIdx]['J'][paramIdx] = simps(jres,ray['s'])
        paramIdx += 1
    return results

def generateKernel(gFile,forwardKernelParamDict):
    data = np.load(gFile)
    #print data
    Gk = data['Gk'].item(0)
    Jk = data['Jk']
    G = Gk.subs(forwardKernelParamDict)
    J = []
    for j in Jk:
        J.append(j.subs(forwardKernelParamDict))
    return {'G':G,'J':J}

def testSweep():
    sol = SolitonModel(1)
    neFunc = sol.generateSolitonsModel()
    
    f =  Fermat(neFunc = neFunc,type = 's')

    theta = np.linspace(-np.pi/8.,np.pi/8.,25)
    #phi = np.linspace(0,2*np.pi,6)
    rays = []
    origin = ac.ITRS(sol.enu.location).cartesian.xyz.to(au.km).value
    for t in theta:
        for p in theta:
            direction = ac.SkyCoord(np.sin(t),
                                    np.sin(p),
                                    1.,frame=sol.enu).transform_to('itrs').cartesian.xyz.value
            x,y,z,s = f.integrateRay(origin,direction,1000,time=0.)
            rays.append({'x':x,'y':y,'z':z})
    plotWavefront(f.nFunc.subs({'t':0}),rays,*getSolitonCube(sol),save = False)
    #plotFuncCube(f.nFunc.subs({'t':0}), *getSolitonCube(sol),rays=rays)

if __name__=='__main__':
    np.random.seed(1234)
    #testSquare()
    testSweep()
    #testThreadedFermat()
    #testSmoothify()
    #testcseLam()


# In[ ]:



