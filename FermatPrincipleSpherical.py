
# coding: utf-8

# In[ ]:

import numpy as np
from scipy.integrate import odeint

import numpy as np
from sympy import symbols,sqrt,sech,Rational,lambdify,Matrix,exp,cosh,cse,simplify,cos,sin
from sympy.vector import CoordSysCartesian

from theano.scalar.basic_sympy import SymPyCCode
from theano import function
from theano.scalar import floats

from Geometry import *

def plotCube(func,xmin,xmax,ymin,ymax,zmin,zmax,N=128,dx=None,dy=None,dz=None,rays=None):
    '''Plot a scalar function within the specified region.
    If rays is a list of dicts [{"x":x,"y":y,"z":z}] then plot them'''
    from mayavi.sources.api import VTKDataSource
    from mayavi import mlab

    assert N>0,"resolution too small N = {0}".format(N)
    
    if dx is None:
        dx = (xmax - xmin)/(N - 1)
    if dy is None:
        dy = (ymax - ymin)/(N-1)
    if dz is None:
        dz = (zmax - zmin)/(N-1)
    
    #xvec = np.linspace(xmin, xmax, int((xmin - xmax)/dx))
    #yvec = np.linspace(ymin, ymax, int(ymin - ymax)/dy)
    #zvec = np.linspace(zmin, zmax, int(zmin - zmax)/dz)
    
    #X,Y,Z = np.mgrid[xmin:xmax:int((xmin - xmax)/dx)*1j,
    #                 ymin:ymax:int((ymin - ymax)/dy)*1j,
    #                 zmin:zmax:int((zmin - zmax)/dz)*1j]
    
    X,Y,Z = np.mgrid[xmin:xmax:N*1j,
                     ymin:ymax:N*1j,
                     zmin:zmax:N*1j]
    
    x,y,z = symbols('x y z')
    
    funcLamb = lambdify((x,y,z),func,"numpy")
    
    data = np.zeros(np.size(X))
    data[:] = funcLamb(X.flatten(),Y.flatten(),Z.flatten())
    data = data.reshape(X.shape)
    
    #mlab.points3d(X.flatten(),Y.flatten(),Z.flatten(),data,scale_mode='vector', scale_factor=10.)
    mlab.contour3d(X,Y,Z,data,contours=5,opacity=0.2)
    if rays is not None:
        for ray in rays:
            mlab.plot3d(ray["x"],ray["y"],ray["z"],tube_radius=0.3)
    
    min = np.min(data)
    max = np.max(data)
    l = mlab.pipeline.volume(mlab.pipeline.scalar_field(X,Y,Z,data),vmin=min, vmax=min + .5*(max-min))
    l._volume_property.scalar_opacity_unit_distance = (xmax-xmin)/2.
    l._volume_property.shade = False
    mlab.colorbar()
    
    mlab.axes()
    mlab.show()
    
class SmoothVoxel(object):
    """a smooth voxel is something that tried to emulate a cube in space with a smooth function."""
    def __init__(self,octTree=None):
        '''define a '''
        self.octTree = octTree #representation in octTree
    def makeVoxel(self,center,dx,dy,dz,voxelId):
        # cube part
        n = 2
        x,y,z = symbols('x y z')
        a = symbols("a{0}".format(voxelId))
        offx = (x - center[0])/dx
        offy = (y - center[1])/dy
        offz = (z - center[2])/dz
        voxel = Rational(1)/(Rational(1) + (Rational(2)*offx) ** Rational(2*n))
        voxel = voxel + Rational(1)/(Rational(1) + (Rational(2)*offy) ** Rational(2*n))
        voxel = voxel + Rational(1)/(Rational(1) + (Rational(2)*offz) ** Rational(2*n))
        voxel = voxel/Rational(3)
        voxel = voxel**(Rational(n))
        voxel = a*voxel * exp(- offx**Rational(2) - offy**Rational(2) - offz**Rational(2))
        return voxel
    def smoothifyOctTree_n(self,octTree=None):
        if octTree is None:
            octTree = self.octTree
        model = Rational(1)
        voxels = getAllDecendants(octTree)
        #print(voxels)
        voxelId = 0
        for vox in voxels:
            amp = 1 - vox.properties['n'][1]
            thisVox = self.makeVoxel(vox.centroid,vox.dx, vox.dy, vox.dz, voxelId)
            model = model - thisVox.subs({"a{0}".format(voxelId):amp})
            voxelId += 1
        return model
    def smoothifyOctTree_ne(self,octTree=None):
        if octTree is None:
            octTree = self.octTree
        model = Rational(0)
        voxels = getAllDecendants(octTree)
        #print(voxels)
        voxelId = 0
        for vox in voxels:
            amp = vox.properties['ne'][1]
            thisVox = self.makeVoxel(vox.centroid,vox.dx, vox.dy, vox.dz, voxelId)
            model = model + thisVox.subs({"a{0}".format(voxelId):amp})
            voxelId += 1
        return model
            
    
class Fermat(object):
    def __init__(self,nFunc,frequency = 120e6):
        self.frequency = frequency#Hz
        self.nFunc = nFunc
        self.eulerLambda, self.jacLambda = self.generateEulerEqnsSym(self.nFunc)
    
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
        
    def euler(self,px,py,pz,x,y,z,s):
        N = np.size(px)
        euler = np.zeros([7,N])
        i = 0
        while i < 7:
            euler[i,:] = self.eulerLambda[i](px,py,pz,x,y,z,s)
            i += 1
        return euler
    
    def eulerODE(self,y,z):
        '''return pxdot,pydot,pzdot,xdot,ydot,zdot,sdot'''
        e = self.euler(y[0],y[1],y[2],y[3],y[4],z,y[5]).flatten()
        #print(e)
        return e
    
    def jac(self,px,py,pz,x,y,z,s):
        N = np.size(px)
        jac = np.zeros([7,7,N])
        i = 0
        while i < 7:
            j = 0
            while j < 7:
                jac[i,j,:] = self.jacLambda[i][j](px,py,pz,x,y,z,s)
                j += 1
            i += 1
        return jac
    
    def jacODE(self,y,z):
        '''return d ydot / d y'''
        j = self.jac(y[0],y[1],y[2],y[3],y[4],z,y[5]).reshape([7,7])
        #print('J:',j)
        return j
        
    def generateEulerEqnsSym(self,nFunc):
        '''Generate function with call signature f(t,y,*args)
        and accompanying jacobian jac(t,y,*args), jac[i,j] = d f[i] / d y[j]'''

        x,y,z,px,py,pz,s = symbols('x y z px py pz s')

        pxdot = nFunc.diff('x')*nFunc/pz
        pydot = nFunc.diff('y')*nFunc/pz
        pzdot = nFunc.diff('z')*nFunc/pz

        xdot = px / pz
        ydot = py / pz
        zdot = Rational(1)

        sdot = nFunc / pz

        eulerEqns = (pxdot,pydot,pzdot,xdot,ydot,zdot,sdot)
        #print(cse(Matrix(eulerEqns),optimizations='basic'))
        #print(cse(Matrix(eulerEqns)))
        euler = [lambdify((px,py,pz,x,y,z,s),eqn,"numpy") for eqn in eulerEqns]
        self.eulerLambda = euler
        jac = []
        
        for eqn in eulerEqns:
            #print([eqn.diff(var) for var in (px,py,pz,x,y,z,s)])
            jac.append([lambdify((px,py,pz,x,y,z,s),eqn.diff(var),"numpy") for var in (px,py,pz,x,y,z,s)])
        self.jacLambda = jac
        
        return self.eulerLambda, self.jacLambda
    
def integrateRay(x0,cosines,fermat,zinit,zmax):
    cosines /= np.linalg.norm(cosines)
    f = np.sqrt(1. + cosines[0]**2 + cosines[1]**2)
    px = cosines[0]/f
    py = cosines[1]/f
    pz = cosines[2]/f
    zarray = np.linspace(zinit,zmax,100)
    #print("Integrating from {0} in direction {1} until {2}".format(x0,cosines,zmax))
    init = [px,py,pz,x0[0],x0[1],zinit,0] # px,py,pz,x,y,s
    Y,info =  odeint(fermat.eulerODE, init, zarray, Dfun = fermat.jacODE, col_deriv = 0, full_output=1)
    x = Y[:,3]
    y = Y[:,4]
    z = Y[:,5]
    s = Y[:,6]
    return x,y,z,s
    
def cseLambdify(params,func):
    '''performs common sub expression elimination before compiling
    NOT WORKING YET
    '''
    repl, redu = cse(func,optimizations='basic')
    print(repl,redu)
    cseLamb = []
    syms = list(params)
    for se in repl:
        cseLamb.append(lambdify(syms,se[1],modules=['numpy'],dummify=False))
        syms.append(se[0])
    cseLamb.append( lambdify(syms,redu[0],modules = ['numpy'],dummify=False))
    return cseLamb
    
def cseLambEval(params,cseLamb):
    funs = list(params)
    for fun in cseLamb:
        funs.append(fun(*funs))
    return funs[-1]
        
        
def testcseLam():
    x,y,z = symbols('x y z')
    
    nFunc = Rational(1)
    for i in range(10):
        x0 = np.random.uniform(low=-50,high=50)
        y0 = np.random.uniform(low=-50,high=50)
        z0 = np.random.uniform(low=10,high=100)
        nFunc = nFunc - np.random.uniform(low=0.05,high=0.15)*exp(-((x-x0)**2 + (y-y0)**2 + (z-z0)**2)/50)
    func = cos(x*y/z) + cos((x+y)/z)**2 + cos(x-z)**3 + sin(cos(x)+sin(x))**4 + sin(x)*y - cos(3*x)/x + sin(x-z)**2
    lam1 = lambdify((x,y,z),func,'numpy')
    lam2 = cseLambdify((x,y,z),func)
    get_ipython().magic(u'timeit lam1(1,2,3)')
    print(cseLambEval((1,2,3),lam2))
    
def testSweep():
    import pylab as plt
    x,y,z = symbols('x y z')
    
    nFunc = Rational(1)
    for i in range(10):
        x0 = np.random.uniform(low=-50,high=50)
        y0 = np.random.uniform(low=-50,high=50)
        z0 = np.random.uniform(low=10,high=100)
        nFunc = nFunc - np.random.uniform(low=0.05,high=0.15)*exp(-((x-x0)**2 + (y-y0)**2 + (z-z0)**2)/50)
    
    f =  Fermat(nFunc)
    ne = f.n2ne(nFunc)

    theta = np.linspace(-np.pi/4.,np.pi/4.,5)
    rays = []
    for t in theta:
        x,y,z,s = integrateRay(np.array([0,0,0]),np.array([np.cos(t+np.pi/2.),0,np.sin(t+np.pi/2.)]),f,0,100)
        rays.append({'x':x,'y':y,'z':z})
        #plt.plot(x,z)
    plotCube(f.n2ne(nFunc),-50.,50.,-50.,50.,0.,100.,N=128,dx=None,dy=None,dz=None,rays=rays)
    #plt.show()
def testSquare():
    import pylab as plt
    x = np.linspace(-1.,1.,1000)
    X,Y = np.meshgrid(x,x)
    
    n = 2
    x0 = 0.25
    y0 = 0.75
    f1 = ((1./(1 + (2*((X-x0))/0.5)**(2*n)) + 1./(1 + (2*((Y-y0))/0.5)**(2*n)))/2.)**n * np.exp(-(X-x0)**2/0.5**2 - (Y-y0)**2/0.5**2)
    f2 = ((1./(1 + (2*((X-0.25))/0.5)**(2*n)) + 1./(1 + (2*((Y-0.25))/0.5)**(2*n)))/2.)**n * np.exp(-(X-0.25)**2/0.5**2 - (Y-0.25)**2/0.5**2)    
    plt.imshow(f1+0.9*f2)
    plt.colorbar()
    plt.show()

def testSmoothify():
    octTree = OctTree([0,0,500],dx=100,dy=100,dz=1000)
    octTree = subDivide(octTree)
    octTree = subDivide(octTree)
    s = SmoothVoxel(octTree)
    model = s.smoothifyOctTree()
    plotCube(model ,-50.,50.,-50.,50.,0.,1000.,N=128,dx=None,dy=None,dz=None)
    
#testSquare()
#testSweep()
#testSmoothify()
#testcseLam()

x,y,z = symbols('x')
    
nFunc = Rational(1)
for i in range(10):
    x0 = np.random.uniform(low=-50,high=50)
    y0 = np.random.uniform(low=-50,high=50)
    z0 = np.random.uniform(low=10,high=100)
    nFunc = nFunc - np.random.uniform(low=0.05,high=0.15)*exp(-((x-x0)**2 + (y-y0)**2 + (z-z0)**2)/50)
func = cos(x)
func += sin(x*func)
func /=exp(x*func)
func += x*func.diff(x)
func *= func.diff('x')
#func = exp(cos(x*y/z) + cos((x+y)/z)**2 + cos(x-sin(z))**3 + sin(cos(x)+sin(x))**4 + sin(x)*y - cos(3*x)/x + sin(x-z)**2)
lam1 = lambdify((x),func,'numpy')
lam2 = cseLambdify((x),func)
get_ipython().magic(u'timeit lam1(1)')
get_ipython().magic(u'timeit cseLambEval((1),lam2)')

