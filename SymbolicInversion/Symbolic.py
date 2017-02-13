
# coding: utf-8

# In[4]:

import numpy as np
from sympy import symbols,sqrt,sech,Rational,lambdify,Matrix,exp,cosh,cse,simplify,cos,sin
from sympy.vector import CoordSysCartesian

from mayavi.sources.api import VTKDataSource
from mayavi import mlab

from Geometry import *
from time import time as tictoc

def cseLambdify(params,func):
    '''performs common sub expression elimination before compiling
    NOT WORKING YET
    '''
    repl, redu = cse(func,optimizations='basic')
    #print(repl,redu)
    cseLamb = []
    syms = list(params)
    for se in repl:
        cseLamb.append(lambdify(syms,se[1],modules=['numpy',{'sech': lambda x: 2./(np.exp(x) + np.exp(-x))}],dummify=False))
        syms.append(se[0])
    cseLamb.append( lambdify(syms,redu[0],modules = ['numpy',{'sech': lambda x: 2./(np.exp(x) + np.exp(-x))}],dummify=False))
    return cseLamb
    
def cseLambEval(params,cseLamb):
    funs = list(params)
    for fun in cseLamb:
        funs.append(fun(*funs))
    return funs[-1]

def plotFuncCube(func,xmin,xmax,ymin,ymax,zmin,zmax,N=128,dx=None,dy=None,dz=None,rays=None):
    '''Plot a scalar function within the specified region.
    If rays is a list of dicts [{"x":x,"y":y,"z":z}] then plot them'''
    #from mayavi.sources.api import VTKDataSource
    #from mayavi import mlab

    assert N>0,"resolution too small N = {0}".format(N)
    
    if dx is None:
        dx = (xmax - xmin)/(N - 1)
    if dy is None:
        dy = (ymax - ymin)/(N-1)
    if dz is None:
        dz = (zmax - zmin)/(N-1)
    
    X,Y,Z = np.mgrid[xmin:xmax:N*1j,
                     ymin:ymax:N*1j,
                     zmin:zmax:N*1j]
    
    x,y,z,t = symbols('x y z t')
    
    #funcLamb = lambdify((x,y,z),func,"numpy")
    funcLamb = cseLambdify((x,y,z,t),func)
    #print(func)
    data = np.zeros(np.size(X))
    #data[:] = funcLamb(X.flatten(),Y.flatten(),Z.flatten())
    data[:] = cseLambEval((X.flatten(),Y.flatten(),Z.flatten(),0.),funcLamb)
    data = data.reshape(X.shape)
    
    #mlab.points3d(X.flatten(),Y.flatten(),Z.flatten(),data,scale_mode='vector', scale_factor=10.)
    logmins = np.log10(np.min(data))
    logmaxs = np.log10(np.max(data))
    #contours = 10**np.linspace(logmins + (logmaxs - logmins)*0.5,logmins + (logmaxs - logmins)*0.95,5)
    mlab.contour3d(X,Y,Z,data,contours=5,opacity=0.2)
    if rays is not None:
        for ray in rays:
            mlab.plot3d(ray["x"],ray["y"],ray["z"],tube_radius=1.5)
    
    #l = mlab.pipeline.volume(mlab.pipeline.scalar_field(X,Y,Z,data))#,vmin=min, vmax=min + .5*(max-min))
    #l._volume_property.scalar_opacity_unit_distance = min((xmax-xmin)/4.,(ymax-ymin)/4.,(zmax-zmin)/4.)
    #l._volume_property.shade = False
    mlab.colorbar()
    
    mlab.axes()
    mlab.show()
    
def plotWavefront(func,rays,xmin,xmax,ymin,ymax,zmin,zmax,N=128,dx=None,dy=None,dz=None,save=False):
    assert N>0,"resolution too small N = {0}".format(N)
    
    if dx is None:
        dx = (xmax - xmin)/(N - 1)
    if dy is None:
        dy = (ymax - ymin)/(N-1)
    if dz is None:
        dz = (zmax - zmin)/(N-1)
    
    X,Y,Z = np.mgrid[xmin:xmax:N*1j,
                     ymin:ymax:N*1j,
                     zmin:zmax:N*1j]
    
    x,y,z = symbols('x y z')
    
    #funcLamb = lambdify((x,y,z),func,"numpy")
    funcLamb = cseLambdify((x,y,z),func)
    #print(func)
    data = np.zeros(np.size(X))
    #data[:] = funcLamb(X.flatten(),Y.flatten(),Z.flatten())
    data[:] = cseLambEval((X.flatten(),Y.flatten(),Z.flatten()),funcLamb)
    data = data.reshape(X.shape)
        
    def getWave(rays,idx):
        xs = np.zeros(len(rays))
        ys = np.zeros(len(rays))
        zs = np.zeros(len(rays))
        ridx = 0
        while ridx < len(rays):
            xs[ridx] = rays[ridx]['x'][idx]
            ys[ridx] = rays[ridx]['y'][idx]
            zs[ridx] = rays[ridx]['z'][idx]
            ridx += 1
        return xs,ys,zs
    
    nt = np.size(rays[0]['x'])
    #mlab.contour3d(X,Y,Z,data,contours=5,opacity=0.2)
    l = mlab.pipeline.volume(mlab.pipeline.scalar_field(X,Y,Z,data))#,vmin=min, vmax=min + .5*(max-min))
    l._volume_property.scalar_opacity_unit_distance = min((xmax-xmin)/4.,(ymax-ymin)/4.,(zmax-zmin)/4.)
    l._volume_property.shade = False
    for ray in rays:
        mlab.plot3d(ray["x"],ray["y"],ray["z"],tube_radius=1.5)
    mlab.colorbar()
    #mlab.points3d(0,0,0,scale_mode='vector', scale_factor=10.)
    plt = mlab.points3d(*getWave(rays,0),color=(1,0,0),scale_mode='vector', scale_factor=10.)
    mlab.move(-200,0,0)
    view = mlab.view()
    @mlab.animate(delay=100)
    def anim():
        f = mlab.gcf()
        save = True
        while True:
            i = 0
            while i < nt:
                #print("updating scene")
                xs,ys,zs = getWave(rays,i)
                plt.mlab_source.set(x=xs,y=ys,z=zs)
                #mlab.view(*view)
                if save:
                    mlab.savefig('figs/wavefronts/wavefront_{0:04d}.png'.format(i),magnification = 2)#size=(1920,1080))
                #f.scene.render()
                i += 1
                yield
            save = False
    anim()
    mlab.show()
    if save:
        pass
        import os
        os.system('ffmpeg -r 10 -f image2 -s 1900x1080 -i figs/wavefronts/wavefront_%04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p figs/wavefronts/wavefront.mp4')
    
class SmoothVoxel(object):
    """A smooth voxel is something that tried to emulate a cube in space with a smooth function.
    Too slow by far.
    """
    def __init__(self,octTree=None):
        '''define a '''
        self.octTree = octTree #representation in octTree
    def getVoxelAmp(self,voxelId):
        return "a{0}".format(voxelId)
        
    def makeVoxel(self,center,dx,dy,dz,voxelId):
        # cube part
        n = 2
        x,y,z = symbols('x y z')
        a = symbols(self.getVoxelAmp(voxelId))
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
            model = model - thisVox.subs({self.getVoxelAmp(voxelId):amp})
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
            model = model + thisVox.subs({self.getVoxelAmp(voxelId):amp})
            voxelId += 1
        return model
    
def makeVoxel(center,dx,dy,dz,voxelId):
    # cube part
    n = 2
    x,y,z = symbols('x y z')
    a = symbols('a_{0}'.format(voxelId))
    offx = (x - center[0])/dx
    offy = (y - center[1])/dy
    offz = (z - center[2])/dz
    voxel = ((Rational(1)/(Rational(1) + (Rational(2)*offx) ** Rational(2*n)) + Rational(1)/(Rational(1) + (Rational(2)*offy) ** Rational(2*n)) + Rational(1)/(Rational(1) + (Rational(2)*offz) ** Rational(2*n)))/Rational(3))**(Rational(n)) * a * exp(- offx**Rational(2) - offy**Rational(2) - offz**Rational(2))
    return voxel

def testVoxelPerformance(N=1000):
    x = np.random.uniform(size=N)
    y = np.random.uniform(size=N)
    z = np.random.uniform(size=N)
    func = Rational(0)
    t1 = tictoc()
    i = 0
    while i < N:
        func += makeVoxel([x[i],y[i],z[i]],0.1,0.1,0.1,i)
        i += 1
    t2 = tictoc()
    print("time to form: {0} s".format(t2-t1))
    return func
    
    
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
    print(cseLambEval((1,2,3),lam2))
    
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
    
if __name__=='__main__':
    func = testVoxelPerformance(N=1000)
    a = {}
    i = 0
    while i < 1000:
        a['a_{0}'.format(voxelId)] = np.random.uniform()
        i += 1
    

