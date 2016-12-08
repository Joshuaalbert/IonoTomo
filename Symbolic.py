
# coding: utf-8

# In[3]:

import numpy as np
from sympy import symbols,sqrt,sech,Rational,lambdify,Matrix,exp,cosh,cse,simplify,cos,sin
from sympy.vector import CoordSysCartesian

from mayavi.sources.api import VTKDataSource
from mayavi import mlab

from Geometry import *

def cseLambdify(params,func):
    '''performs common sub expression elimination before compiling
    NOT WORKING YET
    '''
    repl, redu = cse(func,optimizations='basic')
    print(repl,redu)
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
    
    x,y,z = symbols('x y z')
    
    #funcLamb = lambdify((x,y,z),func,"numpy")
    funcLamb = cseLambdify((x,y,z),func)
    #print(func)
    data = np.zeros(np.size(X))
    #data[:] = funcLamb(X.flatten(),Y.flatten(),Z.flatten())
    data[:] = cseLambEval((X.flatten(),Y.flatten(),Z.flatten()),funcLamb)
    data = data.reshape(X.shape)
    
    #mlab.points3d(X.flatten(),Y.flatten(),Z.flatten(),data,scale_mode='vector', scale_factor=10.)
    mlab.contour3d(X,Y,Z,data,contours=5,opacity=0.2)
    if rays is not None:
        for ray in rays:
            mlab.plot3d(ray["x"],ray["y"],ray["z"],tube_radius=0.3)
    
    l = mlab.pipeline.volume(mlab.pipeline.scalar_field(X,Y,Z,data))#,vmin=min, vmax=min + .5*(max-min))
    l._volume_property.scalar_opacity_unit_distance = min((xmax-xmin)/4.,(ymax-ymin)/4.,(zmax-zmin)/4.)
    l._volume_property.shade = False
    mlab.colorbar()
    
    mlab.axes()
    mlab.show()
    
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

