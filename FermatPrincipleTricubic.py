
# coding: utf-8

# In[ ]:

import numpy as np
from scipy.integrate import odeint

import numpy as np
from sympy import symbols,sqrt,sech,Rational,lambdify,Matrix,exp,cosh,cse,simplify,cos,sin
from sympy.vector import CoordSysCartesian

#from theano.scalar.basic_sympy import SymPyCCode
#from theano import function
#from theano.scalar import floats

from IRI import *
#from Symbolic import *
from scipy.integrate import simps
from ENUFrame import ENU

import astropy.coordinates as ac
import astropy.units as au
import astropy.time as at

from time import time as tictoc
from TricubicInterpolation import TriCubic

class Fermat(object):
    def __init__(self,neTCI=None,frequency = 120e6,type='s'):
        self.type = type
        self.frequency = frequency#Hz
        if neTCI is not None:
            self.ne2n(neTCI)          
            return
        
    def loadFunc(self,file):
        '''Load the model given in `file`'''
        data = np.load(file)
        if 'ne' in data.keys():
            ne = data['ne']
            xvec = data['xvec']
            yvec = data['yvec']
            zvec = data['zvec']
            self.ne2n(TriCubic(xvec,yvec,zvec,ne))
            return
        if 'n' in data.keys():
            ne = data['n']
            xvec = data['xvec']
            yvec = data['yvec']
            zvec = data['zvec']
            self.n2ne(TriCubic(xvec,yvec,zvec,n))
            return
    
    def saveFunc(self,file):
        np.savez(file,xvec=self.nTCI.xvec,yvec=self.nTCI.yvec,zvec=self.nTCI.zvec,n=self.nTCI.m,ne=self.neTCI.m)
            
    def ne2n(self,neTCI):
        '''Analytically turn electron density to refractive index. Assume ne in m^-3'''
        self.neTCI = neTCI
        #copy object
        self.nTCI = neTCI.copy(default=1.)
        #inplace change to refractive index
        self.nTCI.m *= -8.980**2/self.frequency**2
        self.nTCI.m += 1.
        self.nTCI.m = np.sqrt(self.nTCI.m)
        #wp = 5.63e4*np.sqrt(ne/1e6)/2pi#Hz^2 m^3 lightman p 226
        return self.nTCI
    
    def n2ne(self,nTCI):
        """Get electron density in m^-3 from refractive index"""
        self.nTCI = nTCI
        #convert to 
        self.neTCI = nTCI.copy()
        self.neTCI.m *= -self.neTCI.m
        self.neTCI.m += 1.
        self.neTCI.m *= self.frequency**2/8.980**2
        #wp = 5.63e4*np.sqrt(ne/1e6)/2pi#Hz^2 m^3 lightman p 226
        return self.neTCI
    
    def eulerODE(self,y,t,*args):
        '''return pxdot,pydot,pzdot,xdot,ydot,zdot,sdot'''
        #print(y)
        px,py,pz,x,y,z,s = y
        n,nx,ny,nz = self.nTCI.interp3(x,y,z)
        #print(n)
        #n,nx,ny,nz = 1.,0,0,0
        if (n>1):
            print(x,y,z,n)
        if self.type == 'z':
            sdot = n / pz
            pxdot = nx*n/pz
            pydot = ny*n/pz
            pzdot = nz*n/pz

            xdot = px / pz
            ydot = py / pz
            zdot = 1.
        
        if self.type == 's':
            sdot = 1.
            pxdot = nx
            pydot = ny
            pzdot = nz

            xdot = px / n
            ydot = py / n
            zdot = pz / n
        
        return [pxdot,pydot,pzdot,xdot,ydot,zdot,sdot]
    
    def jacODE(self,y,t,*args):
        '''return d ydot / d y, with derivatives down column for speed'''
        px,py,pz,x,y,z,s = y
        #n,nx,ny,nz,nxy,nxz,nyz = self.nTCI.interp3(x,y,z,doDouble=True)
        nxx,nyy,nzz = 0.,0.,0.
        #n,nx,ny,nz,nxy,nxz,nyz = 1.,0,0,0,0,0,0
        if (n>1):
            print(x,y,z,n)
        if self.type == 'z':
            x0 = n
            x1 = nx
            x2 = pz**(-2)
            x3 = x0*x2
            x4 = 1./pz
            x5 = ny
            x6 = x4*(x0*nxy + x1*x5)
            x7 = nz
            x8 = x4*(x0*nxz + x1*x7)
            x9 = x4*(x0*nyz + x5*x7)
            jac = np.array([[ 0,  0, -x1*x3, x4*(x0*nxx + x1**2),x6, x8, 0.],
                            [ 0,  0, -x3*x5,x6, x4*(x0*nyy + x5**2), x9, 0.],
                            [ 0,  0, -x3*x7,x8, x9, x4*(x0*nzz + x7**2), 0.],
                            [x4,  0, -px*x2, 0, 0,  0, 0.],
                            [ 0, x4, -py*x2, 0, 0, 0, 0.],
                            [ 0,  0, 0, 0, 0, 0, 0.],
                            [ 0,  0,-x3,x1*x4, x4*x5, x4*x7, 0.]])
        
        if self.type == 's':
            x0 = n
            x1 = nxy
            x2 = nxz
            x3 = nyz
            x4 = 1./x0
            x5 = nx
            x6 = x0**(-2)
            x7 = px*x6
            x8 = ny
            x9 = nz
            x10 = py*x6
            x11 = pz*x6
            jac = np.array([[ 0,  0,  0, nxx, x1, x2, 0.],
                            [ 0,  0,  0, x1, nyy, x3, 0.],
                            [ 0,  0,  0, x2, x3, nzz, 0.],
                            [x4,  0,  0, -x5*x7, -x7*x8, -x7*x9, 0.],
                            [ 0, x4,  0, -x10*x5, -x10*x8, -x10*x9, 0.],
                            [ 0,  0, x4, -x11*x5, -x11*x8, -x11*x9, 0.],
                            [ 0,  0,  0, 0, 0, 0, 0.]])
        return jac
        
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
        Y,info =  odeint(self.eulerODE, init, tarray, args=(time,),Dfun = self.jacODE, col_deriv = True, full_output=1)
        #print(info['hu'].shape,np.sum(info['hu']),info['hu'])
        #print(Y)
        x = Y[:,3]
        y = Y[:,4]
        z = Y[:,5]
        s = Y[:,6]
        return x,y,z,s   

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

def plotWavefront(neTCI,rays,save=False):
    xmin = neTCI.xvec[0]
    xmax = neTCI.xvec[-1]
    ymin = neTCI.yvec[0]
    ymax = neTCI.yvec[-1]
    zmin = neTCI.zvec[0]
    zmax = neTCI.zvec[-1]
    
    X,Y,Z = np.mgrid[xmin:xmax:len(neTCI.xvec)*1j,
                     ymin:ymax:len(neTCI.yvec)*1j,
                     zmin:zmax:len(neTCI.zvec)*1j]
    
    #reshape array
    data = neTCI.m.reshape([len(neTCI.xvec),len(neTCI.yvec),len(neTCI.zvec)])
    l = mlab.pipeline.volume(mlab.pipeline.scalar_field(X,Y,Z,data))#,vmin=min, vmax=min + .5*(max-min))
    l._volume_property.scalar_opacity_unit_distance = min((xmax-xmin)/4.,(ymax-ymin)/4.,(zmax-zmin)/4.)
    l._volume_property.shade = False
    #mlab.contour3d(X,Y,Z,data,contours=5,opacity=0.2)
    mlab.colorbar()
    
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
    
    if rays is not None:
        for ray in rays:
            mlab.plot3d(ray["x"],ray["y"],ray["z"],tube_radius=1.5)
        plt = mlab.points3d(*getWave(rays,0),color=(1,0,0),scale_mode='vector', scale_factor=10.)
        #mlab.move(-200,0,0)
        view = mlab.view()
        @mlab.animate(delay=100)
        def anim():
            nt = len(rays[0]["s"])
            f = mlab.gcf()
            save = False
            while True:
                i = 0
                while i < nt:
                    #print("updating scene")
                    xs,ys,zs = getWave(rays,i)
                    plt.mlab_source.set(x=xs,y=ys,z=zs)
                    #mlab.view(*view)
                    if save:
                        #mlab.view(*view)
                        mlab.savefig('figs/wavefronts/wavefront_{0:04d}.png'.format(i))#,magnification = 2)#size=(1920,1080))
                    #f.scene.render()
                    i += 1
                    yield
                save = False
        anim()
    mlab.show()
    if save:
        return
        import os
        os.system('ffmpeg -r 10 -f image2 -s 1900x1080 -i figs/wavefronts/wavefront_%04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p figs/wavefronts/wavefront.mp4')

def plotModel(neTCI,save=False):
    '''Plot the model contained in a tricubic interpolator (a convienient container for one)'''
    plotWavefront(neTCI,None,save=save)
    
def createPrioriModel(iri = None):
    if iri is None:
        iri = IriModel()
    eastVec = np.linspace(-200,200,40)
    northVec = np.linspace(-200,200,40)
    upVec = np.linspace(-10,3000,100)
    E,N,U = np.meshgrid(eastVec,northVec,upVec,indexing='ij')
    #get the points in ITRS frame
    #points = ac.SkyCoord(E.flatten()*au.km,N.flatten()*au.km,U.flatten()*au.km,frame=iri.enu).transform_to('itrs').cartesian.xyz.to(au.km).value
    
    ##generate cartesian grid in ITRS frame
    #Nx = int(np.ceil((np.max(points[0,:]) - np.min(points[0,:]))/30.))
    #Ny = int(np.ceil((np.max(points[1,:]) - np.min(points[1,:]))/30.))
    #Nz = int(np.ceil((np.max(points[2,:]) - np.min(points[2,:]))/30.))

    #xvec = np.linspace(np.min(points[0,:]),np.max(points[0,:]),Nx)
    #yvec = np.linspace(np.min(points[1,:]),np.max(points[1,:]),Ny)
    #zvec = np.linspace(np.min(points[2,:]),np.max(points[2,:]),Nz)
    #ij indexing - mgrid is that way by default
    #X,Y,Z = np.mgrid[np.min(points[0,:]):np.max(points[0,:]):1j*Nx,
    #                 np.min(points[1,:]):np.max(points[1,:]):1j*Ny,
    #                 np.min(points[2,:]):np.max(points[2,:]):1j*Nz]
    #X,Y,Z = np.meshgrid(xvec,yvec,zvec,indexing='ij')
    #Get values at points
    ne = iri.evaluate(E,N,U)
    print("created an a priori cube of shape: {0}".format(ne.shape))
    return eastVec,northVec,upVec,ne

    
def testSweep():
    '''Test the full system.'''
    # The priori ionosphere 
    iri = IriModel()
    eastVec,northVec,upVec,ne = createPrioriModel(iri)
    print("creating TCI object")
    neTCI = TriCubic(eastVec,northVec,upVec,ne)
    print("creating fermat object")
    f =  Fermat(neTCI = neTCI,type = 's')
    print(np.min(f.nTCI.m),np.max(f.nTCI.m))
    theta = np.linspace(-np.pi/15.,np.pi/15.,25)
    #phi = np.linspace(0,2*np.pi,6)
    rays = []
    origin = ac.ITRS(iri.enu.location).transform_to(iri.enu).cartesian.xyz.to(au.km).value
    t1 = tictoc()
    for t in theta:
        for p in theta:
            direction = ac.SkyCoord(np.sin(t),
                                    np.sin(p),
                                    1.,frame=iri.enu).cartesian.xyz.value#.transform_to('itrs').cartesian.xyz.value
            x,y,z,s = f.integrateRay(origin,direction,1000,time=0.)
            rays.append({'x':x,'y':y,'z':z,'s':s})
    print("time:",(tictoc()-t1)/len(rays))
    #print(rays)
    plotWavefront(neTCI,rays,save=False)
    #plotWavefront(f.nFunc.subs({'t':0}),rays,*getSolitonCube(sol),save = False)
    #plotFuncCube(f.nFunc.subs({'t':0}), *getSolitonCube(sol),rays=rays)

if __name__=='__main__':
    np.random.seed(1234)
    #testSquare()
    testSweep()
    #testThreadedFermat()
    #testSmoothify()
    #testcseLam()


# In[ ]:



