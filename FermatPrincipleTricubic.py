
# coding: utf-8

# In[7]:

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
        '''Load symbolic functions'''
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
        #convert to 
        self.nTCI = neTCI.copy()
        self.nTCI.m *= -8.980**2/self.frequency**2
        self.nTCI.m += 1.
        self.nTCI.m = np.sqrt(self.nTCI.m)
        print(self.neTCI.m)
        print(self.nTCI.m)
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
        n,nx,ny,nz,nxy,nxz,nyz = self.nTCI.interp3(x,y,z,doDouble = True)
        j = np.zeros([7,7])
        
        if self.type == 'z':
            #pxdot/px = 0
            #pxdot/py = 0
            #pxdot/pz
            j[2,0] = -nx*n/pz**2
            #pxdot/x
            j[3,0] = nx*nx/pz
            #pxdot/y
            j[4,0] = nxy*n/pz + nx*ny/pz
            #pxdot/z
            j[5,0] = nxz*n/pz + nx*nz/pz
            
            #py
            #pxdot/px = 0
            #pxdot/py = 0
            #pydot/pz
            j[2,1] = -ny*n/pz**2
            #pydot/x
            j[3,1] = nxy*n/pz + ny*nx/pz
            #pydot/y
            j[4,1] = ny*ny/pz
            #pxdot/z
            j[5,1] = nyz*n/pz + ny*nz/pz
            
            #pz
            #pzdot/px = 0
            #pzdot/py = 0
            #pzdot/pz
            j[2,2] = -nz*n/pz**2
            #pzdot/x
            j[3,2] = nxz*n/pz + nz*nx/pz
            #pzdot/y
            j[4,2] = nyz*n/pz + nz*ny/pz
            #pxdot/z
            j[5,2] = nz*nz/pz
            
            #xdot/px
            j[0,3] = 1./pz
            #xdot/py = 0
            #xdot/pz
            j[2,3] = -px/pz**2
            #ydot/px = 0
            #ydot/py
            j[1,4] = 1./pz
            #ydot/pz
            j[2,4] = -py/pz**2
            
            #zdot all = 0
            
            #sdot/pz
            j[2,6] = -n/pz**2
            #sdot/x
            j[3,6] = nx/pz
            j[4,6] = ny/pz
            j[5,6] = nz/pz
        
        if self.type == 's':
            #pxdot
            j[4,0] = nxy
            j[5,0] = nxz
            #pydot
            j[3,1] = nxy
            j[5,1] = nyz
            #pzdot
            j[3,2] = nxz
            j[4,2] = nyz
            #xdot
            j[0,3] = 1./n
            j[3,3] = -px/n**2 * nx
            j[4,3] = -px/n**2 * ny
            j[5,3] = -px/n**2 * nz
            #ydot
            j[1,4] = 1./n
            j[3,4] = -py/n**2 * nx
            j[4,4] = -py/n**2 * ny
            j[5,4] = -py/n**2 * nz
            #zdot
            j[2,5] = 1./n
            j[3,5] = -pz/n**2 * nx
            j[4,5] = -pz/n**2 * ny
            j[5,5] = -pz/n**2 * nz
            #sdot all = 0
            
        return j
        
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

def plotWavefront(neTCI,rays,N=128,dx=None,dy=None,dz=None,save=False):
    assert N>0,"resolution too small N = {0}".format(N)
    xmin = neTCI.xvec[0]
    xmax = neTCI.xvec[-1]
    ymin = neTCI.yvec[0]
    ymax = neTCI.yvec[-1]
    zmin = neTCI.zvec[0]
    zmax = neTCI.zvec[-1]
    
    if dx is None:
        dx = (xmax - xmin)/(N - 1)
    if dy is None:
        dy = (ymax - ymin)/(N-1)
    if dz is None:
        dz = (zmax - zmin)/(N-1)
    
    X,Y,Z = np.mgrid[xmin:xmax:len(neTCI.xvec)*1j,
                     ymin:ymax:len(neTCI.yvec)*1j,
                     zmin:zmax:len(neTCI.zvec)*1j]
    data = neTCI.m.reshape([len(neTCI.xvec),len(neTCI.yvec),len(neTCI.zvec)])
        
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
    

def testSweep():
    iri = IriModel()
    xvec = np.linspace(-100,100,10)
    yvec = np.linspace(-100,100,11)
    zvec = np.linspace(0,1500,50)
    
    x0,y0,z0 = iri.enu.location.geocentric
    xvec = np.linspace(x0.to(au.km).value-500,x0.to(au.km).value+500,50)
    yvec = np.linspace(y0.to(au.km).value - 500,y0.to(au.km).value + 500,50)
    zvec = np.linspace(z0.to(au.km).value-500,z0.to(au.km).value+500,50)
    X,Y,Z = np.meshgrid(xvec,yvec,zvec,indexing='ij')
    print(iri.enu.location.geodetic)
    points = ac.SkyCoord(X.flatten()*au.km,Y.flatten()*au.km,Z.flatten()*au.km,frame='itrs').transform_to(iri.enu).cartesian.xyz.value
    #points = []
    #for x,y,z in zip(X.flatten(),Y.flatten(),Z.flatten()):
    #    points.append(ac.SkyCoord(x*au.km,y*au.km,y*au.km,frame=iri.enu).transform_to('itrs').cartesian.xyz.to(au.km).value)
    #points = np.array(points)
    print(points)
    #X = points[0,:].reshape(X.shape)
    #Y = points[1,:].reshape(Y.shape)
    #Z = points[2,:].reshape(Z.shape)
    #X,Y,Z=np.meshgrid(xvec,yvec,zvec,indexing='ij')
    ne = iri.evaluate(X,Y,Z)
    neTCI = TriCubic(xvec,yvec,zvec,ne)
    
    f =  Fermat(neTCI = neTCI,type = 's')

    theta = np.linspace(-np.pi/8.,np.pi/8.,25)
    #phi = np.linspace(0,2*np.pi,6)
    rays = []
    origin = ac.ITRS(iri.enu.location).cartesian.xyz.to(au.km).value
    t1 = tictoc()
    for t in theta:
        for p in theta:
            direction = ac.SkyCoord(np.sin(t),
                                    np.sin(p),
                                    1.,frame=iri.enu).transform_to('itrs').cartesian.xyz.value
            x,y,z,s = f.integrateRay(origin,direction,1000,time=0.)
            rays.append({'x':x,'y':y,'z':z})
    print("time:",(tictoc()-t1)/len(rays))
    #print(rays)
    #plotWavefront(neTCI,rays,N=128,dx=None,dy=None,dz=None,save=False)
    #plotWavefront(f.nFunc.subs({'t':0}),rays,*getSolitonCube(sol),save = False)
    #plotFuncCube(f.nFunc.subs({'t':0}), *getSolitonCube(sol),rays=rays)

if __name__=='__main__':
    np.random.seed(1234)
    #testSquare()
    testSweep()
    #testThreadedFermat()
    #testSmoothify()
    #testcseLam()


# In[15]:




# In[ ]:



