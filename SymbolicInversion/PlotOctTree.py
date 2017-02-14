
# coding: utf-8

# In[2]:

from mayavi.sources.api import VTKDataSource
from mayavi import mlab
from Geometry import *

from scipy.interpolate import griddata

def getCrossLines(octTree):
    if octTree.hasChildren:
        xmin = octTree.centroid
        xmin[0] -= octTree.dx
        xmax = octTree.centroid
        xmax[0] += octTree.dx
        ymin = octTree.centroid
        ymin[0] -= octTree.dy
        ymax = octTree.centroid
        ymax[0] += octTree.dy
        zmin = octTree.centroid
        zmin[0] -= octTree.dz
        zmax = octTree.centroid
        zmax[0] += octTree.dz
        lines = [np.vstack((xmin,xmax)),np.vstack((ymin,ymax)),np.vstack((zmin,zmax))]
        for child in octTree.children:
            lines = lines + getCrossLines(child)
        return lines
    else:
        return []

def plotOctTree(octTree):
    '''Do a density plot'''

    #from mayavi.sources.api import VTKDataSource
    #from mayavi import mlab

    #from scipy.interpolate import griddata

    #plot grid lines and rays     
    print("Plotting:",octTree)
    lines = getCrossLines(octTree)
    for line in lines:
        mlab.plot3d(line[:,0],line[:,1],line[:,2],color=(1,1,1))
        #for key in vox.lineSegments.keys():
        #    p1 = vox.lineSegments[key].origin
        #    p2 = vox.lineSegments[key].eval(vox.lineSegments[key].sep)
        #    ax.plot([p1[0],p2[0]],[p1[2],p2[2]],ls='-')'
        #       p12 = np.vstack((p1,p2))
        #       mlab.plot3d(p12[:,0],p12[:,1],p12[:,2],color=(0,0,0))
    
    mlab.axes()
    mlab.show()
    
def mayaviPlot(x,m,mBackground=None,maxNumPts=None,octTree=None):
    '''Do a density plot'''

    #from mayavi.sources.api import VTKDataSource
    #from mayavi import mlab

    #from scipy.interpolate import griddata

    xmin,ymin,zmin = np.min(x[:,0]),np.min(x[:,1]),np.min(x[:,2])
    xmax,ymax,zmax = np.max(x[:,0]),np.max(x[:,1]),np.max(x[:,2])
    X,Y,Z = np.mgrid[xmin:xmax:128j,ymin:ymax:128j,zmin:zmax:128j]
    
    if mBackground is not None:
        data  = m - mBackground
    else:
         data = m
    #plot grid lines and rays     
    voxels = getAllDecendants(octTree)
    for vox in voxels:
        #plot S plane (2)
        for plane in vox.boundingPlanes:
            for edge in plane.edges:
                p1 = edge.origin
                p2 = edge.eval(edge.sep)
                p12 = np.vstack((p1,p2))
                mlab.plot3d(p12[:,0],p12[:,1],p12[:,2],color=(1,1,1))
        #for key in vox.lineSegments.keys():
        #    p1 = vox.lineSegments[key].origin
        #    p2 = vox.lineSegments[key].eval(vox.lineSegments[key].sep)
        #    ax.plot([p1[0],p2[0]],[p1[2],p2[2]],ls='-')'
        #       p12 = np.vstack((p1,p2))
        #       mlab.plot3d(p12[:,0],p12[:,1],p12[:,2],color=(0,0,0))
    
    field = griddata((x[:,0],x[:,1],x[:,2]),data,(X.flatten(),Y.flatten(),Z.flatten()),method='linear').reshape(X.shape)
    
    #mlab.points3d(x[:,0],x[:,1],x[:,2],data,scale_mode='vector', scale_factor=10.)
    mlab.contour3d(X,Y,Z,field,contours=3,opacity=0.2)
    
    min = np.min(data)
    max = np.max(data)
    l = mlab.pipeline.volume(mlab.pipeline.scalar_field(X,Y,Z,field),vmin=min, vmax=min + .5*(max-min))
    l._volume_property.scalar_opacity_unit_distance = (xmax-xmin)/2.
    l._volume_property.shade = False
    mlab.colorbar()
    
    mlab.axes()
    mlab.show()
    
def fun():
    import numpy as np
    import pylab as plt
    d = np.genfromtxt('exampleIRI.txt',names=True)
    extra = d['ne'] + np.mean(d['ne'])*np.exp(-(d['height']-600.)**2/(50**2))
    plt.plot(d['height'],d['ne'],c='black',label='IRI')
    plt.plot(d['height'],extra,ls='--',c='red',label='perturbation')
    plt.legend(frameon=False)
    plt.xlabel('Height (km)')
    plt.ylabel(r'Electron density $n_e$ (${\rm m}^{-3}$)')
    plt.yscale('log')
    plt.grid()
    plt.title('International Reference Ionosphere')
    plt.show()
#fun()


# In[ ]:



