import logging as log
import numpy as np
import os
import pylab as plt

__MAYAVI__ = False
try:
    os.environ["QT_API"] = "pyqt"
    from mayavi import mlab
    __MAYAVI__ = True
except:
    try:
        os.environ["QT_API"] = "pyside"
        from mayavi import mlab
        __MAYAVI__ = True
    except:
        log.info("Unable to import mayavi")

if not __MAYAVI__:
    log.info("No mayavi, therefore exitting")
    #exit(1)

def plot_tci(tci,rays=None,filename=None,show=False):
    '''Plot the given tci using mayavi if possible.
    tci : TriCubic object to plot
    rays : array of shape (num_antennas, num_times, num_dirs, 4, num_steps)
    filename : name of figure file to save to without extension e.g. "figure1"
    show : boolean, whether to show the resulting figure.'''
    xmin = tci.xvec[0]
    xmax = tci.xvec[-1]
    ymin = tci.yvec[0]
    ymax = tci.yvec[-1]
    zmin = tci.zvec[0]
    zmax = tci.zvec[-1]
    f = mlab.figure(bgcolor=(0,0,0), size=(800, 800))
    #visual.set_viewer(f)

    X,Y,Z = np.mgrid[xmin:xmax:len(tci.xvec)*1j,
                     ymin:ymax:len(tci.yvec)*1j,
                     zmin:zmax:len(tci.zvec)*1j]
       
    #reshape array
    data = tci.M
    plt.plot(tci.zvec,np.max(np.max(data,axis=0),axis=0))
    plt.savefig("{}_profile.png".format(filename))
    plt.close()
    vmin = np.percentile(data.flatten(),5)
    vmax = np.percentile(data.flatten(),65)
    src = mlab.pipeline.scalar_field(X,Y,Z,data)
    l = mlab.pipeline.volume(src,vmin=vmin, vmax=vmax)
    l.volume_property.scalar_opacity_unit_distance = min((xmax-xmin)/4.,(ymax-ymin)/4.,(zmax-zmin)/4.)
    l.volume_property.shade = False
#    voi = mlab.pipeline.extract_grid(src)
#    voi.set(x_min=tci.nx>>1, x_max=tci.nx, y_min=tci.ny>>1, y_max=tci.ny, z_min=0, z_max=tci.nz)
#    mlab.pipeline.iso_surface(voi, contours=[1e9,1e10])#np.percentile(data.flatten(),50), np.percentile(data.flatten(),65)], colormap='cool')
#
    #mlab.contour3d(X,Y,Z,data,contours=10,opacity=0.2)
    mlab.colorbar()
    
    if rays is not None:
        #[Na, Nt, Nd, 4, N]
        i = 0
        while i < rays.shape[0]:
            j = 0
            k = 0
            while k < rays.shape[2]:
                x,y,z = rays[i,0,k,0,:],rays[i,0,k,1,:],rays[i,0,k,2,:]
                mlab.plot3d(x,y,z,tube_radius=0.75)
                k += 1
            i += 1
    if filename is not None:
        mlab.savefig('{}.png'.format(filename))#,magnification = 2)#size=(1920,1080))
        log.info("Saved to {}.png".format(filename))
    if show:
        mlab.show()
    mlab.close(all=True)


