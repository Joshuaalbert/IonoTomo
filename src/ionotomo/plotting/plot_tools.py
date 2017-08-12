import os

#__MAYAVI__ = False
#try:
#    os.environ["QT_API"] = "pyqt"
#    from mayavi import mlab
#    __MAYAVI__ = True
#except:
#    try:
#        os.environ["QT_API"] = "pyside"
#        from mayavi import mlab
#        __MAYAVI__ = True
#    except:
#        print("Unable to import mayavi")

from ionotomo.geometry.tri_cubic import TriCubic
from ionotomo.astro.frames.uvw_frame import UVW
import numpy as np
import pylab as plt
import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au
## utility functions

def interp_nearest(x,y,z,x_,y_):
    dx = np.subtract.outer(x_,x)
    dy = np.subtract.outer(y_,y)
    r = dx**2
    dy *= dy
    r += dy
    np.sqrt(r,out=r)
    arg = np.argmin(r,axis=1)
    z_ = z[arg]
    return z_


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
    
    X,Y,Z = np.mgrid[xmin:xmax:len(tci.xvec)*1j,
                     ymin:ymax:len(tci.yvec)*1j,
                     zmin:zmax:len(tci.zvec)*1j]
    
    #reshape array
    data = tci.get_shaped_array()
    xy = np.mean(data,axis=2)
    yz = np.mean(data,axis=0)
    zx = np.mean(data,axis=1)
    fig,(ax1,ax2,ax3) = plt.subplots(1,3)
    ax1.imshow(xy,origin='lower',aspect='auto')
    ax1.set_title("X-Y projection")
    ax2.imshow(yz,origin='lower',aspect='auto')
    ax2.set_title("Y-Z projection")
    ax3.imshow(zx,origin='lower',aspect='auto')
    ax3.set_title("Z-X projection")
    if filename is not None:
        plt.savefig("{}.png".format(filename),format='png')
    if show:
        plt.show()
    else:
        plt.close()
        
def make_animation(datafolder,prefix='fig',fps=3):
    '''Given a datafolder with figures of format `prefix`-%04d.png create a 
    video at framerate `fps`.
    Output is datafolder/animation.mp4'''
    if os.system('ffmpeg -framerate {} -i {}/{}-%04d.png -vf scale="trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -profile:v high -pix_fmt yuv420p -g 30 -r 30 {}/animation.mp4'.format(fps,datafolder,prefix,datafolder)):
        print("{}/animation.mp4 exists already".format(datafolder))


def animate_tci_slices(TCI,output_folder,num_seconds=10.):
    '''Animate the slicing of a tci by showing the xz, yz, zy planes as they
    sweep across the volume (possibly depreciated)'''
    try:
        os.makedirs(output_folder)
    except:
        pass
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(12,12))
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    M = TCI.get_shaped_array()
    if np.sum(M<0) > 0:
        print("Using linear scaling")
        log_spacing = False
    else:
        print("Using log scaling")
        log_spacing = True
        M[M==0] = np.min(M[M>0])
    levels = [np.min(M),np.max(M)]
    for q in np.linspace(1,99,15*5+2):
        if log_spacing:
            l = 10**np.percentile(np.log10(M),q)
            if l not in levels and len(levels) < 15:
                levels.append(l)
        else:
            l = np.percentile(M,q)
            if l not in levels  and len(levels) < 15:
                levels.append(l) 
    levels = np.sort(levels)
    #N = max(1,int((len(levels)-2)/13))
    #levels = [levels[0]] + levels[1:-1][::N] + [levels[-1]]
    print("plotting levels : {}".format(levels))
    #M[M<levels[0]] = np.nan
    #M[M>levels[-1]] = np.nan
    vmin = np.min(M)
    vmax = np.max(M)
    Y_1,X_1 = np.meshgrid(TCI.yvec,TCI.xvec,indexing='ij')
    Z_2,Y_2 = np.meshgrid(TCI.zvec,TCI.yvec,indexing='ij')
    Z_3,X_3 = np.meshgrid(TCI.zvec,TCI.xvec,indexing='ij')
    i = 0
    while i < TCI.nz:
        xy = M[:,:,i].transpose()#x by y
        j1 = int(i/float(TCI.nz)*TCI.nx)
        #j1 = TCI.nx >> 1
        yz = M[j1,:,:].transpose()#y by z
        j2 = (TCI.ny - 1) - int(i/float(TCI.nz)*TCI.ny)
        #j2 = TCI.ny >> 1
        xz = M[:,j2,:].transpose()#x by z
        im = ax2.imshow(xy,origin='lower',vmin=vmin,vmax=vmax,aspect = 'auto',
                        extent=[TCI.xvec[0],TCI.xvec[-1],TCI.yvec[0],TCI.yvec[-1]],cmap=plt.cm.bone)
        CS = ax2.contour(xy, levels,
                     origin='lower',
                     linewidths=2,
                     extent=[TCI.xvec[0],TCI.xvec[-1],TCI.yvec[0],TCI.yvec[-1]],cmap=plt.cm.hot_r)
        zc = CS.collections[-1]
        plt.setp(zc, linewidth=4)
        plt.clabel(CS, levels[1::2],  # label every second level
                   inline=1,
                   fmt='%.2g',
                   fontsize=14)
        ax2.set_title("Height: {:.2g} km".format(TCI.zvec[i]))
        ax2.set_xlabel('X km')
        ax2.set_ylabel('Y km')
        
        im = ax3.imshow(yz,origin='lower',vmin=vmin,vmax=vmax,aspect = 'auto',
                        extent=[TCI.yvec[0],TCI.yvec[-1],TCI.zvec[0],TCI.zvec[-1]],cmap=plt.cm.bone)
        CS = ax3.contour(yz, levels,
                     origin='lower',
                     linewidths=2,
                     extent=[TCI.yvec[0],TCI.yvec[-1],TCI.zvec[0],TCI.zvec[-1]],cmap=plt.cm.hot_r)
        zc = CS.collections[-1]
        plt.setp(zc, linewidth=4)
        plt.clabel(CS, levels[1::2],  # label every second level
                   inline=1,
                   fmt='%.2g',
                   fontsize=14)
        #ax3.set_title("Solution")
        ax3.set_title("X_slice: {:.2g} km".format(TCI.xvec[j1]))
        ax3.set_ylabel('Z km')
        ax3.set_xlabel('Y km')
        
        im = ax4.imshow(xz,origin='lower',vmin=vmin,vmax=vmax,aspect = 'auto',
                        extent=[TCI.xvec[0],TCI.xvec[-1],TCI.zvec[0],TCI.zvec[-1]],cmap=plt.cm.bone)
        CS = ax4.contour(xz, levels,
                     origin='lower',
                     linewidths=2,
                     extent=[TCI.xvec[0],TCI.xvec[-1],TCI.zvec[0],TCI.zvec[-1]],cmap=plt.cm.hot_r)
        zc = CS.collections[-1]
        plt.setp(zc, linewidth=4)
        plt.clabel(CS, levels[1::2],  # label every second level
                   inline=1,
                   fmt='%.2g',
                   fontsize=14)
        ax4.set_title("Y_slice: {:.2g} km".format(TCI.yvec[j2]))
        ax4.set_xlabel('X km')
        ax4.set_ylabel('Z km')
        plt.savefig("{}/fig-{:04d}.png".format(output_folder,i))  
        ax1.cla()
        ax2.cla()
        ax3.cla()
        ax4.cla()
        i += 1
    make_animation(output_folder,prefix='fig',fps=int(TCI.nz/float(num_seconds)))        

  
def plot_datapack(datapack,ant_idx=-1,time_idx=[0], dir_idx=-1,figname=None,vmin=None,vmax=None):
    assert datapack.ref_ant is not None, "set DataPack ref_ant first"
    directions, patch_names = datapack.get_directions(dir_idx=dir_idx)
    antennas, antLabels = datapack.get_antennas(ant_idx=ant_idx)
    times,timestamps = datapack.get_times(time_idx=time_idx)
    dtec = np.stack([np.mean(datapack.get_dtec(ant_idx = ant_idx,dir_idx=dir_idx,time_idx=time_idx),axis=1)],axis=1)
    Na = len(antennas)
    Nt = len(times)
    Nd = len(directions)
    ref_ant_idx = None
    for i in range(Na):
        if antLabels[i] == datapack.ref_ant:
            ref_ant_idx = i
    fixtime = times[Nt>>1]
    phase = datapack.get_center_direction()
    array_center = datapack.radio_array.get_center()
    uvw = UVW(location = array_center.earth_location,obstime = fixtime,phase = phase)
    ants_uvw = antennas.transform_to(uvw)

    #make plots, M by 4
    M = (Na>>2) + 1 + 1
    fig = plt.figure(figsize=(11.,11./4.*M))
    #use direction average as phase tracking direction
    if vmax is None:  
        vmax = np.percentile(dtec.flatten(),99)
        #vmax=np.max(dtec)
    if vmin is None:
        vmin = np.percentile(dtec.flatten(),1)
        #vmin=np.min(dtec)
    
        
    N = 25
    dirs_uvw = directions.transform_to(uvw)
    factor300 = 300./dirs_uvw.w.value
    U,V = np.meshgrid(np.linspace(np.min(dirs_uvw.u.value*factor300),np.max(dirs_uvw.u.value*factor300),N),
                          np.linspace(np.min(dirs_uvw.v.value*factor300),np.max(dirs_uvw.v.value*factor300),N))
    
    i = 0 
    while i < Na:
        ax = fig.add_subplot(M,4,i+1)

        dx = np.sqrt((ants_uvw.u[i] - ants_uvw.u[ref_ant_idx])**2 + (ants_uvw.v[i] - ants_uvw.v[ref_ant_idx])**2).to(au.km).value
        ax.annotate(s="{} : {:.2g} km".format(antLabels[i],dx),xy=(.2,.8),xycoords='axes fraction')
        if i == 0:
            #ax.annotate(s="{} : {:.2g} km\n{}".format(antLabels[i],dx,fixtime.isot),xy=(.2,.8),xycoords='axes fraction')
            #ax.annotate(s=fixtime.isot,xy=(.2,0.05),xycoords='axes fraction')
            ax.set_title(fixtime.isot)
        #ax.set_title("Ref. Proj. Dist.: {:.2g} km".format(dx))
        ax.set_xlabel("U km")
        ax.set_ylabel("V km")
        
            
        
        D = interp_nearest(dirs_uvw.u.value*factor300,dirs_uvw.v.value*factor300,dtec[i,0,:],U.flatten(),V.flatten()).reshape(U.shape)
        im = ax.imshow(D,origin='lower',extent=(np.min(U),np.max(U),np.min(V),np.max(V)),aspect='auto',
                      vmin = vmin, vmax= vmax,cmap=plt.cm.coolwarm,alpha=1.)
        sc1 = ax.scatter(dirs_uvw.u.value*factor300,dirs_uvw.v.value*factor300, c='black',
                        marker='+')
        i += 1
    ax = fig.add_subplot(M,4,Na+1)
    plt.colorbar(im,cax=ax,orientation='vertical')
    if figname is not None:
        plt.savefig("{}.png".format(figname),format='png')
    else:
        plt.show()
    plt.close()
    
def animate_datapack(datapack,output_folder, ant_idx=-1,time_idx=-1,dir_idx=-1):
    try:
        os.makedirs(output_folder)
    except:
        pass
    antennas,antenna_labels = datapack.get_antennas(ant_idx = ant_idx)
    patches, patch_names = datapack.get_directions(dir_idx = dir_idx)
    times,timestamps = datapack.get_times(time_idx=time_idx)
    datapack.set_reference_antenna(antenna_labels[0])
    #plot_datapack(datapack,ant_idx = ant_idx, time_idx = time_idx, dir_idx = dir_idx,figname='{}/dobs'.format(output_folder))
    dobs = datapack.get_dtec(ant_idx = ant_idx, time_idx = time_idx, dir_idx = dir_idx)
    vmin = np.percentile(dobs,1)
    vmax = np.percentile(dobs,99)
    Na = len(antennas)
    Nt = len(times)
    Nd = len(patches) 
    j = 0
    idx = 0
    while j < Nt:
        fig = "{}/fig-{:04d}".format(output_folder,idx)
        plot_datapack(datapack,ant_idx=ant_idx,time_idx=[j,j+1], dir_idx=dir_idx,figname=fig,vmin=vmin,vmax=vmax)
        idx += 1
        j += 2
    make_animation(output_folder,prefix="fig".format(output_folder),fps=int(5.))


