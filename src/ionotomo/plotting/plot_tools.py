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
try:
    import cmocean
    phase_cmap = cmocean.cm.phase
except:
    phase_cmap = plt.cm.hsv

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

  
def plot_datapack(datapack,ant_idx=-1,time_idx=[0], dir_idx=-1,freq_idx=-1,figname=None,vmin=None,vmax=None,mode='perantenna',observable='phase',phase_wrap=True):
    '''Plot phase at central frequency'''
    assert datapack.ref_ant is not None, "set DataPack ref_ant first"
    if len(time_idx) == 1 and figname is not None:
        figname = [figname]
    if len(time_idx) > 1 and figname is not None:
        assert len(time_idx) == len(figname)
    directions, patch_names = datapack.get_directions(dir_idx=dir_idx)
    antennas, antLabels = datapack.get_antennas(ant_idx=ant_idx)
    times,timestamps = datapack.get_times(time_idx=time_idx)
    freqs = datapack.get_freqs(freq_idx=freq_idx)
    if observable == 'phase':
        obs = datapack.get_phase(ant_idx = ant_idx,dir_idx=dir_idx,time_idx=time_idx, freq_idx = freq_idx)
    elif observable == 'prop':
        obs = datapack.get_prop(ant_idx = ant_idx,dir_idx=dir_idx,time_idx=time_idx, freq_idx = freq_idx)
    elif observable == 'variance':
        phase_wrap=False
        obs = datapack.get_variance(ant_idx = ant_idx,dir_idx=dir_idx,time_idx=time_idx, freq_idx = freq_idx)
    elif observable == 'std':
        phase_wrap = False
        obs = np.sqrt(datapack.get_variance(ant_idx = ant_idx,dir_idx=dir_idx,time_idx=time_idx, freq_idx = freq_idx))
    
    if phase_wrap:
        obs = np.angle(np.exp(1j*obs))
        vmin = -np.pi
        vmax = np.pi
        cmap = phase_cmap
    else:
        vmin = np.percentile(obs.flatten(), 5)
        vmax = np.percentile(obs.flatten(), 95)
        cmap = plt.cm.bone


    Na = len(antennas)
    Nt = len(times)
    Nd = len(directions)
    Nf = len(freqs)
    ref_ant_idx = None
    for i in range(Na):
        if antLabels[i] == datapack.ref_ant:
            ref_ant_idx = i

    for idx,j in enumerate(time_idx):
        print("Plotting {}".format(j))
        fixtime = times[idx]
        fixfreq = freqs[Nf>>1]
        phase = datapack.get_center_direction()
        array_center = datapack.radio_array.get_center()
        uvw = UVW(location = array_center.earth_location,obstime = fixtime,phase = phase)
        ants_uvw = antennas.transform_to(uvw)
        dirs_uvw = directions.transform_to(uvw)

        factor300 = 300./dirs_uvw.w.value

        if mode == 'perantenna':
            #make plots, M by M
            M = int(np.ceil(np.sqrt(Na)))
            fig = plt.figure(figsize=(4*M,4*M))
            #use direction average as phase tracking direction
            
            
            N = 25
            
            
            U,V = np.meshgrid(np.linspace(np.min(dirs_uvw.u.value*factor300),
                np.max(dirs_uvw.u.value*factor300),N),
                np.linspace(np.min(dirs_uvw.v.value*factor300),
                    np.max(dirs_uvw.v.value*factor300),N),indexing='ij')
            i = 0 
            while i < Na:
                ax = fig.add_subplot(M,M,i+1)

                dx = np.sqrt((ants_uvw.u[i] - ants_uvw.u[ref_ant_idx])**2 + (ants_uvw.v[i] - ants_uvw.v[ref_ant_idx])**2).to(au.km).value
                ax.annotate(s="{} : {:.2g} km".format(antLabels[i],dx),xy=(.2,.8),xycoords='axes fraction')
                if i == 0:
                    #ax.annotate(s="{} : {:.2g} km\n{}".format(antLabels[i],dx,fixtime.isot),xy=(.2,.8),xycoords='axes fraction')
                    #ax.annotate(s=fixtime.isot,xy=(.2,0.05),xycoords='axes fraction')
                    ax.set_title("Phase {} MHz : {}".format(fixfreq/1e6,fixtime.isot))
                #ax.set_title("Ref. Proj. Dist.: {:.2g} km".format(dx))
                ax.set_xlabel("Projected East km")
                ax.set_ylabel("Projected West km")

                D = interp_nearest(dirs_uvw.u.value*factor300,dirs_uvw.v.value*factor300,obs[i,idx,:,Nf>>1],U.flatten(),V.flatten()).reshape(U.shape)
                
                im = ax.imshow(D,origin='lower',extent=(np.min(U),np.max(U),np.min(V),np.max(V)),aspect='auto',
                              vmin = vmin, vmax= vmax,cmap=cmap,alpha=1.)
                sc1 = ax.scatter(dirs_uvw.u.value*factor300,dirs_uvw.v.value*factor300, c='black',
                                marker='+')
                i += 1
            #plt.tight_layout()
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            fig.colorbar(im, cax=cbar_ax,orientation='vertical')
        elif mode == 'perdirection':
            M = int(np.ceil(np.sqrt(Na)))
            fig = plt.figure(figsize=(4*M,4*M))
            #use direction average as phase tracking direction
            vmax = np.pi
            vmin = -np.pi

            N = 25
            
            U,V = np.meshgrid(np.linspace(np.min(ants_uvw.u.to(au.km).value),
                np.max(ants_uvw.u.to(au.km).value),N),
                np.linspace(np.min(ants_uvw.v.to(au.km).value),
                    np.max(ants_uvw.v.to(au.km).value),N),indexing='ij')
            k = 0 
            while k < Nd:
                ax = fig.add_subplot(M,M,k+1)

                #dx = np.sqrt((ants_uvw.u[i] - ants_uvw.u[ref_ant_idx])**2 + (ants_uvw.v[i] - ants_uvw.v[ref_ant_idx])**2).to(au.km).value
                ax.annotate(s="{} : {} ".format(patch_names[k],directions[k]),xy=(.2,.8),xycoords='axes fraction')
                if k == 0:
                    #ax.annotate(s="{} : {:.2g} km\n{}".format(antLabels[i],dx,fixtime.isot),xy=(.2,.8),xycoords='axes fraction')
                    #ax.annotate(s=fixtime.isot,xy=(.2,0.05),xycoords='axes fraction')
                    ax.set_title("Phase {} MHz : {}".format(fixfreq/1e6,fixtime.isot))
                #ax.set_title("Ref. Proj. Dist.: {:.2g} km".format(dx))
                ax.set_xlabel("Projected East km")
                ax.set_ylabel("Projected North km")
                
                
                D = interp_nearest(ants_uvw.u.to(au.km).value,ants_uvw.v.to(au.km).value,np.angle(np.exp(1j*phase_obs[:,idx,k,Nf>>1])),U.flatten(),V.flatten()).reshape(U.shape)
                im = ax.imshow(D,origin='lower',extent=(np.min(U),np.max(U),np.min(V),np.max(V)),aspect='auto',
                              vmin = vmin, vmax= vmax,cmap=phase_cmap,alpha=1.)
                sc1 = ax.scatter(ants_uvw.u.to(au.km).value,ants_uvw.v.to(au.km).value, c='black',
                                marker='+')
                k += 1
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            fig.colorbar(im, cax=cbar_ax,orientation='vertical')


        #plt.tight_layout()
        if figname is not None:
            plt.savefig("{}.png".format(figname[idx]),format='png')
        else:
            plt.show()
        plt.close()
    
def animate_datapack(datapack,output_folder, ant_idx=-1,time_idx=-1,dir_idx=-1,num_threads=1,mode='perantenna',observable='phase'):
    from dask.threaded import get
    from functools import partial
    try:
        os.makedirs(output_folder)
    except:
        pass
    times,timestamps = datapack.get_times(time_idx=time_idx)
    Nt = len(times)
#    j = 0
#    idx = 0
#    dsk = {}
#    objective = []
#    for thread in range(num_threads):
#        figs = []
#        time_idx = []
#        for j in range(thread,Nt,num_threads):
#            figs.append(os.path.join(output_folder,"fig-{:04d}".format(j)))
#            time_idx.append(j)
#        dsk[thread] = (partial(plot_datapack,ant_idx=ant_idx,time_idx=time_idx, dir_idx=dir_idx,figname=figs,mode=mode),datapack)
#        objective.append(thread)
    for j in range(Nt):
        fig = os.path.join(output_folder,"fig-{:04d}".format(j))
        plot_datapack(datapack,ant_idx=ant_idx,time_idx=[j], dir_idx=dir_idx,figname=fig,mode=mode,observable=observable)

    #get(dsk,objective,num_workers=num_threads)
    make_animation(output_folder,prefix="fig",fps=int(10))


