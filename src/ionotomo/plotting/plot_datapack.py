import pylab as plt
import numpy as np
import os
from concurrent import futures
from ionotomo.astro.real_data import DataPack
from ionotomo.astro.frames.uvw_frame import UVW
import astropy.coordinates as ac
import astropy.time as at
import astropy.units as au

try:
    import cmocean
    phase_cmap = cmocean.cm.phase
except ImportError:
    phase_cmap = plt.cm.hsv


from scipy.spatial import ConvexHull, cKDTree
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.colors as colors
import matplotlib
import time


class DatapackPlotter(object):
    def __init__(self,datapack):
        if isinstance(datapack,str):
            datapack = DataPack(filename=datapack)
        self.datapack = datapack
        assert self.datapack.ref_ant is not None, "set DataPack ref_ant first"

    def _create_polygon_plot(self,points, values=None, N = None,ax=None,cmap=plt.cm.bone,overlay_points=True,title=None,polygon_labels=None,reverse_x=False):
        # get nearest points (without odd voronoi extra regions)
        k = cKDTree(points)
        dx = np.max(points[:,0]) - np.min(points[:,0])
        dy = np.max(points[:,1]) - np.min(points[:,1])
        N = N or int(min(max(100,points.shape[0]*2),500))
        x = np.linspace(np.min(points[:,0])-0.1*dx,np.max(points[:,0])+0.1*dx,N)
        y = np.linspace(np.min(points[:,1])-0.1*dy,np.max(points[:,1])+0.1*dy,N)
        X,Y = np.meshgrid(x,y,indexing='ij')
        # interior points population
        points_i = np.array([X.flatten(),Y.flatten()]).T
        # The match per input point
        dist,i = k.query(points_i,k=1)
        # the polygons are now created using convex hulls
        # order is by point order
        patches = []
        for group in range(points.shape[0]):
            points_g = points_i[i==group,:]
            hull = ConvexHull(points_g)
            nodes = points_g[hull.vertices,:]
            poly = Polygon(nodes,closed=False)
            patches.append(poly)
        if ax is None:
            fig,ax = plt.subplots()
            print("Making new plot")
        if values is None:
            values = np.zeros(len(patches))#random.uniform(size=len(patches))
        p = PatchCollection(patches,cmap=cmap)
        p.set_array(values)
        ax.add_collection(p)
        #plt.colorbar(p)
        if overlay_points:
            ax.scatter(points[:,0],points[:,1],marker='+',c='black')
        if reverse_x:
            ax.set_xlim([np.max(points_i[:,0]),np.min(points_i[:,0])])
        else:
            ax.set_xlim([np.min(points_i[:,0]),np.max(points_i[:,0])])
        ax.set_ylim([np.min(points_i[:,1]),np.max(points_i[:,1])])
        ax.set_facecolor('black')
        if title is not None:
            if reverse_x:
                ax.text(np.max(points_i[:,0])-0.05*dx,np.max(points_i[:,1])-0.05*dy,title,ha='left',va='top',backgroundcolor=(1.,1.,1., 0.5))
            else:
                ax.text(np.min(points_i[:,0])+0.05*dx,np.max(points_i[:,1])-0.05*dy,title,ha='left',va='top',backgroundcolor=(1.,1.,1., 0.5))
#            Rectangle((x, y), 0.5, 0.5,
#    alpha=0.1,facecolor='red',label='Label'))
#            ax.annotate(title,xy=(0.8,0.8),xycoords='axes fraction')
        return ax, p

    def plot(self, ant_idx=-1, time_idx = [0], dir_idx=-1, freq_idx=[0], fignames=None, vmin=None,vmax=None,mode='perantenna',observable='phase',phase_wrap=True, plot_crosses=True,plot_facet_idx=False,plot_patchnames=False,labels_in_radec=False,show=False):
        """
        Plot datapack with given parameters.
        """
        SUPPORTED = ['perantenna']
        assert mode in SUPPORTED, "only 'perantenna' supported currently".format(SUPPORTED)
        if fignames is None:
            save_fig = False
            show = True
        else:
            save_fig = True
            show = show and True #False
        if plot_patchnames:
            plot_facet_idx = False
        if plot_patchnames or plot_facet_idx:
            plot_crosses = False
        if not show:
            print('turning off display')
            matplotlib.use('Agg')

        ###
        # Set up plotting

        if fignames is not None:
            if not isinstance(fignames,(tuple,list)):
                fignames = [fignames]
        directions, patch_names = self.datapack.get_directions(dir_idx=dir_idx)
        antennas, antenna_labels = self.datapack.get_antennas(ant_idx=ant_idx)
        times,timestamps = self.datapack.get_times(time_idx=time_idx)
        freqs = self.datapack.get_freqs(freq_idx=freq_idx)
        if fignames is not None:
            assert len(times) == len(fignames)

        if observable == 'phase':
            obs = self.datapack.get_phase(ant_idx = ant_idx,dir_idx=dir_idx,time_idx=time_idx, freq_idx = freq_idx)
        elif observable == 'variance':
            phase_wrap=False
            obs = self.datapack.get_variance(ant_idx = ant_idx,dir_idx=dir_idx,time_idx=time_idx, freq_idx = freq_idx)
        elif observable == 'std':
            phase_wrap = False
            obs = np.sqrt(self.datapack.get_variance(ant_idx = ant_idx,dir_idx=dir_idx,time_idx=time_idx, freq_idx = freq_idx))
        elif observable == 'snr':
            obs = np.abs(self.datapack.get_phase(ant_idx = ant_idx,dir_idx=dir_idx,time_idx=time_idx, freq_idx = freq_idx) \
                    / (np.sqrt(self.datapack.get_variance(ant_idx = ant_idx,dir_idx=dir_idx,time_idx=time_idx, freq_idx = freq_idx)) + 1e-10))
            phase_wrap = False

        if phase_wrap:
            obs = np.angle(np.exp(1j*obs))
            vmin = -np.pi
            vmax = np.pi
            cmap = phase_cmap
        else:
            vmin = vmin or np.percentile(obs.flatten(),1)
            vmax = vmax or np.percentile(obs.flatten(),99)
            cmap = plt.cm.bone

        Na = len(antennas)
        Nt = len(times)
        Nd = len(directions)
        Nf = len(freqs)
        fixfreq = freqs[Nf>>1]
        ref_ant_idx = None
        for i in range(Na):
            if antenna_labels[i] == self.datapack.ref_ant:
                ref_ant_idx = i

        
        #ants_uvw = antennas.transform_to(uvw)

        ref_dist = np.sqrt((antennas.x - antennas.x[ref_ant_idx])**2 + (antennas.y - antennas.y[ref_ant_idx])**2+ (antennas.z - antennas.z[ref_ant_idx])**2).to(au.km).value
        if labels_in_radec:
            ra = directions.ra.deg
            dec = directions.dec.deg
            points = np.array([ra,dec]).T
        else:
            fixtime = times[0]
            phase_center = self.datapack.get_center_direction()
            array_center = self.datapack.radio_array.get_center()
            uvw = UVW(location = array_center.earth_location,obstime = fixtime,phase = phase_center)
            dirs_uvw = directions.transform_to(uvw)
            u_rad = np.arctan2(dirs_uvw.u.value,dirs_uvw.w.value)
            v_rad = np.arctan2(dirs_uvw.v.value,dirs_uvw.w.value)
            points = np.array([u_rad,v_rad]).T

        if mode == 'perantenna':
            
            M = int(np.ceil(np.sqrt(Na)))
            fig,axs = plt.subplots(nrows=M,ncols=M,sharex='col',sharey='row',squeeze=False, \
                    figsize=(4*M,4*M))
            fig.subplots_adjust(wspace=0., hspace=0.)
            axes_patches = []
            c = 0
            for row in range(M):
                for col in range(M):
                    ax = axs[row,col]
                    if col == 0:
                        ax.set_ylabel("Projected North (radians)" if not labels_in_radec else "DEC (deg)")
                            
                    if row == M - 1:
                        ax.set_xlabel("Projected East (radians)" if not labels_in_radec else "RA (deg)")
                    
                    if c >= Na:
                        continue
                    _, p = self._create_polygon_plot(points, values=None, N = None,
                            ax=ax,cmap=cmap,overlay_points=plot_crosses,
                            title="{} {:.1f}km".format(antenna_labels[c], ref_dist[c]),
                            reverse_x=labels_in_radec)
                    p.set_clim(vmin,vmax)
                    axes_patches.append(p)
                    c += 1
            
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            fig.colorbar(p, cax=cbar_ax, orientation='vertical')
            if show:
                plt.ion()
                plt.show()
            for j in range(Nt):
                print("Plotting {}".format(times[j].isot))
                for i in range(Na):
                    axes_patches[i].set_array(obs[i,j,:,0])
                axs[0,0].set_title("{} {:.1f} MHz : {}".format(observable, fixfreq/1e6, times[j].isot))
                fig.canvas.draw()
                if save_fig:
                    plt.savefig(fignames[j])
            if show:
                plt.close(fig)
                plt.ioff()

def _parallel_plot(arg):
    datapack,time_idx,kwargs,output_folder=arg
    dp = DatapackPlotter(datapack=datapack)
    fignames = [os.path.join(output_folder,"fig-{:04d}.png".format(j)) for j in time_idx]
    dp.plot(time_idx=time_idx,fignames=fignames,**kwargs)
    return fignames
    
def animate_datapack(datapack,output_folder,num_processes=1,**kwargs):
    try:
        os.makedirs(output_folder)
    except:
        pass

    dp = DatapackPlotter(datapack=datapack)
    times,timestamps = dp.datapack.get_times(time_idx=-1)
    args = []
    for i in range(num_processes):
        args.append((datapack,range(i,len(times),num_processes),kwargs,output_folder))
    with futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        jobs = executor.map(_parallel_plot,args)
        results = list(jobs)

def make_animation(datafolder,prefix='fig',fps=3):
    '''Given a datafolder with figures of format `prefix`-%04d.png create a 
    video at framerate `fps`.
    Output is datafolder/animation.mp4'''
    if os.system('ffmpeg -framerate {} -i {}/{}-%04d.png -vf scale="trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -profile:v high -pix_fmt yuv420p -g 30 -r 30 {}/animation.mp4'.format(fps,datafolder,prefix,datafolder)):
        print("{}/animation.mp4 exists already".format(datafolder))    



def test_vornoi():
    from scipy.spatial import Voronoi, voronoi_plot_2d
    import pylab as plt
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    import numpy as np

    points = np.random.uniform(size=[10,2])
    v = Voronoi(points)
    nodes = v.vertices
    regions = v.regions

    ax = plt.subplot()
    patches = []
    for reg in regions:
        if len(reg) < 3:
            continue
        poly = Polygon(np.array([nodes[i] for i in reg]),closed=False)
        patches.append(poly)
    p = PatchCollection(patches)
    p.set_array(np.random.uniform(size=len(patches)))
    ax.add_collection(p)
    #plt.colorbar(p)
    ax.scatter(points[:,0],points[:,1])
    ax.set_xlim([np.min(points[:,0]),np.max(points[:,0])])
    ax.set_ylim([np.min(points[:,1]),np.max(points[:,1])])
    plt.show()

def test_nearest():
    from scipy.spatial import ConvexHull, cKDTree
    import pylab as plt
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    import numpy as np

    points = np.random.uniform(size=[42,2])
    k = cKDTree(points)
    dx = np.max(points[:,0]) - np.min(points[:,0])
    dy = np.max(points[:,1]) - np.min(points[:,1])
    N = int(min(max(100,points.shape[0]*2),500))
    x = np.linspace(np.min(points[:,0])-0.1*dx,np.max(points[:,0])+0.1*dx,N)
    y = np.linspace(np.min(points[:,1])-0.1*dy,np.max(points[:,1])+0.1*dy,N)
    X,Y = np.meshgrid(x,y,indexing='ij')
    points_i = np.array([X.flatten(),Y.flatten()]).T
    dist,i = k.query(points_i,k=1)
    patches = []
    for group in range(points.shape[0]):
        points_g = points_i[i==group,:]
        hull = ConvexHull(points_g)
        nodes = points_g[hull.vertices,:]
        poly = Polygon(nodes,closed=False)
        patches.append(poly)
    ax = plt.subplot()
    p = PatchCollection(patches)
    p.set_array(np.random.uniform(size=len(patches)))
    ax.add_collection(p)
    #plt.colorbar(p)
    ax.scatter(points[:,0],points[:,1])
    ax.set_xlim([np.min(points_i[:,0]),np.max(points_i[:,0])])
    ax.set_ylim([np.min(points_i[:,1]),np.max(points_i[:,1])])
    ax.set_facecolor('black')
    plt.show()

def test():
    #from ionotomo.astro.real_data import generate_example_datapack
    #datapack = generate_example_datapack(Ndir=10,Nant=10,Ntime=20)
    #dp = DatapackPlotter(datapack='../data/rvw_datapack_full_phase_dec27_smooth.hdf5')
    #dp.plot(ant_idx=-1,dir_idx=-1,time_idx=-1,labels_in_radec=True)

    animate_datapack('../data/rvw_datapack_full_phase_dec27_smooth.hdf5',
            'test_output',num_processes=1,observable='phase',labels_in_radec=True,show=True)




if __name__=='__main__':
    test()

