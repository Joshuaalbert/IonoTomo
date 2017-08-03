
# coding: utf-8

# In[ ]:

import dask.array as da
import os
import numpy as np
import pylab as plt
import h5py
from dask.dot import dot_graph
#from dask.multiprocessing import get
from dask import get
from functools import partial
from time import sleep, clock
from scipy.integrate import simps

from dask.callbacks import Callback
from distributed import Client
#from multiprocessing.pool import ThreadPool

from ForwardEquation import forward_equation, forward_equation_dask
from Gradient import compute_gradient, compute_gradient_dask
from tri_cubic import TriCubic
from LineSearch import line_search
from InfoCompleteness import precondition
from InitialModel import create_initial_model
from CalcRays import calc_rays,calc_rays_dask
from real_data import plot_datapack
from PlotTools import animate_tci_slices
from Covariance import Covariance


def store_Fdot(resettable,output_folder,n1,n2,Fdot,gamma,beta,dm,dgamma,v,sigma_m,L_m):
    filename="{}/F{}gamma{}.hdf5".format(output_folder,n1,n2)
    if os.path.isfile(filename) and resettable:
        return filename
    #gamma.save(filename)
    out = Fdot.copy()
    xvec = dm.xvec
    yvec = dm.yvec
    zvec = dm.zvec
    gamma_dm = scalarProduct(gamma.get_shaped_array(),dm.get_shaped_array(),sigma_m,L_m,xvec,yvec,zvec)
    a = dm.m*(beta*gamma_dm - scalarProduct(dgamma.get_shaped_array(),Fdot.get_shaped_array(),sigma_m,L_m,xvec,yvec,zvec))
    a -= v.m*gamma_dm
    a /= scalarProduct(dgamma.get_shaped_array(),dm.get_shaped_array(),sigma_m,L_m,xvec,yvec,zvec)
    out.m + a
    print("Beta: {}".format(beta))
    print("Difference: {}".format(np.dot(out.m,gamma.m)/np.linalg.norm(out.m)/np.linalg.norm(gamma.m)))
    
    if resettable:
        out.save(filename)
        return filename
    else:
        return out

def pull_Fdot(resettable,filename):
    if resettable:
        return TriCubic(filename=filename)
    else:
        return filename

def pull_gamma(resettable,filename):
    if resettable:
        return TriCubic(filename=filename)
    else:
        return filename

def store_gamma(resettable,output_folder,n,rays, g, dobs, i0, K_ne, m_tci, mPrior, CdCt, sigma_m, Nkernel, size_cell,cov_obj):
    filename='{}/gamma_{}.hdf5'.format(output_folder,n)
    if os.path.isfile(filename) and resettable:
        return filename
    gradient = compute_gradient_dask(rays, g, dobs, i0, K_ne, m_tci, mPrior.get_shaped_array(), CdCt, sigma_m, Nkernel, size_cell,cov_obj)
    TCI = TriCubic(m_tci.xvec,m_tci.yvec,m_tci.zvec,gradient)
    
    if resettable:
        TCI.save(filename)
        return filename
    else:
        return TCI

def plot_gamma(output_folder,n,TCI):
    foldername = '{}/gamma_{}'.format(output_folder,n)
    animate_tci_slices(TCI,foldername,num_seconds=20.)
    return foldername
    

def store_forwardEq(resettable,output_folder,n,template_datapack,ant_idx,time_idx,dir_idx,rays,K_ne,m_tci,i0):
    filename = "{}/g_{}.hdf5".format(output_folder,n)
    if os.path.isfile(filename) and resettable:
        return filename
    assert not np.any(np.isnan(m_tci.m)), "nans in model"
    g = forward_equation(rays,K_ne,m_tci,i0)
    assert not np.any(np.isnan(g)), "nans in g"
    datapack = template_datapack.clone()
    datapack.set_dtec(g,ant_idx=ant_idx,time_idx=time_idx,dir_idx = dir_idx)
    
    dobs = template_datapack.get_dtec(ant_idx=ant_idx,time_idx=time_idx,dir_idx = dir_idx)
    vmin = np.min(dobs)
    vmax = np.max(dobs)
    plot_datapack(datapack,ant_idx=ant_idx,time_idx=time_idx,dir_idx = dir_idx,
                figname=filename.split('.')[0], vmin = vmin, vmax = vmax)#replace('hdf5','png'))
    if resettable:
        datapack.save(filename)
        return filename
    else:
        return datapack

def pull_forwardEq(resettable,filename,ant_idx,time_idx,dir_idx):
    if resettable:
        datapack = DataPack(filename=filename)
        g = datapack.get_dtec(ant_idx=ant_idx,time_idx=time_idx,dir_idx = dir_idx)
        return g
    else:
        g = filename.get_dtec(ant_idx=ant_idx,time_idx=time_idx,dir_idx = dir_idx)
        return g

def calc_epsilon(output_folder,n,phi,m_tci,rays,K_ne,i0,g,dobs,CdCt):
    bins = max(10,int(np.ceil(np.sqrt(g.size))))
    drad = 8.44797256e-7/120e6
    dtau = 1.34453659e-7/120e6**2
    r = dtau/drad*1e9#mu sec factor
    plt.figure()
    plt.hist(g.flatten(),alpha=0.2,label='g',bins=bins)
    plt.hist(dobs.flatten(),alpha=0.2,label='dobs',bins=bins)
    plt.legend(frameon=False)
    plt.savefig("{}/data-hist-{}.png".format(output_folder,n))
    plt.clf()
    plt.hist((g-dobs).flatten()*drad*1e16,bins=bins)
    plt.xlabel(r"$d\phi$ [rad] | {:.2f} delay [ns]".format(r))
    plt.savefig("{}/datadiff-hist-{}.png".format(output_folder,n))
    plt.close('all')
    ep,S,reduction = line_search(rays,K_ne,m_tci,i0,phi.get_shaped_array(),g,dobs,CdCt,figname="{}/line_search{}".format(output_folder,n))
    return ep,S,reduction

def store_m(resettable,output_folder,n,m_tci0,phi,rays,K_ne,i0,g,dobs,CdCt,state_file):
    filename = "{}/m_{}.hdf5".format(output_folder,n)
    with h5py.File(state_file,'w') as state:
        if '/{}/epsilon_n'.format(n) not in state:
            epsilon_n,S,reduction = calc_epsilon(output_folder,n,phi,m_tci0,rays,K_ne,i0,g,dobs,CdCt)
            state['/{}/epsilon_n'.format(n)] = epsilon_n
            state['/{}/S'.format(n)] = S
            state['/{}/reduction'.format(n)] = reduction
            state.flush()
        else:
            epsilon_n,S,reduction = state['/{}/epsilon_n'.format(n)], state['/{}/S'.format(n)], state['/{}/reduction'.format(n)]
            
    if os.path.isfile(filename) and resettable:
        return filename
    m_tci = m_tci0.copy()
    m_tci.m -= epsilon_n*phi.m
    if resettable:
        m_tci.save(filename)
        return filename
    else:
        return m_tci

def pull_m(resettable,filename):
    if resettable:
        return TriCubic(filename=filename)
    else:
        return filename #object not filename


def scalarProduct(a,b,sigma_m,L_m,xvec,yvec,zvec):
    out = a*b
    out = simps(simps(simps(out,zvec,axis=2),yvec,axis=1),xvec,axis=0)
    #out /= (np.pi*8.*sigma_m**2 * L_m**3)
    return out

def calcBeta(dgamma, v, dm,sigma_m,L_m):
    xvec = dgamma.xvec
    yvec = dgamma.yvec
    zvec = dgamma.zvec
    beta = 1. + scalarProduct(dgamma.get_shaped_array(),v.get_shaped_array(),sigma_m,L_m,xvec,yvec,zvec)/(scalarProduct(dgamma.get_shaped_array(),dm.get_shaped_array(),sigma_m,L_m,xvec,yvec,zvec) + 1e-15)
    print("E[|dm|] = {} | E[|dgamma|] = {}".format(np.mean(np.abs(dm.m)),np.mean(np.abs(dgamma.m))))
    return beta

def diffTCI(TCI1,TCI2):
    TCI = TCI1.copy()
    TCI.m -= TCI2.m
    return TCI

def store_F0dot(resettable,output_folder,n,F0,gamma):
    filename="{}/F0gamma{}.hdf5".format(output_folder,n)
    if os.path.isfile(filename) and resettable:
        return filename
    out = gamma.copy()
    out.m *= F0.m
    if resettable:
        out.save(filename)
        return filename
    else:
        return out

def plot_model(output_folder,n,mModel,mPrior,K_ne):
    tmp = mModel.m.copy()
    np.exp(mModel.m,out=mModel.m)
    mModel.m *= K_ne
    mModel.m -= K_ne*np.exp(mPrior.m)
    foldername = '{}/m_{}'.format(output_folder,n)
    animate_tci_slices(mModel,foldername,num_seconds=20.)
    mModel.m = tmp
    print("Animation of model - prior in {}".format(foldername))
    return foldername

def create_bfgs_dask(resettable,output_folder,N,datapack,L_ne,size_cell,i0, ant_idx=-1, dir_idx=-1, time_idx = [0]):
    try:
        os.makedirs(output_folder)
    except:
        pass
    print("Using output folder: {}".format(output_folder))
    state_file = "{}/state".format(output_folder)
    straight_line_approx = True
    tmax = 1000.
    antennas,antenna_labels = datapack.get_antennas(ant_idx = ant_idx)
    patches, patch_names = datapack.get_directions(dir_idx = dir_idx)
    times,timestamps = datapack.get_times(time_idx=time_idx)
    datapack.set_reference_antenna(antenna_labels[i0])
    #plot_datapack(datapack,ant_idx = ant_idx, time_idx = time_idx, dir_idx = dir_idx,figname='{}/dobs'.format(output_folder))
    dobs = datapack.get_dtec(ant_idx = ant_idx, time_idx = time_idx, dir_idx = dir_idx)
    Na = len(antennas)
    Nt = len(times)
    Nd = len(patches) 
    fixtime = times[Nt>>1]
    phase = datapack.get_center_direction()
    array_center = datapack.radio_array.get_center()
    #Average time axis down and center on fixtime
    if Nt == 1:
        var = (0.5*np.percentile(dobs[dobs>0],25) + 0.5*np.percentile(-dobs[dobs<0],25))**2
        Cd = np.ones([Na,1,Nd],dtype=np.double)*var
        Ct = (np.abs(dobs)*0.05)**2
        CdCt = Cd + Ct        
    else:
        dt = times[1].gps - times[0].gps
        print("Averaging down window of length {} seconds [{} timestamps]".format(dt*Nt, Nt))
        Cd = np.stack([np.var(dobs,axis=1)],axis=1)
        dobs = np.stack([np.mean(dobs,axis=1)],axis=1)
        Ct = (np.abs(dobs)*0.05)**2
        CdCt = Cd + Ct
        time_idx = [Nt>>1]
        times,timestamps = datapack.get_times(time_idx=time_idx)
        Nt = len(times)
    print("E[S/N]: {} +/- {}".format(np.mean(np.abs(dobs)/np.sqrt(CdCt+1e-15)),np.std(np.abs(dobs)/np.sqrt(CdCt+1e-15))))
    vmin = np.min(datapack.get_dtec(ant_idx = ant_idx, time_idx = time_idx, dir_idx = dir_idx))
    vmax = np.max(datapack.get_dtec(ant_idx = ant_idx, time_idx = time_idx, dir_idx = dir_idx))
    plot_datapack(datapack,ant_idx=ant_idx,time_idx=time_idx,dir_idx = dir_idx,
            figname='{}/dobs'.format(output_folder), vmin = vmin, vmax = vmax)#replace('hdf5','png'))
    ne_tci = create_initial_model(datapack,ant_idx = ant_idx, time_idx = time_idx, dir_idx = dir_idx, zmax = tmax,spacing=size_cell)
    #make uniform
    #ne_tci.m[:] = np.mean(ne_tci.m)
    ne_tci.save("{}/nePriori.hdf5".format(output_folder))
    rays = calc_rays(antennas,patches,times, array_center, fixtime, phase, ne_tci, datapack.radio_array.frequency, 
                    straight_line_approx, tmax, ne_tci.nz)
    m_tci = ne_tci.copy()
    K_ne = np.mean(m_tci.m)
    m_tci.m /= K_ne
    np.log(m_tci.m,out=m_tci.m)
    
    Nkernel = max(1,int(float(L_ne)/size_cell))
    sigma_m = np.log(10.)#ne = K*exp(m+dm) = K*exp(m)*exp(dm), exp(dm) in (0.1,10) -> dm = (log(10) - log(0.1))/2.
    cov_obj = Covariance(m_tci,sigma_m,L_ne,7./2.)
    #uvw = UVW(location = datapack.radio_array.get_center().earth_location,obstime = fixtime,phase = phase)
    #ants_uvw = antennas.transform_to(uvw).cartesian.xyz.to(au.km).value.transpose()
    #dirs_uvw = patches.transform_to(uvw).cartesian.xyz.value.transpose()
    F0 = precondition(ne_tci, datapack,ant_idx=ant_idx, dir_idx=dir_idx, time_idx = time_idx)
    #
    dsk = {}
    for n in range(int(N)):
        #g_n
        dsk['store_forwardEq{}'.format(n)] = (store_forwardEq,resettable,'output_folder',n,'template_datapack','ant_idx','time_idx','dir_idx','rays',
                                              'K_ne','pull_m{}'.format(n),'i0')
        dsk['pull_forwardEq{}'.format(n)] = (pull_forwardEq,resettable,'store_forwardEq{}'.format(n),'ant_idx','time_idx','dir_idx')
        #gradient
        dsk['store_gamma{}'.format(n)] = (store_gamma,resettable,'output_folder',n,'rays', 'pull_forwardEq{}'.format(n), 'dobs', 'i0', 'K_ne', 
                                        'pull_m{}'.format(n),'mprior', 'CdCt', 'sigma_m', 'Nkernel', 'size_cell', 'cov_obj')
        dsk['pull_gamma{}'.format(n)] = (pull_gamma,resettable,'store_gamma{}'.format(n))
        #m update
        dsk['store_m{}'.format(n+1)] = (store_m,resettable,'output_folder',n+1,'pull_m{}'.format(n),'pull_phi{}'.format(n),'rays',
                                    'K_ne','i0','pull_forwardEq{}'.format(n),'dobs','CdCt','state_file')
        dsk['pull_m{}'.format(n+1)] = (pull_m,resettable,'store_m{}'.format(n+1))
        dsk['plot_m{}'.format(n+1)] = (plot_model,'output_folder',n+1,'pull_m{}'.format(n+1),'mprior','K_ne')
        dsk['plot_gamma{}'.format(n)] = (plot_gamma,'output_folder',n,'pull_gamma{}'.format(n))
        #phi
        dsk['pull_phi{}'.format(n)] = (pull_Fdot,resettable,'store_F{}(gamma{})'.format(n,n))
        dsk['store_F{}(gamma{})'.format(n+1,n+1)] = (store_Fdot,resettable,'output_folder', n+1, n+1 ,
                                                     'pull_F{}(gamma{})'.format(n,n+1),
                                                     'pull_gamma{}'.format(n+1),
                                                     'beta{}'.format(n),
                                                     'dm{}'.format(n),
                                                     'dgamma{}'.format(n),
                                                     'v{}'.format(n),
                                                     'sigma_m','L_m'
                                                    )
        for i in range(1,n+1):
            dsk['store_F{}(gamma{})'.format(i,n+1)] = (store_Fdot, resettable,'output_folder',i, n+1 ,
                                                     'pull_F{}(gamma{})'.format(i-1,n+1),
                                                     'pull_gamma{}'.format(n+1),
                                                     'beta{}'.format(i-1),
                                                     'dm{}'.format(i-1),
                                                     'dgamma{}'.format(i-1),
                                                     'v{}'.format(i-1),
                                                       'sigma_m','L_m'
                                                    )
            dsk['pull_F{}(gamma{})'.format(i,n+1)] = (pull_Fdot,resettable,'store_F{}(gamma{})'.format(i,n+1))
        #should replace for n=0
        dsk['store_F0(gamma{})'.format(n)] = (store_F0dot, resettable,'output_folder',n, 'pull_F0','pull_gamma{}'.format(n))
        dsk['pull_F0(gamma{})'.format(n)] = (pull_Fdot,resettable,'store_F0(gamma{})'.format(n))
#         #epsilon_n       
#         dsk['ep{}'.format(n)] = (calc_epsilon,n,'pull_phi{}'.format(n),'pull_m{}'.format(n),'rays',
#                                     'K_ne','i0','pull_forwardEq{}'.format(n),'dobs','CdCt')
        #
        dsk['beta{}'.format(n)] = (calcBeta,'dgamma{}'.format(n),'v{}'.format(n),'dm{}'.format(n),'sigma_m','L_m')
        dsk['dgamma{}'.format(n)] = (diffTCI,'pull_gamma{}'.format(n+1),'pull_gamma{}'.format(n))
        dsk['dm{}'.format(n)] = (diffTCI,'pull_m{}'.format(n+1),'pull_m{}'.format(n))
        dsk['v{}'.format(n)] = (diffTCI,'pull_F{}(gamma{})'.format(n,n+1),'pull_phi{}'.format(n))
    dsk['pull_F0'] = F0 
    dsk['template_datapack'] = datapack
    dsk['ant_idx'] = ant_idx
    dsk['time_idx'] = time_idx
    dsk['dir_idx'] = dir_idx
    dsk['pull_m0'] = 'mprior'
    dsk['i0'] = i0
    dsk['K_ne'] = K_ne
    dsk['dobs'] = dobs
    dsk['mprior'] = m_tci
    dsk['CdCt'] = CdCt
    dsk['sigma_m'] = sigma_m
    dsk['Nkernel'] = Nkernel
    dsk['L_m'] = L_ne
    dsk['size_cell'] = size_cell
    dsk['cov_obj'] = cov_obj
    #calc rays
    #dsk['rays'] = (calc_rays_dask,'antennas','patches','times', 'array_center', 'fixtime', 'phase', 'ne_tci', 'frequency',  'straight_line_approx','tmax')
    dsk['rays'] = rays
    dsk['output_folder'] = output_folder
    dsk['resettable'] = resettable
    dsk['state_file'] = state_file
    
    return dsk


class TrackingCallbacks(Callback):
    def _start(self,dsk):
        self.start_time = clock()
    def _pretask(self, key, dask, state):
        """Print the key of every task as it's started"""
        self.t1 = clock()
        print('Starting {} at {} seconds'.format(key,self.t1-self.start_time))
    def _posttask(self,key,result,dsk,state,id):
        print("{} took {} seconds".format(repr(key),clock() - self.t1))
    def _finish(self,dsk,state,errored):
        self.endTime = clock()
        dt = (self.endTime - self.start_time)
        print("Approximate time to complete: {} time units".format(dt))
        

    
if __name__=='__main__':
    from real_data import DataPack
    from AntennaFacetSelection import select_antennas_facets
    from dask.diagnostics import Profiler, ResourceProfiler, CacheProfiler
    from dask.diagnostics import visualize
    #from InitialModel import createTurbulentlModel
    i0 = 0
    datapack = DataPack(filename="output/test/simulate/simulate_3/datapack_sim.hdf5")
    #datapack = DataPack(filename="output/test/datapack_obs.hdf5")
    #flags = datapack.find_flagged_antennas()
    #datapack.flag_antennas(flags)
    datapack_sel = select_antennas_facets(20, datapack, ant_idx=-1, dir_idx=-1, time_idx = np.arange(1))
    #pert_tci = createTurbulentlModel(datapack_sel,ant_idx = -1, time_idx = -1, dir_idx = -1, zmax = 1000.)
    L_ne = 25.
    size_cell = 5.
    dsk = create_bfgs_dask(True, "output/test/bfgs_3_1", 25,datapack_sel,L_ne,size_cell,i0, ant_idx=-1, dir_idx=-1, time_idx = np.arange(1))
    #dot_graph(dsk,filename="{}/BFGS_graph".format(output_folder),format='png')
    #dot_graph(dsk,filename="{}/BFGS_graph".format(output_folder),format='svg')
    #client = Client()
    #with TrackingCallbacks():
    with Profiler() as prof, ResourceProfiler(dt=0.25) as rprof, CacheProfiler() as cprof:
        get(dsk,['plot_m25'])
    visualize([prof,rprof,cprof])
    

