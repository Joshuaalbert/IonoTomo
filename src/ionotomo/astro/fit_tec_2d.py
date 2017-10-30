import numpy as np
from ionotomo.astro.real_data import DataPack
from ionotomo.plotting.plot_tools import plot_datapack
from ionotomo.astro.frames.uvw_frame import UVW
import os
import logging as log
from scipy.interpolate import griddata
from ionotomo.utils.gaussian_process import *    

def fit_2d_cubic(x,y,xstar):
    '''Fit a cubic interpolation at the points xstar with the data x and y where
    x is shape (num_points, 2) and y is shape (num_points) and xstar is 
    (num_points_eval,2)'''
#    X,Y = np.meshgrid(xstar[:,0],xstar[:,1],indexing='ij')
    ystar = griddata(x, y, (xstar[:,0],xstar[:,1]), method='cubic')
    return ystar

def fit_datapack(datapack,template_datapack,ant_idx = -1, time_idx=-1, dir_idx=-1, freq_idx=-1):
    """Fit a datapack to a template datapack using Bayesian optimization
    of given kernel. Conjugation occurs at 350km."""

    antennas,antenna_labels = datapack.get_antennas(ant_idx)
    directions, patch_names = datapack.get_directions(dir_idx)
    times,timestamps = datapack.get_times(time_idx)
    freqs = datapack.get_freqs(freq_idx)
    phase = datapack.get_phase(ant_idx = ant_idx,dir_idx=dir_idx,time_idx=time_idx,freq_idx=-1)
    Na = len(antennas)
    Nt = len(times)
    Nd = len(directions) 
    Nf = len(freqs)

    template_datapack = template_datapack.clone()
    antennas_,antenna_labels_ = template_datapack.get_antennas(ant_idx)
    directions_, patch_names_ = template_datapack.get_directions(dir_idx)
    times_,timestamps_ = template_datapack.get_times(time_idx)
    freqs_ = template_datapack.get_freqs(freq_idx)
    Na_ = len(antennas_)
    Nt_ = len(times_)
    Nd_ = len(directions_)
    Nf_ = len(freqs_)

    ref_ant_idx = template_datapack.get_antenna_idx(datapack.ref_ant)
    fit_dtec = np.zeros([Na_,Nt_,Nd_])#np.stack([np.mean(datapack.get_dtec(ant_idx = ant_idx,dir_idx=dir_idx,time_idx=time_idx),axis=1)],axis=1)

    fixtime = times[Nt>>1]
    phase = datapack.get_center_direction()
    array_center = datapack.radio_array.get_center()
    uvw = UVW(location = array_center.earth_location,obstime = fixtime,phase = phase)
    ants_uvw = antennas.transform_to(uvw)
    dirs_uvw = directions.transform_to(uvw).cartesian.xyz.value.T
    ants_uvw_ = antennas_.transform_to(uvw)
    dirs_uvw_ = directions_.transform_to(uvw).cartesian.xyz.value.T
    flag_ants = []
    x_multi = []
    y_multi = []
    sigma_y_multi = []
    y_mean = []
    l = np.max(np.max(dirs_uvw[:,0:2],axis=0) - np.min(dirs_uvw[:,0:2],axis=0))
    K1 = SquaredExponential(2,l=l)#RationalQuadratic(l,1.,0.01)
    K2 = Diagonal(2)        
    K = K1 + K2
    xstar = dirs_uvw_[:,0:2]
    i = 0
    while i < Na_:
        if antenna_labels_[i] not in antenna_labels:
            flag_ants.append(antenna_labels_[i])
            i += 1
            continue
        log.info("Fitting lvl2 {}".format(antenna_labels_[i]))
        x = dirs_uvw[:,0:2]
        y = dtec[datapack.get_antenna_idx(antenna_labels_[i]),0,:]
        sigma_y = 0.01
        y_mean.append(np.mean(y))
        x_multi.append(x)
        y_multi.append(y-y_mean[-1])
        sigma_y_multi.append(0.01)
        hp = level2_solve(x,y,sigma_y,K)
        K.hyperparams = hp
        ystar,cov,lml = level1_solve(x,y,sigma_y,xstar,K)
        fit_dtec[i,0,:] = ystar
        i += 1
    fit_dtec -= fit_dtec[ref_ant_idx,:,:]
    template_datapack.flag_antennas(flag_ants)
    template_datapack.set_dtec(fit_dtec,ant_idx,time_idx,dir_idx)
    return template_datapack

