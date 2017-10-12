import astropy.units as au
import astropy.time as at
import astropy.coordinates as ac
import numpy as np
from time import clock
import pylab as plt
import h5py
import os
from ionotomo import *
from ionotomo.inversion.iterative_newton import forward_equation
from ionotomo.plotting.plot_tools import plot_datapack
import logging as log
#from ionotomo.astro.real_data import DataPack
#from ionotomo.inversion.fermat import Fermat
#from pointing_frame import Pointing
#from uvw_frame import UVW
#from IRI import a_priori_model, determine_inversion_domain
#from tri_cubic import TriCubic
#from progressbar import ProgressBar


def simulate_phase(datapack,ne_tci=None,num_threads=1,datafolder=None,
                     ant_idx=-1,time_idx=-1,dir_idx=-1,freq_idx=-1,do_plot_datapack=False,flag_remaining=False):
    '''Simulate a turulent ionosphere and store simulated observables in datapack.
    Each timestamp gets its own ionosphere and a random clock an const offset.
    datapack : DataPack
        The template datapack and also where observables are stored
    num_threads : int
        Number of simultaneous threads to run. default 1
    datafolder : str
        path to store ionospheres, if None (default) then do not store ionospheres
    ant_idx : list of int or -1 for all
        list containing indices of antennas to use. Others are flagged.
    time_idx : list of int or -1 for all
        list containing indices of timestamps to use. Others are flagged.
    dir_idx : list of int or -1 for all
        list containing indices of directions to use. Others are flagged.
    freq_idx : list of int of -1 for all
        list containing indices of frequencies to use. Others are flagged.'''
    #Set up datafolder
    if datafolder is not None:
        datafolder = os.path.join(os.getcwd(),datafolder)
        try:
            os.makedirs(datafolder)
        except:
            pass 
    if ne_tci is None:
        ne_tci = create_turbulent_model(datapack,factor=2.,corr=20.)
    if datafolder is not None:
        ne_tci.save(os.path.join(datafolder,"turbulent_ne.hdf5"))
    antennas,antenna_labels = datapack.get_antennas(ant_idx = ant_idx)
    patches, patch_names = datapack.get_directions(dir_idx=dir_idx)
    times,timestamps = datapack.get_times(time_idx=time_idx)
    freqs = datapack.get_freqs(freq_idx=freq_idx)
    Na = len(antennas)
    Nt = len(times)
    Nd = len(patches)  
    Nf = len(freqs)
    fixtime = times[Nt>>1]
    phase = datapack.get_center_direction()
    array_center = datapack.radio_array.get_center()
    rays = calc_rays(antennas,patches,times, array_center, fixtime, phase, ne_tci, freqs[Nf>>1], True, 1000., None)
    model = (np.log(ne_tci.M/1e11), 5e-9*np.random.normal(size=[Na,Nt]), np.pi/2.*np.pi*np.random.normal(size=Na))
    dobs = forward_equation(model, ne_tci, rays, freqs, K=1e11, i0 = 0)
    datapack.set_phase(dobs,ant_idx = ant_idx,time_idx=time_idx,dir_idx=dir_idx,freq_idx=freq_idx)
    datapack.set_reference_antenna(antenna_labels[0])
    
    if do_plot_datapack:
        plot_datapack(datapack,time_idx=time_idx,figname=None)

    #flagging
    if flag_remaining:
        antennas,antenna_labels_ = datapack.get_antennas(ant_idx = -1)
        patches, patch_names_ = datapack.get_directions(dir_idx= -1)
        times,timestamps_ = datapack.get_times(time_idx=-1)
        freqs_ = datapack.get_freqs(freq_idx=-1)
        flag_ants = []
        for a in antenna_labels_:
            if a not in antenna_labels:
                flag_ants.append(a)
        flag_times = []
        for t in timestamps_:
            if t not in timestamps:
                flat_times.append(t)
        flag_dirs = []
        for d in patch_names_:
            if d not in patch_names:
                flag_dirs.append(d)
        flag_freqs = []
        for l,f in enumerate(freqs_):
            if f not in freqs:
                flag_freqs.append(l)
        datapack.flag_antennas(flag_ants)
        datapack.flag_times(flag_times)
        datapack.flag_directions(flag_dirs)
        datapack.flag_freqs(flag_freqs)
    return datapack
    
