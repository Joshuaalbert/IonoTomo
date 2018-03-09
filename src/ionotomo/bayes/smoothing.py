import numpy as np
import tensorflow as tf
import logging
logging.basicConfig(format='%(asctime)s %(message)s')
import pylab as plt
import cmocean
from scipy.spatial import cKDTree
from ionotomo.tomography.pipeline import Pipeline
from ionotomo.settings import TFSettings
from timeit import default_timer
from ionotomo import *
import gpflow as gp
import sys
import h5py
import threading
from timeit import default_timer
#%matplotlib notebook
from concurrent import futures
from functools import partial
from threading import Lock
import astropy.units as au
import astropy.time as at


class Smoothing(object):
    def __init__(self,datapack):
        if isinstance(datapack, str):
            datapack = DataPack(filename=datapack)
        self.datapack = datapack

    def _make_coord_array(t,d,f):
        Nt,Nd,Nf = t.shape[0],d.shape[0], f.shape[0]
        X = np.zeros([Nt,Nd,Nf,4],dtype=np.float64)
        for j in range(Nt):
            for k in range(Nd):
                for l in range(Nf):
                    X[j,k,l,0:2] = d[k,:]
                    X[j,k,l,2] = t[j]   
                    X[j,k,l,3] = f[l]
        X = np.reshape(X,(Nt*Nd*Nf,4))
        return X

    def _solve_block(phase, error, coords, lock, pargs=None,verbose=False):
        if verbose:
            logging.warning("{}".format(pargs))
        error_scale = np.mean(np.abs(phase))*0.1/np.mean(error)
        if verbose:
            logging.warning("Error scaling {}".format(error_scale))
        y_mean = np.mean(phase)
        y_scale = np.std(phase) + 1e-6
        y = (phase - y_mean)/y_scale
        y = y.flatten()[:,None]
        var = (error/y_scale*error_scale)**2
        var = var.flatten()
        
        t,d,f = coords
        t_scale = np.max(t) - np.min(t) + 1e-6
        d_scale = np.std(d) + 1e-6
        f_scale = np.max(f) - np.min(f) + 1e-6
        t = (t - np.mean(t))/(t_scale+1e-6)
        d = (d - np.mean(d))/(d_scale+1e-6)
        f = (f - np.mean(f))/(f_scale+1e-6)
        X = Smoothing._make_coord_array(t,d,f)

        with tf.Session(graph=tf.Graph()) as sess:
            k_space = gp.kernels.RBF(2,active_dims = [0,1],lengthscales=[0.5])
            k_time = gp.kernels.RBF(1,active_dims = [2],lengthscales=[0.5])
            k_freq = gp.kernels.RBF(1,active_dims = [3], lengthscales=[0.5])
            kern = k_space*k_time*k_freq
            mean = gp.mean_functions.Constant()
            lock.acquire()
            try:
                with gp.defer_build():
                    m = gp.models.GPR(X, y, kern, mean_function=mean,var=var)
                    m.likelihood.variance.set_trainable(False)
                    m.compile()
            finally:
                lock.release()
            o = gp.train.ScipyOptimizer(method='BFGS')
            o.minimize(m,maxiter=1000)
            if verbose:
                logging.warning(m)
            kern_lengthscales = (
                    m.kern.rbf_1.lengthscales.value[0]*d_scale,
                    m.kern.rbf_2.lengthscales.value[0]*t_scale,
                    m.kern.rbf_3.lengthscales.value[0]*f_scale
                    )
            kern_variance = m.kern.rbf_1.variance.value*m.kern.rbf_2.variance.value*m.kern.rbf_3.variance.value*y_scale**2
            if verbose:
                logging.warning(kern_lengthscales)
                logging.warning(kern_variance)
            return kern_lengthscales, kern_variance
    def _ref_distance(self,uvw,antennas,i0=0):
        ants_uvw = antennas.transform_to(uvw)
        u = ants_uvw.u.to(au.km).value
        v = ants_uvw.v.to(au.km).value
        dist = np.sqrt((u - u[i0])**2 + (v - v[i0])**2)
        return dist

    def solve_time_intervals(self, save_file, ant_idx, time_idx, dir_idx, freq_idx, interval, shift, num_threads=1,verbose=False):
        """
        Solve for kernel characteristics over given domain.
        ant_idx, time_idx, dir_idx, freq_idx: the domain selectors
        interval: int interval in time to solve.
        shift: int the shift in time between solves.
        num_threads: int (default 1) the number of parallel solvers.
        Return interval start array, interval end array,
        the kernel length scales per antenna and variances per antenna 
        """
        datapack = self.datapack
        directions, patch_names = datapack.get_directions(dir_idx)
        times,timestamps = datapack.get_times(time_idx)
        antennas,antenna_labels = datapack.get_antennas(ant_idx)
        freqs = datapack.get_freqs(freq_idx)
        if ant_idx is -1:
            ant_idx = range(len(antennas))
        if time_idx is -1:
            time_idx = range(len(times))
        if freq_idx is -1:
            freq_idx = range(len(freqs))
        if dir_idx is -1:
            dir_idx = range(len(directions))

        phase = datapack.get_phase(ant_idx,time_idx,dir_idx,freq_idx)
        Na,Nt,Nd,Nf = phase.shape
        logging.warning("Working on shapes {}".format(phase.shape))

        assert interval <= Nt

        variance = datapack.get_variance(ant_idx,time_idx,dir_idx,freq_idx)
        error = np.sqrt(variance)
        data_mask = variance < 0
        error[data_mask] = 10.
        logging.warning("Total masked phases: {}".format(np.sum(data_mask)))

        uvw = UVW(location=datapack.radio_array.get_center(), obstime=times[0],
              phase=datapack.get_center_direction())
        dirs_uvw = directions.transform_to(uvw)
        #already centered on zero
        d = np.array([np.arctan2(dirs_uvw.u.value, dirs_uvw.w.value),
                     np.arctan2(dirs_uvw.v.value, dirs_uvw.w.value)]).T
        t = times.gps
        f = freqs
        directional_sampling = 1
        time_sampling = 2
        freq_sampling = 1#(Nf >> 1 ) + 1
        directional_slice = slice(0,Nd,directional_sampling)
        freq_slice = slice(0,Nf,freq_sampling)
#        spatial_scale = np.mean(d**2)
#        d /= spatial_scale
#        
#        t = times.gps[:interval]
#        t -= np.mean(t)
#        time_scale = np.max(t) - np.min(t)
#        t /= time_scale
#        
#        f = freqs - np.mean(freqs)
#        freq_scale = np.max(f) - np.min(f)
#        f /= freq_scale
#
#        logging.warning('Spatial scale: {}'.format(spatial_scale))
#        logging.warning('Time scale: {}'.format(time_scale))
#        logging.warning('Freq scale: {}'.format(freq_scale))
#
#        ###
#        # Create model coords for learning
#
        
#
#        time_slice = slice(0,interval,time_sampling)
        
#
#        t_s = t[time_slice]
#        d_s = d[directional_slice,:]
#        f_s = f[freq_slice]
#
#        Nt_s = t_s.shape[0]
#        Nd_s = d_s.shape[0]
#        Nf_s = f_s.shape[0]
#        
#        X_s = self._make_coord_array(t_s,d_s,f_s)

        lock = Lock()

        with futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            jobs = []
            
            for i,ai in enumerate(ant_idx):
                for j,aj in enumerate(time_idx[::shift]):
                    start = j*shift
                    stop = min(start+interval,Nt)
                    time_slice = slice(start,stop,time_sampling)
                    
                    jobs.append(executor.submit(
                        Smoothing._solve_block,
                        phase[i,time_slice,directional_slice,freq_slice],
                        error[i,time_slice,directional_slice,freq_slice],
                        (t[time_slice],d[directional_slice],f[freq_slice]),
                        lock,
                        pargs="Working on {} time chunk ({}) {} to ({}) {}".format(antenna_labels[i],
                            start,timestamps[start],stop-1,timestamps[stop-1]),
                        verbose=verbose
                        )
                        )
            ref_dist = []
            for j,aj in enumerate(time_idx[::shift]):
                start = j*shift
                stop = min(start+interval,Nt)
                ref_dist.append(self._ref_distance(
                        UVW(location=datapack.radio_array.get_center(), 
                            obstime=times[(start+stop)//2],
                            phase=datapack.get_center_direction()),
                        antennas, i0=0))
            ref_dist = np.stack(ref_dist,axis=0)
            results = futures.wait(jobs)
            if verbose:
                logging.warning(results)
            Nt_ = len(jobs)//Na
            kern_lengthscales = np.zeros([Na,Nt_,3])
            kern_variances = np.zeros([Na,Nt_,1])
            mean_time = np.zeros(Nt_)
            results = [j.result() for j in jobs]
            res_idx = 0
            for i,ai in enumerate(ant_idx):
                for j,aj in enumerate(time_idx[::shift]):
                    start = j*interval
                    stop = min((j+1)*interval,Nt)
                    time_slice = slice(start,stop,time_sampling)
                    mean_time[j] = np.mean(t[time_slice])
                    res = results[res_idx]
                    res_idx += 1
                    kern_lengthscales[i,j,:] = res[0]
                    kern_variances[i,j,0] = res[1]
            np.savez(save_file,**{"kern_ls":kern_lengthscales,"kern_var":kern_variances,"time":mean_time,"antenna":antenna_labels,"ref_dist":ref_dist})

    
if __name__=='__main__':
    import os

    if len(sys.argv) == 2:
        starting_datapack = sys.argv[1]
    else:
        starting_datapack = "../data/rvw_datapack_dd_phase_dec27_SB200-219_unwrap.hdf5"
    smoothing = Smoothing(starting_datapack)
    smoothing.solve_time_intervals("gp_params.npz",-1,-1,-1,range(0,20,10),32,150,num_threads=12,verbose=True)
