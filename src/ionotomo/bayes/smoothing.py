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
import astropy.coordinates as ac
import astropy.units as au
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
from collections import deque

from ionotomo.bayes.gpflow_contrib import GPR_v2,Gaussian_v2
from scipy.cluster.vq import kmeans2



class Smoothing(object):
    """
    Class for all types of GP smoothing/conditioned prediction
    """

    def __init__(self,datapack):
        if isinstance(datapack, str):
            datapack = DataPack(filename=datapack)
        self.datapack = datapack

    def _make_coord_array(t,d,f):
        """Static method to pack coordinates
        """
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


    def _solve_block_svgp(phase, error, coords, lock, init=(0.1,0.2,10.),pargs=None,verbose=False):
        try:
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
            d_scale = np.std(d - np.mean(d,axis=0)) + 1e-6
            f_scale = np.max(f) - np.min(f) + 1e-6
            t = (t - np.mean(t))/(t_scale+1e-6)
            d = (d - np.mean(d,axis=0))/(d_scale+1e-6)
            f = (f - np.mean(f))/(f_scale+1e-6)
            X = Smoothing._make_coord_array(t,d,f)

            M = 50
            Z = kmeans2(X, M, minit='points')[0]

            with tf.Session(graph=tf.Graph()) as sess:
                lock.acquire()
                try:
                    with gp.defer_build():
                        k_space = gp.kernels.RBF(2,active_dims = [0,1],lengthscales=[init[0]])
                        k_time = gp.kernels.RBF(1,active_dims = [2],lengthscales=[init[1]])
                        k_freq = gp.kernels.RBF(1,active_dims = [3], lengthscales=[init[2]])
                        
                        #k_white = gp.kernels.White(4)
                        kern = k_space * k_time * k_freq# + k_white
                        mean = gp.mean_functions.Zero()#Constant()

                        m = gp.models.svgp.SVGP(X, y, kern, mean_function = mean, 
                                likelihood=Gaussian_v2(Y_var=var, trainable=False), 
                                Z=Z, num_latent=1, minibatch_size=100, whiten=True)
                        m.feature.set_trainable(False)
                        m.kern.rbf_1.lengthscales.prior = gp.priors.Gaussian(1./d_scale,0.5/d_scale)
                        m.kern.rbf_2.lengthscales.prior = gp.priors.Gaussian(0,1./3.)
                        m.kern.rbf_3.lengthscales.set_trainable(False)
                        m.compile()
                finally:
                    lock.release()
                iterations=150
                gp.train.AdamOptimizer(0.1).minimize(m, maxiter=iterations)
                
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
        except Exception as e:
            print(e)


    def _solve_block(phase, error, coords, lock, pargs=None,verbose=False):
        try:
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
            d_scale = np.std(d - np.mean(d,axis=0)) + 1e-6
            f_scale = np.max(f) - np.min(f) + 1e-6
            t = (t - np.mean(t))/(t_scale+1e-6)
            d = (d - np.mean(d,axis=0))/(d_scale+1e-6)
            f = (f - np.mean(f))/(f_scale+1e-6)
            X = Smoothing._make_coord_array(t,d,f)

            ###
            # stationary points
            d_slice = np.s_[:]
            t_slice = np.s_[len(t)>>1:(len(t)>>1) + 10]
            f_slice = np.s_[len(f)>>1:(len(f)>>1)+1]

            Z = Smoothing._make_coord_array(t[t_slice],d[d_slice,:],f[f_slice])
            Zy = ((phase[t_slice,d_slice,f_slice] - y_mean)/y_scale).flatten()[:,None]
            Zvar = (error[t_slice,d_slice,f_slice].flatten() / y_scale * error_scale)**2

            with tf.Session(graph=tf.Graph()) as sess:
                lock.acquire()
                try:
                    with gp.defer_build():
                        k_space = gp.kernels.RBF(2,active_dims = [0,1],lengthscales=[0.1])
                        k_time = gp.kernels.RBF(1,active_dims = [2],lengthscales=[0.25])
                        k_freq = gp.kernels.RBF(1,active_dims = [3], lengthscales=[10.0])
                        
                        #k_white = gp.kernels.White(4)
                        kern = k_space * k_time * k_freq# + k_white
                        mean = gp.mean_functions.Constant()

                        m = GPR_v2(X, y, kern, Z=Z,Zy=Zy,Zvar=Zvar, mean_function=mean,var=var,trainable_var=False, minibatch_size=400)
                        m.kern.rbf_3.lengthscales.set_trainable(False)
                        m.compile()
                finally:
                    lock.release()
                o = gp.train.ScipyOptimizer(method='BFGS')
                #o = gp.train.AdamOptimizer(0.01)
                sess = m.enquire_session()
                with sess.as_default():
                    marginal_log_likelihood = [m.objective.eval()]
                    for i in range(3):
                        o.minimize(m,maxiter=4)
                        #marginal_log_likelihood.append(m.objective.eval())
                        #print(marginal_log_likelihood[-1])
                    #plt.plot(marginal_log_likelihood)
                    #plt.show()

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
        except Exception as e:
            print(e)

    def _ref_distance(self,antennas,i0=0):
        x = antennas.x.to(au.km).value
        y = antennas.y.to(au.km).value
        z = antennas.z.to(au.km).value
        dist = np.sqrt((x-x[i0])**2 + (y-y[i0])**2 + (z-z[i0])**2)
        return dist

    def refine_statistics_timeonly(self,results_file):
        antennas,antenna_labels = self.datapack.get_antennas(-1)
        data = np.load(results_file)
        length_scales = data['kern_ls']
        length_scales -= length_scales.mean(0).mean(0)
        length_scales /= length_scales.std(0).std(0)
        var_scale = data['kern_var']
        times = data['time']
        times -= times.mean()
        times /= times.std()
        labels = data['antenna']
        array_center = ac.ITRS(np.mean(antennas.data))
        enu = ENU(location = array_center)
        ants_enu = antennas.transform_to(enu)
        positions = np.array([ants_enu.east.to(au.km).value[1:], ants_enu.north.to(au.km).value[1:]]).T
        positions -= positions.mean(0)
        positions /= positions.std(0).mean()

        Nt,Np = times.shape[0],positions.shape[0]
        X = np.zeros([Nt,Np,1],dtype=np.float64)
        for j in range(Nt):
            for k in range(Np):
                X[j,k,0] = times[j]
             #   X[j,k,1:3] = positions[k,:]   
        X = np.reshape(X,(Nt*Np,1))
        Y = length_scales[:,:,0:1].transpose([1,0,2]).reshape((-1,1))

        M = 100
        Z = kmeans2(X, M, minit='points')[0]

        with tf.Session(graph=tf.Graph()) as sess:
            with gp.defer_build():
                #k_space = gp.kernels.RBF(2,active_dims = [1,2],lengthscales=[0.5])
                k_time = gp.kernels.RBF(1,active_dims = [0],lengthscales=[0.5])
                k_white = gp.kernels.White(1,variance=1e-5)
                kern = k_time  + k_white
                mean = gp.mean_functions.Constant()

                m = gp.models.svgp.SVGP(X, Y, kern, mean_function = mean, 
                        likelihood=gp.likelihoods.Gaussian(), 
                        Z=Z, num_latent=1, minibatch_size=500, whiten=True)
                m.feature.set_trainable(False)
                k_time.lengthscales.prior = gp.priors.Gaussian(0,1/3.)
                m.compile()
            iterations=1000
            gp.train.AdamOptimizer(0.01).minimize(m, maxiter=iterations)
            print(m)
            xi = np.linspace(np.min(positions),np.max(positions),100)
            yi = np.linspace(np.min(positions),np.max(positions),100)
            y,var = m.predict_y(X)
            y = y.reshape((Nt,Np,1))
            std = np.sqrt(var.reshape((Nt,Np,1)))
            fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)
            [ax1.plot(times,y[:,i,0]) for i in range(y.shape[1])]
            ax1.plot(times,length_scales[:,:,0].mean(0),lw=2,c='black')
            ax1.fill_between(times,y[:,0,0]
            ax1.fill_between(times,length_scales[:,:,0].mean(0)+length_scales[:,:,0].std(0),length_scales[:,:,0].mean(0)-length_scales[:,:,0].std(0),alpha=0.25)
            [ax2.plot(times,length_scales[i,:,0]) for i in range(y.shape[1])]
            plt.show()

        Y = length_scales[:,:,1:2].transpose([1,0,2]).reshape((-1,1))

        M = 100
        Z = kmeans2(X, M, minit='points')[0]

        with tf.Session(graph=tf.Graph()) as sess:
            with gp.defer_build():
                k_time = gp.kernels.RBF(1,active_dims = [0],lengthscales=[0.5])
                k_white = gp.kernels.White(1,variance=1e-5)
                kern = k_time  + k_white
                mean = gp.mean_functions.Constant()

                m = gp.models.svgp.SVGP(X, Y, kern, mean_function = mean, 
                        likelihood=gp.likelihoods.Gaussian(), 
                        Z=Z, num_latent=1, minibatch_size=500, whiten=True)
                m.feature.set_trainable(False)
                k_time.lengthscales.prior = gp.priors.Gaussian(0,1/3.)
                m.compile()
            iterations=1000
            gp.train.AdamOptimizer(0.01).minimize(m, maxiter=iterations)
            print(m)
            xi = np.linspace(np.min(positions),np.max(positions),100)
            yi = np.linspace(np.min(positions),np.max(positions),100)
            y,var = m.predict_y(X)
            y = y.reshape((Nt,Np,1))
            std = np.sqrt(var.reshape((Nt,Np,1)))
            fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)
            [ax1.plot(times,y[:,i,0]) for i in range(y.shape[1])]
            ax1.plot(times,length_scales[:,:,1].mean(0),lw=2,c='black')
            ax1.fill_between(times,length_scales[:,:,1].mean(0)+length_scales[:,:,1].std(0),length_scales[:,:,1].mean(0)-length_scales[:,:,1].std(0),alpha=0.25)
            [ax2.plot(times,length_scales[i,:,1]) for i in range(y.shape[1])]
            plt.show()

        return




    def refine_statistics(self,results_file):
        from scipy.interpolate import griddata        
        #radio_array = RadioArray(array_file=RadioArray.lofar_array)
        antennas,antenna_labels = self.datapack.get_antennas(-1)
        data = np.load(results_file)
        length_scales = data['kern_ls']
        length_scales -= length_scales.mean(0).mean(0)
        length_scales /= length_scales.std(0).std(0)
        var_scale = data['kern_var']
        times = data['time']
        times -= times.mean()
        times /= times.std()
        labels = data['antenna']
        array_center = ac.ITRS(np.mean(antennas.data))
        enu = ENU(location = array_center)
        ants_enu = antennas.transform_to(enu)
        positions = np.array([ants_enu.east.to(au.km).value[1:], ants_enu.north.to(au.km).value[1:]]).T
        positions -= positions.mean(0)
        positions /= positions.std(0).mean()

        Nt,Np = times.shape[0],positions.shape[0]
        X = np.zeros([Nt,Np,3],dtype=np.float64)
        for j in range(Nt):
            for k in range(Np):
                X[j,k,0] = times[j]
                X[j,k,1:3] = positions[k,:]   
        X = np.reshape(X,(Nt*Np,3))
        Y = length_scales[:,:,0:1].transpose([1,0,2]).reshape((-1,1))

        M = 100
        Z = kmeans2(X, M, minit='points')[0]

        with tf.Session(graph=tf.Graph()) as sess:
            with gp.defer_build():
                k_space = gp.kernels.RBF(2,active_dims = [1,2],lengthscales=[0.5])
                k_time = gp.kernels.RBF(1,active_dims = [0],lengthscales=[0.5])
                k_white = gp.kernels.White(3,variance=1e-5)
                kern = k_time * k_space + k_white
                mean = gp.mean_functions.Constant()

                m = gp.models.svgp.SVGP(X, Y, kern, mean_function = mean, 
                        likelihood=gp.likelihoods.Gaussian(), 
                        Z=Z, num_latent=1, minibatch_size=500, whiten=True)
                m.feature.set_trainable(False)
                k_space.lengthscales.prior = gp.priors.Gaussian(0,1./3.)
                k_time.lengthscales.prior = gp.priors.Gaussian(0,1/3.)
                m.compile()
            iterations=1000
            gp.train.AdamOptimizer(0.01).minimize(m, maxiter=iterations)
            print(m)
            xi = np.linspace(np.min(positions),np.max(positions),100)
            yi = np.linspace(np.min(positions),np.max(positions),100)
            y,var = m.predict_y(X)
            y = y.reshape((Nt,Np,1))
            std = np.sqrt(var.reshape((Nt,Np,1)))
            fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)
            [ax1.plot(times,y[:,i,0]) for i in range(y.shape[1])]
            ax1.plot(times,length_scales[:,:,0].mean(0),lw=2,c='black')
            ax1.fill_between(times,length_scales[:,:,0].mean(0)+length_scales[:,:,0].std(0),length_scales[:,:,0].mean(0)-length_scales[:,:,0].std(0),alpha=0.25)
            [ax2.plot(times,length_scales[i,:,0]) for i in range(y.shape[1])]
            plt.show()

        Y = length_scales[:,:,1:2].transpose([1,0,2]).reshape((-1,1))

        M = 100
        Z = kmeans2(X, M, minit='points')[0]

        with tf.Session(graph=tf.Graph()) as sess:
            with gp.defer_build():
                k_space = gp.kernels.RBF(2,active_dims = [1,2],lengthscales=[0.5])
                k_time = gp.kernels.RBF(1,active_dims = [0],lengthscales=[0.5])
                k_white = gp.kernels.White(3,variance=1e-5)
                kern = k_time * k_space + k_white
                mean = gp.mean_functions.Constant()

                m = gp.models.svgp.SVGP(X, Y, kern, mean_function = mean, 
                        likelihood=gp.likelihoods.Gaussian(), 
                        Z=Z, num_latent=1, minibatch_size=500, whiten=True)
                m.feature.set_trainable(False)
                k_space.lengthscales.prior = gp.priors.Gaussian(0,1./3.)
                k_time.lengthscales.prior = gp.priors.Gaussian(0,1/3.)
                m.compile()
            iterations=1000
            gp.train.AdamOptimizer(0.01).minimize(m, maxiter=iterations)
            print(m)
            xi = np.linspace(np.min(positions),np.max(positions),100)
            yi = np.linspace(np.min(positions),np.max(positions),100)
            y,var = m.predict_y(X)
            y = y.reshape((Nt,Np,1))
            std = np.sqrt(var.reshape((Nt,Np,1)))
            fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2)
            [ax1.plot(times,y[:,i,0]) for i in range(y.shape[1])]
            ax1.plot(times,length_scales[:,:,1].mean(0),lw=2,c='black')
            ax1.fill_between(times,length_scales[:,:,1].mean(0)+length_scales[:,:,1].std(0),length_scales[:,:,1].mean(0)-length_scales[:,:,1].std(0),alpha=0.25)
            [ax2.plot(times,length_scales[i,:,1]) for i in range(y.shape[1])]
            plt.show()

        return




        
        # grid the data.
        zi = griddata((positions[:,0], positions[:,1]), np.log10(length_scales[:,0,0]), (xi[None,:], yi[:,None]), method='cubic')
        plt.ion()
        # contour the gridded data, plotting dots at the randomly spaced data points.
        fig,ax = plt.subplots(1,1)
        plt.show()
        i = 0
        vmin = np.min(length_scales[:,:,0])
        vmax = np.max(length_scales[:,:,0])
        set = False
        while True:
            idx = i % len(times)
            zi = griddata((positions[:,0], positions[:,1]), (length_scales[:,idx,0]), (xi[None,:], yi[:,None]), method='nearest')
            ax.cla()
            CS1 = ax.contour(xi,yi,zi,15,linewidths=0.5,colors='k')
            CS2 = ax.contourf(xi,yi,zi,15,cmap=plt.cm.jet,vmin=vmin,vmax=vmax)
            if not set:
                plt.colorbar(CS2) # draw colorbar
                set = True
            # plot data points.
            plt.scatter(positions[:,0],positions[:,1],marker='o',c='b',s=5)
            plt.title(times[idx])
            fig.canvas.draw()
            i += 1
        plt.ioff()

#        plt.xlim(-2,2)
#        plt.ylim(-2,2)
#        plt.title('griddata test (%d points)' % npts)
        #plt.show()


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
#        d = np.array([np.arctan2(dirs_uvw.u.value, dirs_uvw.w.value),
#                     np.arctan2(dirs_uvw.v.value, dirs_uvw.w.value)]).T
        d = np.array([directions.ra.deg, directions.dec.deg]).T
        t = times.gps
        f = freqs
        directional_sampling = 1
        time_sampling = 1
        freq_sampling = 1
        directional_slice = slice(0,Nd,directional_sampling)
        freq_slice = slice(0,Nf,freq_sampling)

        lock = Lock()

        with futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            jobs = []
            
            for i,ai in enumerate(ant_idx):
                for j,aj in enumerate(time_idx[::shift]):
                    start = j*shift
                    stop = min(start+interval,Nt)
                    time_slice = slice(start,stop,time_sampling)
                    
                    jobs.append(executor.submit(
                        Smoothing._solve_block_svgp,
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
        starting_datapack = "../data/rvw_datapack_full_phase_dec27.hdf5"
    smoothing = Smoothing(starting_datapack)
    #smoothing.solve_time_intervals("gp_params.npz",range(1,62),-1,-1,range(0,20),32,32,num_threads=16,verbose=True)
    smoothing.refine_statistics_timeonly('gp_params.npz')
